import argparse
from PIL import Image, ImageDraw
from omegaconf import OmegaConf
from GLIGEN.ldm.models.diffusion.ddim import DDIMSampler
from GLIGEN.ldm.models.diffusion.plms import PLMSSampler
import os
from transformers import CLIPProcessor, CLIPModel
import torch
from GLIGEN.ldm.util import instantiate_from_config
from GLIGEN.trainer import read_official_ckpt, batch_to_device
import numpy as np
import clip
from scipy.io import loadmat
from functools import partial
import GLIGEN.dist as gdist
import torchvision.transforms.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from tqdm import tqdm
from termcolor import colored
import gc
from pycocotools.coco import COCO
import os
from PIL import Image
from GLIGEN.coco_annotations import coco_loader, filter_annotations_and_images, make_meta_dict, resize_annotations
from tqdm import tqdm
import torch.distributed as dist
from functools import lru_cache

device = "cuda"


def set_alpha_scale(model, alpha_scale):
    from GLIGEN.ldm.modules.attention import GatedCrossAttentionDense, GatedSelfAttentionDense
    for module in model.modules():
        if type(module) == GatedCrossAttentionDense or type(module) == GatedSelfAttentionDense:
            module.scale = alpha_scale


def alpha_generator(length, type=None):
    """
    length is total timestpes needed for sampling.
    type should be a list containing three values which sum should be 1

    It means the percentage of three stages:
    alpha=1 stage
    linear deacy stage
    alpha=0 stage.

    For example if length=100, type=[0.8,0.1,0.1]
    then the first 800 stpes, alpha will be 1, and then linearly decay to 0 in the next 100 steps,
    and the last 100 stpes are 0.
    """
    if type == None:
        type = [1,0,0]

    assert len(type)==3
    assert type[0] + type[1] + type[2] == 1

    stage0_length = int(type[0]*length)
    stage1_length = int(type[1]*length)
    stage2_length = length - stage0_length - stage1_length

    if stage1_length != 0:
        decay_alphas = np.arange(start=0, stop=1, step=1/stage1_length)[::-1]
        decay_alphas = list(decay_alphas)
    else:
        decay_alphas = []


    alphas = [1]*stage0_length + decay_alphas + [0]*stage2_length

    assert len(alphas) == length

    return alphas


def load_ckpt(ckpt_path):

    saved_ckpt = torch.load(ckpt_path)
    config = saved_ckpt["config_dict"]["_content"]
    config['model'].target = 'GLIGEN.' + config['model'].target
    config['model'].params["grounding_tokenizer"]["target"] = 'GLIGEN.' + config['model'].params["grounding_tokenizer"].get('target')
    config['autoencoder'].target = 'GLIGEN.' + config['autoencoder'].target
    config['text_encoder'].target = 'GLIGEN.' + config['text_encoder'].target
    config['diffusion'].target = 'GLIGEN.' + config['diffusion'].target

    model = instantiate_from_config(config['model']).to(device).eval()
    autoencoder = instantiate_from_config(config['autoencoder']).to(device).eval()
    text_encoder = instantiate_from_config(config['text_encoder']).to(device).eval()
    diffusion = instantiate_from_config(config['diffusion']).to(device)

    # do not need to load official_ckpt for self.model here, since we will load from our ckpt
    model.load_state_dict( saved_ckpt['model'] )
    autoencoder.load_state_dict( saved_ckpt["autoencoder"]  )
    text_encoder.load_state_dict( saved_ckpt["text_encoder"], strict=False )
    diffusion.load_state_dict( saved_ckpt["diffusion"]  )

    return model, autoencoder, text_encoder, diffusion, config

def load_config(ckpt_path):
    saved_ckpt = torch.load(ckpt_path)
    config = saved_ckpt["config_dict"]["_content"]
    config['model'].target = 'GLIGEN.' + config['model'].target
    config['model'].params["grounding_tokenizer"]["target"] = 'GLIGEN.' + config['model'].params["grounding_tokenizer"].get('target')
    config['autoencoder'].target = 'GLIGEN.' + config['autoencoder'].target
    config['text_encoder'].target = 'GLIGEN.' + config['text_encoder'].target
    config['diffusion'].target = 'GLIGEN.' + config['diffusion'].target


    model = instantiate_from_config(config['model']).to(device).eval()
    autoencoder = instantiate_from_config(config['autoencoder']).to(device).eval()
    text_encoder = instantiate_from_config(config['text_encoder']).to(device).eval()
    diffusion = instantiate_from_config(config['diffusion']).to(device)
    return model, autoencoder, text_encoder, diffusion, config

def create_empty_models(ckpt_path):
    saved_ckpt = torch.load(ckpt_path, map_location='cpu')
    config = saved_ckpt["config_dict"]["_content"]
    config['model'].target = 'GLIGEN.' + config['model'].target
    config['model'].params["grounding_tokenizer"]["target"] = 'GLIGEN.' + config['model'].params["grounding_tokenizer"].get('target')
    config['autoencoder'].target = 'GLIGEN.' + config['autoencoder'].target
    config['text_encoder'].target = 'GLIGEN.' + config['text_encoder'].target
    config['diffusion'].target = 'GLIGEN.' + config['diffusion'].target


    model = instantiate_from_config(config['model']).to(device).eval()
    autoencoder = instantiate_from_config(config['autoencoder']).to(device).eval()
    text_encoder = instantiate_from_config(config['text_encoder']).to(device).eval()
    diffusion = instantiate_from_config(config['diffusion']).to(device)

    return model, autoencoder, text_encoder, diffusion, config

def project(x, projection_matrix):
    """
    x (Batch*768) should be the penultimate feature of CLIP (before projection)
    projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer
    defined in CLIP (out_dim, in_dim), thus we need to apply transpose below.
    this function will return the CLIP feature (without normalziation)
    """
    return x@torch.transpose(projection_matrix, 0, 1)

def get_clip_feature(model, processor, input, is_image=False):
	which_layer_text = 'before'
	which_layer_image = 'after_reproject'

	if is_image:
		if input == None:
			return None
		image = Image.open(input).convert("RGB")
		inputs = processor(images=[image],  return_tensors="pt", padding=True)
		inputs['pixel_values'] = inputs['pixel_values'].cuda() # we use our own preprocessing without center_crop 
		inputs['input_ids'] = torch.tensor([[0,1,2,3]]).cuda()  # placeholder
		outputs = model(**inputs)
		feature = outputs.image_embeds 
		if which_layer_image == 'after_reproject':
			feature = project( feature, torch.load('GLIGEN/projection_matrix').cuda().T ).squeeze(0)
			feature = ( feature / feature.norm() )  * 28.7 
			feature = feature.unsqueeze(0)
	else:
		if input == None:
			return None
		inputs = processor(text=input,  return_tensors="pt", padding=True)
		inputs['input_ids'] = inputs['input_ids'].cuda()
		inputs['pixel_values'] = torch.ones(1,3,224,224).cuda() # placeholder 
		inputs['attention_mask'] = inputs['attention_mask'].cuda()
		outputs = model(**inputs)
		if which_layer_text == 'before':
			feature = outputs.text_model_output.pooler_output
	return feature



def complete_mask(has_mask, max_objs):
    mask = torch.ones(1,max_objs)
    if has_mask == None:
        return mask

    if type(has_mask) == int or type(has_mask) == float:
        return mask * has_mask
    else:
        for idx, value in enumerate(has_mask):
            mask[0,idx] = value
        return mask



@torch.no_grad()
def prepare_batch(args, meta_list, model, processor, max_objs=30):
	prompt_list = []
	image_ids = []
	out_list = []
	coco_path = args.coco_path
	for meta in meta_list:
		phrases, images = meta.get("phrases"), meta.get("image_id")

		if images is not None :
			img_path = os.path.join(coco_path + "train2017", images + ".jpg")
			images = img_path
		prompt_list.append(meta["prompt"])
		image_ids.append(meta.get("image_id"))

		images = [None]*len(phrases) if images==None else [images] * len(phrases)
		phrases = [None]*len(images) if phrases==None else phrases 

		boxes = torch.zeros(max_objs, 4)
		masks = torch.zeros(max_objs)
		text_masks = torch.zeros(max_objs)
		image_masks = torch.zeros(max_objs)
		text_embeddings = torch.zeros(max_objs, 768)
		image_embeddings = torch.zeros(max_objs, 768)

		text_features = [get_clip_feature(model, processor, phrase, is_image=False) for phrase in phrases]
		image_features = [get_clip_feature(model, processor, image, is_image=True) for image in images]

		for idx, (box, text_feature, image_feature) in enumerate(zip(meta['locations'], text_features, image_features)):
			boxes[idx] = torch.tensor(box)
			masks[idx] = 1
			if text_feature is not None:
				text_embeddings[idx] = text_feature
				text_masks[idx] = 1 
			if image_feature is not None:
				image_embeddings[idx] = image_feature
				image_masks[idx] = 1 

		out = {
			"boxes": boxes.unsqueeze(0),
			"masks": masks.unsqueeze(0),
			"text_masks": text_masks.unsqueeze(0) * complete_mask(meta.get("text_mask"), max_objs),
			"image_masks": image_masks.unsqueeze(0) * complete_mask(meta.get("image_mask"), max_objs),
			"text_embeddings": text_embeddings.unsqueeze(0),
			"image_embeddings": image_embeddings.unsqueeze(0)
		}
		out_list.append(out)

	# Concatenate along the first dimension (batch dimension) to stack multiple batches together
	final_output = {
		key: torch.cat([out[key] for out in out_list], dim=0)
		for key in out_list[0].keys()
	}
	del text_features, image_features, text_masks, image_masks
	return batch_to_device(final_output, device), prompt_list, image_ids


def crop_and_resize(image):
    crop_size = min(image.size)
    image = TF.center_crop(image, crop_size)
    image = image.resize( (512, 512) )
    return image

def colorEncode(labelmap, colors):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)

    for label in np.unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    return labelmap_rgb

@torch.no_grad()
def prepare_batch_sem(meta, batch=1):

    pil_to_tensor = transforms.PILToTensor()

    sem = Image.open( meta['sem']  ).convert("L") # semantic class index 0,1,2,3,4 in uint8 representation
    sem = TF.center_crop(sem, min(sem.size))
    sem = sem.resize( (512, 512), Image.NEAREST ) # acorrding to official, it is nearest by default, but I don't know why it can prodice new values if not specify explicitly
    try:
        sem_color = colorEncode(np.array(sem), loadmat('color150.mat')['colors'])
        Image.fromarray(sem_color).save("sem_vis.png")
    except:
        pass
    sem = pil_to_tensor(sem)[0,:,:]
    input_label = torch.zeros(152, 512, 512)
    sem = input_label.scatter_(0, sem.long().unsqueeze(0), 1.0)

    out = {
        "sem" : sem.unsqueeze(0).repeat(batch,1,1,1),
        "mask" : torch.ones(batch,1),
    }
    return batch_to_device(out, device)

def check_overlap(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # 겹치는 조건을 확인
    return x1_1 < x2_2 and x2_1 > x1_2 and y1_1 < y2_2 and y2_1 > y1_2

def broadcast_model(model):
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

def check_image_exists(image_id, save_folder):
    "generate images from stopped sign to end generation"
    image_path = os.path.join(save_folder, f"{image_id}.jpg")
    return os.path.exists(image_path)

def generate_new_meta_based_on_insufficient_objects(meta, meta_list, insufficient_objects):
    new_meta = deepcopy(meta)
    
    insufficient_key = next(iter(insufficient_objects.keys()))
    
    # insufficient_key를 포함하는 meta 정보를 meta_list에서 찾기
    matching_metas = [meta for meta in meta_list if insufficient_key in meta["phrases"]]
    if not matching_metas:
        print(f"No matching meta found for key: {insufficient_key}")
        matching_meta = None
    else:
        matching_meta = random.choice(matching_metas)  # 랜덤으로 하나 뽑기
    
    # 중앙에 bbox 설정
    central_bbox = [0.3, 0.3, 0.6, 0.6]  
    
    # 새로운 meta 정보 업데이트
    new_meta["phrases"] = [insufficient_key]
    new_meta["locations"] = [central_bbox]
    new_meta["prompt"] = construct_prompt_from_phrases([insufficient_key])
    new_meta["image_id"] = None if not matching_meta else matching_meta["image_id"]
    # new_meta["image_id"] = None
    
    insufficient_objects[insufficient_key] -= 1
    if insufficient_objects[insufficient_key] <= 0:
        del insufficient_objects[insufficient_key]
        
    return new_meta

import random
from GLIGEN.coco_annotations import construct_prompt_from_phrases, refine_meta
from copy import deepcopy
def modify_meta_for_insufficient_objects(meta_list, insufficient_objects, args, count=0):
    """
    Modify the meta information to include more of the insufficient objects.

    Parameters:
    - meta_list: List of meta information.
    - insufficient_objects: Dictionary containing category names and the number of instances required.
    Returns:
    - modified_meta_list: Modified meta list.
    """
    modified_meta_list = []
    if count >= 10 : 
        meta = meta_list[0]
        total_gen_length = sum(insufficient_objects.values())
        for _ in tqdm(range(total_gen_length), desc="reformatting prompt and bbox"):
            new_meta = generate_new_meta_based_on_insufficient_objects(meta, meta_list, insufficient_objects)
            modified_meta_list.append(new_meta)
        
        print(f"left generation item counts : {insufficient_objects}")
        return modified_meta_list
        
    for meta in meta_list:
        modified_meta = deepcopy(meta)
        modified_meta, insufficient_objects = refine_meta(args.object_counts, insufficient_objects, modified_meta, True, count)
        if modified_meta is None :
            continue
        modified_meta_list.append(modified_meta)

    # Find keys in insufficient_objects where value is not 0
    insufficient_keys = [key for key, val in insufficient_objects.items() if val > 0]
    
    if insufficient_keys :
        print(colored(f"not complete make insufficient keys, so repeat some meta information for matching requred counts"))
    
    # For each insufficient key, find metas that have that key in their phrases and replicate them
    for ins_key in insufficient_keys:
        matching_metas = [meta for meta in meta_list if ins_key in meta["phrases"]]
        
        # If there are no matching metas for the ins_key, remove the key from the dictionary
        if not matching_metas:
            insufficient_objects.pop(ins_key, None)
            print(colored(f"{ins_key} is deleted. not matching data"))
            continue
        
        if matching_metas:
            # Replicate the chosen meta for the insufficient count
            for _ in range(insufficient_objects[ins_key]):
                chosen_meta = random.choice(matching_metas)
                modified_meta = deepcopy(chosen_meta)
                modified_meta, insufficient_objects = refine_meta(args.object_counts, insufficient_objects, modified_meta, True, count)
                modified_meta_list.append(modified_meta)
                # Update the count in insufficient_objects
            
    return modified_meta_list



import sys
import os
def custom_dataset(args, sample, max_length, max_class, min_class, processor=None, model=None):
    coco_json_path = os.path.join(args.coco_path, 'annotations/instances_train2017.json')

    #* load coco json
    new_train_json = coco_loader(coco_json_path)
    new_train_json = filter_annotations_and_images(new_train_json, max_class, min_class)
    image_length = len(new_train_json["images"])

    #* ddp divide data, for multi gpu users
    rank = gdist.get_rank()
    world_size = gdist.get_world_size()
    per_image_length = image_length / world_size
    start_idx = rank * int(per_image_length)
    end_idx = start_idx + per_image_length if rank != world_size - 1 else image_length
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    my_slice = new_train_json["images"][start_idx:end_idx]

    meta_list = []
    filtered_images = []
    filtered_annotations = []
    image_bar = tqdm(total=len(my_slice), desc="Processing annotations", disable=not gdist.is_main_process())
    for image_info in my_slice:
        image_bar.update(1)
        temp_gligen_dict = make_meta_dict(args, new_train_json, sample, image_info, max_length, processor, model)



import sys
import os
from GLIGEN.coco_annotations import refine_meta
@torch.no_grad()
def run(meta_list, args, insufficient_objects=None, pre_config=None, count=0): #TODO: ++ with pseudo labeling
    load_path = "GLIGEN/gligen_checkpoints/checkpoint_generation_text_image.pth"
    original_stdout = sys.stdout
    sys.stdout = None

    if gdist.is_main_process():
        # Only the main process loads the model
        model, autoencoder, text_encoder, diffusion, config = load_ckpt(load_path)
    else:
        # Other processes create an empty model
        model, autoencoder, text_encoder, diffusion, config = create_empty_models(load_path)

    # Synchronize all processes
    if gdist.get_world_size() > 1:
        dist.barrier() #sync

        # Now, non-main processes will receive the broadcasted model weights
        broadcast_model(model)
        broadcast_model(autoencoder)
        broadcast_model(text_encoder)
        broadcast_model(diffusion)

        print(colored(f"----all gpus broadcasting complete----", "red", "on_yellow"))
        dist.barrier() #sync

    config['grounding_tokenizer_input']["target"] = "GLIGEN." + config['grounding_tokenizer_input']["target"]
    grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input'])
    model.grounding_tokenizer_input = grounding_tokenizer_input

    # - - - - - update config from args - - - - - #
    config.update( vars(args) )
    config = OmegaConf.create(config)
    sys.stdout = original_stdout
    
    #* Clip calling
    version = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(version).cuda()
    clip_processor = CLIPProcessor.from_pretrained(version)

    #! for test
    temp_meta_set = [1.0, 0.0, 0.0]
    # - - - - - sampler - - - - - #
    alpha_generator_func = partial(alpha_generator, type=temp_meta_set) #meta_list[0].get("alpha_type")
    # alpha_generator_func = partial(alpha_generator, type=meta_list[0].get("alpha_type")) #meta_list[0].get("alpha_type")
    if config.no_plms:
        sampler = DDIMSampler(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
        steps = 250
    else:
        sampler = PLMSSampler(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
        steps = 50

    if config.gen_length > len(meta_list):
        config.gen_length = len(meta_list)

    # specific_ids = ["000000011673", "000000011948", "000000013721", "000000013901", "000000015858", "000000016193", "000000016509", "000000016706", "000000016765", "000000017365"]
    if insufficient_objects is not None:
        meta_list = modify_meta_for_insufficient_objects(meta_list, insufficient_objects, args, count)
        config.gen_length = len(meta_list)

    object_counts = dict()
    #* generate images in each process (at multiGPU processes)
    gpu_working = tqdm(total=config.gen_length, desc="generation processing", disable=not gdist.is_main_process())
    temp_meta_list = []
    for idx, meta in enumerate(meta_list):
        output_folder = os.path.join(args.generator_path, "images")

        if insufficient_objects is None:
            meta, object_counts = refine_meta(args.object_counts, object_counts, meta, False)
            if meta is None:
                gpu_working.update(1)
                continue

        if check_image_exists(meta["image_id"], output_folder) and (insufficient_objects is None):
            # already generation images, so pass
            print(colored(f"already present images", "blue"))
            gpu_working.update(1)
            continue

        temp_meta_list.append(meta)
        if len(temp_meta_list) != args.gen_batch:
            continue

        #* random noise generation, #TODO: Fix here.
        starting_noise = torch.randn(args.gen_batch, 4, 64, 64).to(args.device)
        # starting_noise = None

        if idx > config.gen_length and insufficient_objects is None:
            print(colored(f"complete generate {config.gen_length} images", "blue"))
            break

        # - - - - - prepare batch - - - - - #
        batch, prompt_list, image_id_list = prepare_batch(args, temp_meta_list, clip_model, clip_processor)
        context = text_encoder.encode(prompt_list)
        uc = text_encoder.encode( args.gen_batch*[""] )
        if args.negative_prompt is not None:
            uc = text_encoder.encode( args.gen_batch*[args.negative_prompt] )


        # - - - - - inpainting related - - - - - #
        inpainting_mask = z0 = None  # used for replacing known region in diffusion process
        inpainting_extra_input = None # used as model input

        # - - - - - input for gligen - - - - - #
        grounding_input = grounding_tokenizer_input.prepare(batch)
        grounding_extra_input = None

        # - - - - - input format - - - - - - - #
        input = dict(
                    x = starting_noise,
                    timesteps = None,
                    context = context, #prompt
                    grounding_input = grounding_input, #given additional input
                    inpainting_extra_input = inpainting_extra_input,
                    grounding_extra_input = grounding_extra_input,
                )

        # - - - - - start sampling - - - - - #
        shape = (config.gen_batch, model.in_channels, model.image_size, model.image_size)
        samples_fake = sampler.sample(S=steps, shape=shape, input=input,  uc=uc, guidance_scale=config.guidance_scale, mask=inpainting_mask, x0=z0)
        samples_fake = autoencoder.decode(samples_fake)

        # - - - - - save - - - - - #
        os.makedirs( output_folder, exist_ok=True)

        start = len( os.listdir(output_folder) )
        image_ids = list(range(start,start+config.gen_batch))
        print(image_ids)

        #* save generated images to the output folder
        for _, (img_id, sample) in enumerate(zip(image_id_list, samples_fake)):
            if (insufficient_objects is not None) or (img_id is None):
                img_id = '1' + str(random.randint(10**11, 10**12 - 1))
            gdist.save_image(img_id, sample, output_folder)

        del samples_fake, batch, context, uc, sample
        #* work after generation
        gpu_working.update(config.gen_batch)
        temp_meta_list = []

        if idx % 30 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    if gdist.get_world_size() > 1:
        dist.barrier() #sync

def custom_dataset(args, sample, max_length, max_class, min_class, processor=None, model=None):
    coco_json_path = os.path.join(args.coco_path, 'annotations/instances_train2017.json')

    #* load coco json
    new_train_json = coco_loader(coco_json_path)
    new_train_json = filter_annotations_and_images(new_train_json, max_class, min_class)
    image_length = len(new_train_json["images"])

    #* ddp divide data, for multi gpu users
    rank = gdist.get_rank()
    world_size = gdist.get_world_size()
    per_image_length = image_length / world_size
    start_idx = rank * int(per_image_length)
    end_idx = start_idx + per_image_length if rank != world_size - 1 else image_length
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    my_slice = new_train_json["images"][start_idx:end_idx]

    meta_list = []
    filtered_images = []
    filtered_annotations = []
    image_bar = tqdm(total=len(my_slice), desc="Processing annotations", disable=not gdist.is_main_process())
    for image_info in my_slice:
        image_bar.update(1)
        temp_gligen_dict = make_meta_dict(args, new_train_json, sample, image_info, max_length, processor, model)

        if temp_gligen_dict is None:
            #* Remove this image_info and continue
            continue
        else:
            meta_list.append(temp_gligen_dict)
            image_info["prompt"] = temp_gligen_dict["prompt"]
            image_id = image_info['id']

            corresponding_annotations = [anno for anno in new_train_json["annotations"] if anno['image_id'] == image_id]
            resized_anns = resize_annotations(corresponding_annotations, image_info['width'], image_info['height'], 512, 512)
            image_info["height"] = 512
            image_info["width"] = 512
            filtered_images.append(image_info)
            filtered_annotations.extend(resized_anns)

    # save distrivuted meta and json information
    if gdist.get_world_size() > 1:
        dist.barrier() #sync

    new_train_json["images"] = filtered_images
    new_train_json["annotations"] = filtered_annotations

    base_path = args.generator_path
    directory_path = os.path.join(base_path, "annotations")
    gdist.save_each_file(directory_path, "train", new_train_json)
    gdist.save_each_file(directory_path, "meta", meta_list)
    if gdist.get_world_size() > 1:
        dist.barrier() #sync

    return meta_list