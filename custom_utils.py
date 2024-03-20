from ast import arg
from pyexpat import model
from sched import scheduler
from xmlrpc.client import Boolean
import torch
import util.misc as utils
import numpy as np
from typing import Tuple, Dict, List, Optional
import os
from custom_prints import over_label_checker, check_components
from termcolor import colored
from torch.utils.data import DataLoader
from datasets import build_dataset, get_coco_api_from_dataset


def new_dataLoader(saved_dict, args):
    dataset_idx_list = []
    for _, value in saved_dict.items():
        if len(value) > 0 :
            np_idx_list = np.array(value, dtype=object)
            dataset_idx_list.extend(np.unique(np_idx_list[:, 3]).astype(np.uint8).tolist())
    
    custom_dataset = build_dataset(image_set='train', args=args, img_ids=dataset_idx_list)
    
    custom_loader = DataLoader(custom_dataset, args.batch_size, shuffle=True, collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    return custom_loader


def load_model_params(mode, model: model, dir: str = None, Branch_Incremental = False):
    new_model_dict = model.state_dict()
    
    if isinstance(dir, list):
        dir = dir[0]
    #temp dir
    checkpoint = torch.load(dir)
    pretraind_model = checkpoint["model"]
    name_list = [name for name in new_model_dict.keys() if name in pretraind_model.keys()]

    if mode != 'eval' and Branch_Incremental:
        name_list = list(filter(lambda x : "class" not in x, name_list))
        name_list = list(filter(lambda x : "label" not in x, name_list)) # for dn_detr
    pretraind_model_dict = {k : v for k, v in pretraind_model.items() if k in name_list } # if "class" not in k => this method used in diff class list
    
    new_model_dict.update(pretraind_model_dict)
    model.load_state_dict(new_model_dict)
    print(colored(f"pretrained Model loading complete: {dir}", "blue", "on_yellow"))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    #No parameter update    
    for name, params in model.named_parameters():
        if name in pretraind_model_dict.keys():
            if mode == "teacher":
                params.requires_grad = False #if you wanna set frozen the pre parameters for specific Neuron update, so then you could set False
        else:
            if mode == "teacher":
                params.requires_grad = False
    
    print(colored(f"Done every model params", "red", "on_yellow"))
            
    return model

def teacher_model_freeze(model):
    for _, params in model.named_parameters():
            params.requires_grad = False
                
    return model

def save_model_params(model_without_ddp, optimizer, lr_scheduler, args, output_dir, task_index, total_tasks, epoch=-1):
    """Save model parameters for each task."""
    
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    
    # Create directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Determine the checkpoint file name based on task and epoch
    checkpoint_filename = f'cp_{total_tasks:02}_{task_index + 1:02}'
    if epoch != -1:
        checkpoint_filename += f'_{epoch}'
    checkpoint_filename += '.pth'
    
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    
    # Save model and other states
    utils.save_on_master({
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'args': args,
    }, checkpoint_path)


import torch.distributed as dist
def check_training_gpu(train_check):
    world_size = utils.get_world_size()
    
    
    if world_size < 2:
        return True
    
    gpu_control_value = torch.tensor(1.0, device=torch.device("cuda"))
    temp_list = [torch.tensor(0.0, device=torch.device("cuda")) for _ in range(4)]
    
    if train_check == False:
        gpu_control_value = torch.tensor(0.0, device=torch.device("cuda"))
        
    dist.all_gather(temp_list, gpu_control_value)
    gpu_control_value = sum([ten_idx.item() for ten_idx in temp_list])
    print(f"used gpu counts : {int(gpu_control_value)}")
    if int(gpu_control_value) == 0:
        print("current using GPU counts is 0, so it's not traing")
        return False

    return True

def buffer_checker(args, task, rehearsal):
    #print text file
    check_components(args, task, rehearsal, True)
        
        
def control_lr_backbone(args, optimizer, frozen):
    if frozen is True:
        lr = 0.0
    else:
        lr = args.lr_backbone
        
    optimizer.param_groups[-1]['lr'] = lr
            
    return optimizer


def dataset_configuration(args, original_set, aug_set):
    
    original_dataset, original_loader, original_sampler = original_set[0], original_set[1], original_set[2]
    if aug_set is None :
        AugRplay_dataset, AugRplay_loader, AugRplay_sampler = None, None, None
    else :
        AugRplay_dataset, AugRplay_loader, AugRplay_sampler = aug_set[0], aug_set[1], aug_set[2]
    
    if args.AugReplay and not args.MixReplay:
        return AugRplay_dataset, AugRplay_loader, AugRplay_sampler
    
    elif args.AugReplay and args.MixReplay:
        print(colored("MixReplay dataset generating", "blue", "on_yellow"))
        return [AugRplay_dataset, original_dataset], [AugRplay_loader, original_loader], [AugRplay_sampler, original_sampler] 
    
    else :
        print(colored("Original dataset generating", "blue", "on_yellow"))
        return original_dataset, original_loader, original_sampler

def compute_attn_weight(teacher_model, student_model, samples, device, ex_device):
    """Compute location loss between teacher and student models."""
    teacher_model.to(device)
    with torch.no_grad():
        teacher_encoder_outputs = []
        hook = teacher_model.transformer.encoder.layers[-1].self_attn.attention_weights.register_forward_hook(
            lambda module, input, output: teacher_encoder_outputs.append(output)
        )
        _ = teacher_model(samples)
        hook.remove()
        teacher_model.to(ex_device)

    return teacher_encoder_outputs[0].detach()

import gc
def refresh_data():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    
from GLIGEN.pre_defined import load_or_merge_meta_files
# from GLIGEN.pre_defined import define_blip
from GLIGEN.gligen_inference import custom_dataset, run
import GLIGEN.dist as gdist
import random, json, sys
def generation_process(args, max_class, min_class, insufficient_objects=None, count=0):
    meta = _init_meta(args.gligen_path)

    
    #* generate prompgts
    if args.coco_generator is True and insufficient_objects is None:
        # processor, model = define_blip("Salesforce/blip2-opt-2.7b")
        processor, model = None, None
        new_meta_list = custom_dataset(args, meta, args.max_length, max_class, min_class, processor, model)
        del model, processor
        torch.cuda.empty_cache()
        
    #* make intergration dataset and load dataset
    base_path = args.generator_path
    directory_path = os.path.join(base_path, "annotations")
    new_meta_list = load_or_merge_meta_files(directory_path) #* load Total_mata
    
    #* synchronization
    if utils.get_world_size() > 1: dist.barrier() #sync
        
    #* list refiner
    new_meta_list = [data for data in new_meta_list if data is not None]

    #* synchronization
    if utils.get_world_size() > 1: dist.barrier() #sync
    my_slice = new_meta_list
    #* Data distribution in multi-GPU environment
    if insufficient_objects is None :
        world_size = utils.get_world_size()
        rank = utils.get_rank()
        total_size = len(new_meta_list)
        per_process_size = total_size // world_size
        start_idx = int(rank * per_process_size)
        end_idx = int(start_idx + per_process_size if rank != world_size - 1 else total_size)
    
        #* Shuffle and get the current process's slice of data
        my_slice = new_meta_list[start_idx:end_idx]
    
    random.shuffle(my_slice)
    #* Execute the main implementation
    run(my_slice, args, insufficient_objects, None, count)
    
    #* Completion message
    print("Complete all generation")
    
    #* Synchronize processes if in distributed setting
    if utils.get_world_size() > 1: dist.barrier()

    return

def _init_meta(SD_pretrained):
    return dict(
            ckpt = SD_pretrained,
            prompt = "a teddy bear sitting next to a bird",
            phrases = ['a teddy bear', 'a bird'],
            locations = [ [0.0,0.09,0.33,0.76], [0.55,0.11,1.0,0.8] ],
            alpha_type = [1.0, 0.0, 0.0],
            save_folder_name="generation_box_text"
        )
    
import os
import shutil
import json
from tqdm import tqdm
#! for test
def is_overlapping(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Return False if one bbox is to the left of the other or above the other
    if x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1:
        return False
    return True
#! for test
def compute_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2

    iou = inter_area / float(bbox1_area + bbox2_area - inter_area)
    return iou

def check_and_copy_different_annotations(pseudo, origin, gen_path):
    gen_image_path = os.path.join(gen_path, "images")
    
    # Create specific folder
    destination_folder = os.path.join(gen_path, "duplicated_images")
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    pseudo_images = pseudo.getImgIds()
    
    # List to save image IDs with differences and their count
    origin_more = []  # When the number of annotations in origin is greater
    #! for test
    IOU_THRESHOLD = 0.5  # Adjust as needed
    for p_img_id in tqdm(pseudo_images, desc="label checking processing "):
        origin_anns = origin.loadAnns(origin.getAnnIds(p_img_id))
        
        overlapping = []
        for i, ann1 in enumerate(origin_anns):
            for j, ann2 in enumerate(origin_anns):
                if i != j and compute_iou(ann1['bbox'], ann2['bbox']) >= IOU_THRESHOLD:
                    overlapping.append(ann1['id'])
                    overlapping.append(ann2['id'])

        # If overlapping bounding boxes are found
        if len(overlapping) > 2:
            original_img_path = os.path.join(gen_image_path, pseudo.loadImgs(p_img_id)[0]["file_name"])
            origin_size = (int(origin.loadImgs(p_img_id)[0]["height"]), int(origin.loadImgs(p_img_id)[0]["width"]))
            pseudo_size = (int(pseudo.loadImgs(p_img_id)[0]["height"]), int(pseudo.loadImgs(p_img_id)[0]["width"]))
            
            # Draw bounding box on the image
            img_with_bbox = draw_bbox_on_image(original_img_path, [ann for ann in origin_anns if ann['id'] in overlapping], [], origin_size, pseudo_size)  # origin in red
            
            # Save the image with the bounding box
            bbox_img_path = os.path.join(destination_folder, pseudo.loadImgs(p_img_id)[0]["file_name"])
            cv2.imwrite(bbox_img_path, img_with_bbox)
    #     original_annotations_count = len(origin.loadAnns(origin.getAnnIds(p_img_id)))
    #     pseudo_annotations_count = len(pseudo.loadAnns(pseudo.getAnnIds(p_img_id)))
    
    #     origin_anns = origin.loadAnns(origin.getAnnIds(p_img_id))
    #     pseudo_anns = pseudo.loadAnns(pseudo.getAnnIds(p_img_id))
        
    #     origin_size = (int(origin.loadImgs(p_img_id)[0]["height"]), int(origin.loadImgs(p_img_id)[0]["width"]))
    #     pseudo_size = (int(pseudo.loadImgs(p_img_id)[0]["height"]), int(pseudo.loadImgs(p_img_id)[0]["width"]))
        
    #     if original_annotations_count > pseudo_annotations_count:
    #         original_img_path = os.path.join(gen_image_path, pseudo.loadImgs(p_img_id)[0]["file_name"])
            
    #         # Draw bounding box on the image
    #         img_with_bbox = draw_bbox_on_image(original_img_path, origin_anns, pseudo_anns, origin_size, pseudo_size)  # origin in red
            
    #         # Save the image with the bounding box
    #         bbox_img_path = os.path.join(destination_folder, pseudo.loadImgs(p_img_id)[0]["file_name"])
    #         cv2.imwrite(bbox_img_path, img_with_bbox)
            
    #         # Store only the image id, category id, and instance id
    #         origin_ids = [(ann["id"], ann["category_id"]) for ann in origin_anns]
    #         pseudo_ids = [(ann["id"], ann["category_id"]) for ann in pseudo_anns]
    #         origin_more.append((p_img_id, original_annotations_count, pseudo_annotations_count, origin_ids, pseudo_ids))
    
    # # Sort by image ID in descending order
    # origin_more.sort(key=lambda x: x[0])
    
    # with open(os.path.join(destination_folder, "origin_more.txt"), "a") as txt_file:
    #     for diff in origin_more:
    #         txt_file.write(f"Image ID: {diff[0]}, Original Count: {diff[1]}, Pseudo Count: {diff[2]}, origin labels :  {diff[3]}, pseudo labels: {diff[4]}\n")

import cv2
def draw_bbox_on_image(image_path, origin_anns, pseudo_anns, origin_size, pseudo_size):
    # 이미지를 불러옵니다.
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    
    # origin의 bounding box를 빨간색으로 그립니다. #* (original scene is Total_coco.json)
    for ann in origin_anns:
        bbox = ann['bbox']
        x, y, w_bbox, h_bbox = [int(coord) for coord in bbox]
        # 정규화
        # x_norm, y_norm, w_norm, h_norm = x/origin_size[1], y/origin_size[0], w_bbox/origin_size[1], h_bbox/origin_size[0]
        # x_norm, y_norm, w_norm, h_norm = x, y/origin_size[0], w_bbox, h_bbox/origin_size[0]
        # 512 크기에 맞게 조정
        # x, y, w_bbox, h_bbox = int(x_norm*512), int(y_norm*512), int(w_norm*512), int(h_norm*512)
        cv2.rectangle(img, (x, y), (x+w_bbox, y+h_bbox), (0, 0, 255), 2)
    
    # pseudo의 bounding box를 파란색으로 그립니다. #* pseudo_data.json
    for ann in pseudo_anns:
        bbox = ann['bbox']
        x, y, w_bbox, h_bbox = [int(coord) for coord in bbox]
        # 정규화
        # x_norm, y_norm, w_norm, h_norm = x/pseudo_size[1], y/pseudo_size[0], w_bbox/pseudo_size[1], h_bbox/pseudo_size[0]
        # # 512 크기에 맞게 조정
        # x, y, w_bbox, h_bbox = int(x_norm*512), int(y_norm*512), int(w_norm*512), int(h_norm*512)
        cv2.rectangle(img, (x, y), (x+w_bbox, y+h_bbox), (255, 0, 0), 2)
    
    return img

def modify_coco_data(coco_data):
    '''
        generation for batch size effect.
    '''
    if next((True for image in coco_data['images'] if image['file_name'].startswith("1")), False):
        print("already revised coco anns")
        return None
    # 원본 데이터를 복사합니다.
    modified_data = coco_data.copy()
    
    # 이미지 정보와 객체 정보를 복사하고 수정합니다.
    new_images = []
    new_annotations = []
    for image in coco_data['images']:
        new_image = image.copy()
        new_image['file_name'] = "1" + new_image['file_name'][1:]
        new_image['id'] = int(new_image['file_name'].split('.')[0])
        new_images.append(new_image)
        
        # 해당 image_id에 대응하는 객체 정보를 찾아 수정합니다.
        for annotation in coco_data['annotations']:
            if annotation['image_id'] == image['id']:
                new_annotation = annotation.copy()
                new_annotation['image_id'] = new_image['id']
                new_annotations.append(new_annotation)
    
    # 수정된 이미지 정보와 객체 정보를 원본 데이터에 추가합니다.
    modified_data['images'].extend(new_images)
    modified_data['annotations'].extend(new_annotations)
    
    return modified_data

def gen_ratio_check(original_data_path, gen_data_path, target_ratio=500, min_c=0, max_c=90):
    eps = 0.000000000000001
    # 첫 번째 JSON 파일 로드
    with open(original_data_path, 'r') as f:
        original_dataset = json.load(f)

    # 두 번째 JSON 파일 로드 (파일 경로는 적절히 수정하세요)
    with open(gen_data_path, 'r') as f:
        gen_dataset = json.load(f)
        
    # 어노테이션 개수 계산 함수
    def _get_annotation_counts(data):
        category_counts = {}
        for annotation in data['annotations']:
            category_id = annotation['category_id']
            if min_c <= category_id <= max_c:  # min_c와 max_c 사이에 있는 category_id만 계산
                if category_id not in category_counts:
                    category_counts[category_id] = 0
                category_counts[category_id] += 1
        return category_counts
        
    annotation_origin = _get_annotation_counts(original_dataset)
    annotation_gen = _get_annotation_counts(gen_dataset)
    
    # original set configuraiton
    category_id_to_name = {}
    for category in original_dataset['categories']:
        if min_c <= category['id'] <= max_c:  # min_c와 max_c 사이에 있는 category_id만 고려
            category_id_to_name[category['id']] = category['name']

    # sorted_category_id_to_name = dict(sorted(annotation_gen.items(), key=lambda item: item[0]))
    
    # 비율 계산 및 필요한 객체 개수 출력
    print("\nRequired counts to meet target ratio:")
    insufficient_objects = {}
    for category_id, name in category_id_to_name.items():
        gen_count = annotation_gen.get(category_id, 0)  # 해당 category_id가 gen_dataset에 없으면 0으로 처리
        if gen_count < target_ratio:
            required_count = target_ratio - gen_count
            insufficient_objects[name] = required_count

    return insufficient_objects

    # for category_id in sorted_category_id_to_name:
    #     if category_id in annotation_origin:
    #         current_ratio = annotation_gen[category_id] / (annotation_origin[category_id] + eps)
    #         if current_ratio < target_ratio:
    #             required_count = (target_ratio * annotation_origin[category_id]) - annotation_gen[category_id]
    #             insufficient_objects[category_id] = required_count
    #             label_name = category_id_to_name[category_id]
    #             insufficient_objects[label_name] = required_count

    return insufficient_objects

from tqdm import tqdm
def filter_annotations_by_threshold(data_path, threshold):
    
    max_category = 79
    if utils.is_main_process():
        # Load JSON data
        json_dir = os.path.join(data_path, "annotations/pseudo_data.json")
        img_dir = os.path.join(data_path, "images")
        with open(json_dir, 'r') as f:
            data = json.load(f)

        # Initialize counts and mappings
        category_counts = {}
        image_categories = {}

        # Calculate counts and record categories per image
        for annotation in data['annotations']:
            category_id = annotation['category_id']
            image_id = annotation['image_id']
            category_counts[category_id] = category_counts.get(category_id, 0) + 1
            image_categories.setdefault(image_id, list()).append(category_id)

        image_ids = list(image_categories.keys())
        random.shuffle(image_ids)  # 이미지 ID를 랜덤하게 섞습니다.

        images_to_remove = set()
        for image_id in image_ids:
            categories = image_categories[image_id]
            if all(category_counts[cat_id] > threshold for cat_id in categories) or any(cat_id > max_category for cat_id in categories):
                images_to_remove.add(image_id)
                for cat_id in categories:
                    category_counts[cat_id] -= 1

        # Remove image files
        for image_id in tqdm(images_to_remove, desc="Deleting images"):
            image_info = next((img for img in data['images'] if img['id'] == image_id), None)
            if image_info:
                image_file_path = os.path.join(img_dir, image_info['file_name'])
                os.remove(image_file_path)
                
        # Filter annotations
        annotations_to_keep = [anno for anno in data['annotations'] if anno['image_id'] not in images_to_remove]
        data['annotations'] = annotations_to_keep

                    
        # Filter images
        image_ids_to_keep = {anno['image_id'] for anno in annotations_to_keep}
        data['images'] = [img for img in data['images'] if img['id'] in image_ids_to_keep]

        # Write updated JSON data
        with open(json_dir, 'w') as f:
            json.dump(data, f, indent=4)


        print(f"Processed and saved the filtered data to 'pseudo_data.json'.")
    #* if use MultiGPU, so then you should sync each GPUs
    if utils.get_world_size() > 1 : dist.barrier()
    return


def get_existing_image_ids(json_dir, insufficient_objects=None):
    with open(json_dir, 'r') as f:
        data = json.load(f)
    if insufficient_objects is not None:
        name_to_id = [cat['id'] for cat in data['categories'] if cat["name"] in insufficient_objects.keys()]
    else:
        name_to_id = None
        
    return [img['file_name'] for img in data['images']], name_to_id


def check_anns(data_path):
    max_category_id = 79
    if utils.is_main_process():
        # Load JSON data
        json_dir = os.path.join(data_path, "annotations/pseudo_data.json")
        img_dir = os.path.join(data_path, "images")
        with open(json_dir, 'r') as f:
            data = json.load(f)

        # 카테고리별 어노테이션 개수를 저장할 딕셔너리 초기화
        category_counts = {}
        
        # Iterate over annotations and count by category, ignoring categories with ID > max_category_id
        for annotation in data['annotations']:
            category_id = annotation['category_id']

            if category_id not in category_counts:
                category_counts[category_id] = 0
            category_counts[category_id] += 1

        # 카테고리 ID와 이름을 매핑하기 위한 딕셔너리 생성
        category_id_to_name = {}
        for category in data['categories']:
            category_id_to_name[category['id']] = category['name']

        sorted_category_id_to_name = dict(sorted(category_counts.items(), key=lambda item: item[0]))

        # 결과 출력
        for category_id, count in sorted_category_id_to_name.items():
            print(f"ID: {category_id} Category: {category_id_to_name[category_id]}, Count: {count}")
    
        # # Get the list of image files that are referenced in the JSON data
        # referenced_images = {img['file_name'] for img in data['images']}

        # # Get all image files from the image directory
        # all_images = set(os.listdir(img_dir))

        # # Identify images not referenced in the JSON data
        # unreferenced_images = all_images - referenced_images

        # # Remove the unreferenced image files
        # for img_file in unreferenced_images:
        #     img_path = os.path.join(img_dir, img_file)
        #     os.remove(img_path)
        #     print(f"Removed unreferenced image: {img_path}")

        # print("Finished removing unreferenced images.")
    #* if use MultiGPU, so then you should sync each GPUs
    if utils.get_world_size() > 1 : dist.barrier()
    return