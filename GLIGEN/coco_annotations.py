from pycocotools.coco import COCO
import random
from tqdm import tqdm
import time
import json
import torch
import os

from torch import BoolTensor, FloatTensor, LongTensor
from typing import Dict, List, Optional, Tuple, Union

import GLIGEN.dist as gdist

def coco_loader(coco_dir):
    
    with open(coco_dir, 'r') as f:
        data = json.load(f)
        
    generate_data = {
            'info': data['info'],
            'licenses': data['licenses'],
            'images': data["images"],
            'annotations': data["annotations"],
            'categories': data['categories']
        }

    return generate_data

def gen_label_id():
    count = 1
    while True:
        yield count
        count += 1


def filter_annotations_and_images(generate_data, max_class, min_class):
    '''
        Filter annotations, not includeing target annotations
    '''
    new_annotations = []
    
    pbar = tqdm(total=len(generate_data['annotations']), desc="Processing annoatations", disable=not gdist.is_main_process())
    for annotation in generate_data['annotations']:
        if annotation['category_id'] in range(min_class, max_class+1):
            new_annotations.append(annotation)
            
        pbar.update(1)
    
    generate_data['annotations'] = new_annotations
    
    # Find image ids that have annotations
    valid_image_ids = set([anno['image_id'] for anno in new_annotations])
    
    # Filter images
    new_images = []
    for image in generate_data['images']:
        if image['id'] in valid_image_ids:
            new_images.append(image)
    
    generate_data['images'] = new_images
    
    return generate_data

from copy import deepcopy
def flip_annotations(generate_data, coco, flip_version="horizontal"):
    new_annotations = []
    flipped_annotations_lr = []
    flipped_annotations_ud = []
    
    for annotation in tqdm(generate_data['annotations'], desc="Flipping annotations", disable=not gdist.is_main_process()):
        new_annotations.append(annotation)
        image_info = coco.loadImgs(annotation["image_id"])
        image_width = image_info['width']
        image_height = image_info['height']
        
        # Flip left-right
        if flip_version == "horizontal":
            flipped_annotation_lr = deepcopy(annotation)
            flipped_annotation_lr['bbox'][0] = image_width - (annotation['bbox'][0] + annotation['bbox'][2])
            flipped_annotations_lr.append(flipped_annotation_lr)
        
        # Flip up-down
        if flip_version == "vertical":
            flipped_annotation_ud = deepcopy(annotation)
            flipped_annotation_ud['bbox'][1] = image_height - (annotation['bbox'][1] + annotation['bbox'][3])
            flipped_annotations_ud.append(flipped_annotation_ud)
    
    if flip_version == "horizontal":
        generate_data['annotations'].extend(flipped_annotations_lr)
    else:
        generate_data['annotations'].extend(flipped_annotations_ud)
    
    return generate_data

def _log_dataset(info_dict):
    # 딕셔너리를 JSON 형식으로 텍스트 파일에 저장
    txt_file_path = "./prompts_and_info.json"  # 저장할 텍스트 파일의 경로
    with open(txt_file_path, 'a') as f:
        json_str = json.dumps(info_dict)
        f.write(json_str + "\n")

def number_to_words(n):
	"""Convert a number into words."""
	if n == 1: return 'one'
	if n == 2: return 'two'
	if n == 3: return 'three'
	if n == 4: return 'four'
	if n == 5: return 'five'
	# ... you can extend this as needed
	return str(n)

import numpy as np
def construct_prompt_from_phrases(text_entities):
	"""Construct a new prompt based on the phrases."""
	# Count the occurrences of each noun
	word_counts = {word: text_entities.count(word) for word in set(text_entities)}
	
	adjusted_nouns = []
	for noun, count in word_counts.items():
		if count > 1:
			adjusted_nouns.append(f"{number_to_words(count)} {noun}")
		else:
			adjusted_nouns.append(noun)
	
	# Construct the new prompt
	new_prompt = f" A DSLR image of {' and '.join(adjusted_nouns)} demonstrating ultra high detail, 4K, 8K, ultra realistic, crisp edges, smooth, hyper detailed textures."
	
	return new_prompt
    
from PIL import Image
def make_meta_dict(args, new_train_json, sample, info, max_length, blip_processor=None, blip_model=None):
    '''
        for gligen generation process. It have to include bbox(for gligen), class labels(for gligen), prompt(for stable diffusion)
    '''
    image_id = info["id"]
    image_width, image_height = info["width"], info["height"] #* original image(coco) height, width. We change to this to 512(SD size)

    annotations =  [anno for anno in new_train_json["annotations"] if anno['image_id'] == image_id]
    labels = [ann['category_id'] for ann in annotations]
    
    if len(labels)  > max_length : 
        return None
    
    category_names = {category["id"]: category["name"] for category in new_train_json["categories"]}
    text_entities = [category_names[label] for label in labels if label in category_names]
    bounding_boxes = [ann['bbox'] for ann in annotations]
    
    normalized_bounding_boxes = []
    for bbox in bounding_boxes:
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h
        x1 /= image_width
        y1 /= image_height
        x2 /= image_width
        y2 /= image_height
        normalized_bounding_boxes.append([round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)])
    
    prompt = "" #Null prompt
    
    #! prompt fixed generation
    if prompt == "" and args.blip2 is not True :
        prompt = construct_prompt_from_phrases(text_entities)
    
    #! prompt blip-2 generation work.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if (prompt == "") and args.blip2:
        coco_path = args.coco_path + "train2017"
        image_path = os.path.join(coco_path, info['file_name'])
        image = Image.open(image_path)

        inputs = blip_processor(image, return_tensors="pt").to(device, torch.float16)
        generated_ids = blip_model.generate(**inputs, max_new_tokens=20)
        generated_text = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        prompt = generated_text
        
        
    save_base_path = args.generator_path
    save_folder_name = os.path.join(save_base_path, "images")
    
    temp_meta_dict = deepcopy(sample)
    temp_meta_dict["ckpt"] = "GLIGEN/gligen_checkpoints/checkpoint_generation_text_image.pth"
    temp_meta_dict["phrases"] = text_entities
    temp_meta_dict["prompt"] = prompt
    temp_meta_dict["save_folder_name"] = save_folder_name
    temp_meta_dict["image_id"] = f"{image_id:012d}"
    temp_meta_dict["locations"] = normalized_bounding_boxes
    
    return temp_meta_dict

def resize_annotations(annotations, original_width, original_height, new_width, new_height):
    resized_annotations = []
    for ann in annotations:
        # Bounding box 정보를 가져옵니다.
        x, y, width, height = ann['bbox']
        
        # Bounding box 좌표와 크기를 정규화합니다.
        x /= original_width
        y /= original_height
        width /= original_width
        height /= original_height
        
        # 정규화된 좌표와 크기를 사용하여 새로운 이미지 크기에 맞게 bounding box를 조정합니다.
        x *= new_width
        y *= new_height
        width *= new_width
        height *= new_height
        
        # 조정된 bounding box를 annotation에 저장합니다.
        ann['bbox'] = [round(x, 3), round(y, 3), round(width, 3), round(height, 3)]
        resized_annotations.append(ann)
    return resized_annotations


def refine_meta(max_count=0, object_counts=dict(), meta=dict(), insufficient=False, count=0):
    '''
        max_count : args.max_count. limit count
        object_counts : counts the generated objects
        meta : meta info for generating GLIGEN process
    '''
    new_phrases = []
    new_locations = []
    new_prompt = None

    for phrase, location in zip(meta["phrases"], meta["locations"]):
        # object_counts에 해당 phrase가 없으면 0으로 초기화
        if insufficient and phrase not in object_counts:
            continue

        if insufficient and object_counts[phrase] > -3:
            new_phrases.append(phrase)
            new_locations.append(location)
            object_counts[phrase] -= 1
            continue
            
        if phrase not in object_counts and not insufficient:
            object_counts[phrase] = 0
            
        if object_counts[phrase] < max_count and not insufficient:
            new_phrases.append(phrase)
            new_locations.append(location)
            object_counts[phrase] += 1

    if not new_phrases:
        return None, object_counts
    
    meta["phrases"] = new_phrases
    meta["locations"] = new_locations
    meta["prompt"] = construct_prompt_from_phrases(new_phrases)

    return meta, object_counts