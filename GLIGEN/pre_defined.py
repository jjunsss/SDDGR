from copy import deepcopy
from pycocotools.coco import COCO
import os
import cv2
from PIL import Image
# from transformers import AutoProcessor, Blip2ForConditionalGeneration
from GLIGEN.coco_annotations import coco_loader, filter_annotations_and_images, make_meta_dict
from tqdm import tqdm
import torch.distributed as dist
import torch
import GLIGEN.dist as gdist
import glob
from termcolor import colored
from pathlib import Path
from functools import lru_cache

# @lru_cache(maxsize=None)
# def define_blip(load_pre, ):
#     # blip-2 for generating prompt
#     # processor = AutoProcessor.from_pretrained(load_pre)
#     # model = Blip2ForConditionalGeneration.from_pretrained(load_pre, torch_dtype=torch.float16)
#     # device = "cuda" if torch.cuda.is_available() else "cpu"
#     # model.to(device)
    
#     return processor, model

def load_or_merge_meta_files(directory_path: str) -> list:
    directory = Path(directory_path)
    total_files = list(directory.glob('Total_*.json'))
    
    if total_files:
        print(f"Total_ files exist: {total_files}. Using the first one.")
        return gdist.load_meta_files(directory / 'Total_meta.json')
    
    print("No Total_ files found. Merging files from 4 GPUs.")
    if gdist.get_world_size() > 1:
        dist.barrier() #sync
        
    if gdist.is_main_process():
        meta_file_paths = [directory / f"{i}_meta.json" for i in range(4)]
        gdist.merge_meta_files(meta_file_paths, directory / 'Total_meta.json')
        
        coco_file_paths = [directory / f"{i}_train.json" for i in range(4)]
        gdist.merge_coco_like_jsons(coco_file_paths, directory / 'Total_coco.json')
    
    if gdist.get_world_size() > 1:
        dist.barrier() #sync
    
    return gdist.load_meta_files(directory / 'Total_meta.json')