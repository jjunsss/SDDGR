# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import numpy as np
from typing import Iterable
from tqdm import tqdm
import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.data_prefetcher import data_prefetcher
import os
import time
from typing import Tuple, Collection, Dict, List
import torch
import util.misc as utils
import numpy as np
from typing import Tuple, Dict, List, Optional
from tqdm import tqdm
from custom_training import *
from custom_utils import *
import wandb
from models import inference_model
from custom_fake_target import *
import shutil


def extra_epoch_for_replay(args, dataset_name: str, data_loader: Iterable, model: torch.nn.Module, criterion: torch.nn.Module,
                            device: torch.device, rehearsal_classes=None, current_classes=None):

    '''
        Run additional epoch to collect replay buffer. 
        1. initialize prefeter, (icarl) feature extractor and prototype.d
        2. run rehearsal training.dd
        3. (icarl) detach values in rehearsal_classes.
    '''
    # current_classes = [2, 3, 4]
    prefetcher = create_prefetcher(dataset_name, data_loader, device, args)

    with torch.no_grad():
        for idx in tqdm(range(len(data_loader)), disable=not utils.is_main_process()): #targets
            samples, targets, = prefetcher.next()
                
            # extra training을 통해서 replay 데이터를 수집하도록 설정
            rehearsal_classes = rehearsal_training(args, samples, targets, model, criterion, 
                                                   rehearsal_classes, current_classes)
    
            if idx % 100 == 0:
                torch.cuda.empty_cache()
            
            # 정완 디버그
            if args.debug:
                if idx == args.num_debug_dataset:
                    break

    return rehearsal_classes


def create_prefetcher(dataset_name: str, data_loader: Iterable, device: torch.device, args: any) \
        -> data_prefetcher:
    if dataset_name == "Original":    
        return data_prefetcher(data_loader, device, prefetch=True, Mosaic=False)
    elif dataset_name == "AugReplay":
        return data_prefetcher(data_loader, device, prefetch=True, Mosaic=True)
    elif dataset_name == "Pseudo":
        return data_prefetcher(data_loader, device, prefetch=True, pseudo_labeling=True)
    else:
        return data_prefetcher(data_loader, device, prefetch=True, Mosaic=False)

import random
def train_one_epoch(args, task_idx, last_task, epo, model: torch.nn.Module, teacher_model, criterion: torch.nn.Module, dataset_train,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler,
                    device: torch.device, dataset_name: str, current_classes: List = [], rehearsal_classes: Dict = {},
                    first_training = False):
    """
        in here, if I control each training section(original, circular order)
    """
    prefetcher = create_prefetcher(dataset_name, data_loader, device, args)
    
    set_tm = time.time()
    sum_loss = 0.0
    count = 0
    for idx in tqdm(range(len(data_loader)), disable=not utils.is_main_process()): #targets
        if idx % 100 == 0:
            refresh_data()
        samples, targets = prefetcher.next()
        
        if dataset_name == "AugReplay" and not first_training:
            replay_samples, replay_targets = prefetcher.next()
            
        if dataset_name == "Original" or first_training:
            sum_loss, count = training(args, task_idx, last_task, epo, idx, count, sum_loss, samples, targets,  
                                        model, teacher_model, criterion, optimizer, current_classes, "original")

        if dataset_name == "AugReplay" and args.Rehearsal and not first_training:
            CER_Prob = random.random() # if I set this to 0 or 1, so then usually fixed CER mode.
            if CER_Prob < 0.5: # this term is for randomness training in "replay and original"
                sum_loss, count = training(args, task_idx, last_task, epo, idx, count, sum_loss, samples, targets,  
                                            model, teacher_model, criterion, optimizer, current_classes, "original")
                sum_loss, count = training(args, task_idx, last_task, epo, idx, count, sum_loss, replay_samples, replay_targets,  
                                            model, teacher_model, criterion, optimizer, current_classes, "circular")
            else :
                sum_loss, count = training(args, task_idx, last_task, epo, idx, count, sum_loss, replay_samples, replay_targets,  
                                            model, teacher_model, criterion, optimizer, current_classes, "circular")
                sum_loss, count = training(args, task_idx, last_task, epo, idx, count, sum_loss, samples, targets,  
                                            model, teacher_model, criterion, optimizer, current_classes, "original")

        # 정완 디버그
        if args.debug:
            if count == args.num_debug_dataset:
                break
        
        if utils.is_main_process() and args.wandb:
            wandb.log({"loss": sum_loss / count})
        
    if utils.is_main_process():
        print("Total Time : ", time.time() - set_tm)
        

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, DIR, args) :
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    
    #FIXME: check your cocoEvaluator function for writing the results (I'll give you code that changed)
    coco_evaluator = CocoEvaluator(base_ds, iou_types, DIR)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    
    cnt = 0 # for debug
        
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = inference_model(args, model, samples, targets, eval=True)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict, True)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict and k in ['loss_ce', 'loss_giou', 'loss_bbox']}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             )
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, args.model_name)
        #print(results)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        
        gt = outputs[0]['gt'] if args.model_name == 'dn_detr' else outputs['gt']
        
        # cocoeval에서 gt와 dt를 맞추어주기 위함
        if gt is not None :
            for r in res.values():
                labels = r['labels'].cpu().numpy()
                r['labels'] = torch.tensor([
                    gt[tgt_id-1] for tgt_id in labels
                ], dtype=torch.int64).cuda()
        
        if coco_evaluator is not None:
            coco_evaluator.update(res)
            
        if args.debug:
            cnt += 1
            if cnt == args.num_debug_dataset:
                break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    return stats, coco_evaluator

def pseudo_process(args, dataset_frame, data_loader: Iterable, image_paths, model: torch.nn.Module, device: torch.device, insufficient_cats=None, count=0, min_class=0, max_class=0):
    model.eval()  # 모델을 평가 모드로 설정합니다.
    model.to(device)
    prefetcher = create_prefetcher("Pseudo", data_loader, device, args)
    
    pbar = tqdm(total=len(data_loader), desc="pseudo labeling generation", disable=not utils.is_main_process())
    heigth, width = 512, 512
    image_info = []
    annotations = []
    removal_list = [] 
    
    with torch.no_grad():  # 그라디언트 계산을 비활성화합니다.
        for idx in range(len(data_loader)):  # data_loader를 통해 배치를 반복적으로 불러옵니다.
            images, image_name = prefetcher.next()
            # 모델을 사용하여 이미지에 대한 예측을 수행합니다.
            outputs = model(images)
            image_name = image_name.pop()
            labels, areas, boxes, threshold = pseudo_target(outputs, count, min_class, max_class)
            pbar.set_postfix(threshold=f"{threshold:.2f}", count=count)
            
            if boxes is None:
                removal_list.append(os.path.join(image_paths, image_name))
                pbar.update(1)
                continue
            
            if insufficient_cats is not None :
                valid_indices = [i for i, label in enumerate(labels) if label.item() in insufficient_cats]
                if not valid_indices:
                    removal_list.append(os.path.join(image_paths, image_name))
                    pbar.update(1)
                    continue
            
                # labels = labels[valid_indices]
                # areas = areas[valid_indices]
                # boxes = boxes[valid_indices]
                
            #* make images in coco format json
            image_info.append(_generate_imageinfo(image_name, heigth, width))
            
            #* boxes resize to pseudo targets
            boxes[:, 0] = boxes[:, 0] * width
            boxes[:, 1] = boxes[:, 1] * heigth
            boxes[:, 2] = boxes[:, 2] * width
            boxes[:, 3] = boxes[:, 3] * heigth
            
            #* make annotations in coco format json
            for label, area, box in zip(labels, areas, boxes):
                annotations.append(_generate_anninfo(image_name, label, area, box))
            pbar.update(1)
    
    # 제거될 이미지들을 모아둘 폴더의 이름을 설정합니다.
    removal_folder_name = f"removed_images_{args.divide_ratio}"
    removal_folder_path = os.path.join(args.output_dir, removal_folder_name)
    
    # 폴더가 존재하지 않는 경우, 폴더를 생성합니다.
    if not os.path.exists(removal_folder_path):
        os.makedirs(removal_folder_path)
    
    #* delete generated images
    print(colored(f"remove file length :  {len(removal_list)}", "blue", "on_yellow"))
    for path in removal_list: #! move deleted images version
        # dest_path = os.path.join(removal_folder_path, os.path.basename(path))
        os.remove(path)
        
    # for path in removal_list: #! delete designated images version
    #     os.remove(path)
    
    dataset_frame["images"] = image_info
    dataset_frame["annotations"] = annotations
    return dataset_frame

def _generate_imageinfo(image_info, height, width):
    image_id = image_info.split(".").pop(0)
    return {
        'file_name': image_info,
        'height': height,
        'width': width,
        'id': int(image_id)
    }

def gen_label_id():
    count = 1
    while True:
        yield count
        count += 1
        
genorator_id = gen_label_id()

def _generate_anninfo(image_info, label, area, box):
    image_id = image_info.split(".").pop(0)
    return {
        'segmentation': [],
        'area': area.item(),  # Assuming area is a single value tensor
        'iscrowd': 0,
        'image_id': int(image_id),
        'bbox': box.tolist(),  # Assuming box is a tensor
        'category_id': label.item(),  # Assuming label is a single value tensor
        'id': next(genorator_id)
    }
