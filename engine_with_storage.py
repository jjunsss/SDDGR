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
import os
import sys
import random
from typing import Iterable
from tqdm import tqdm
import torch.distributed as dist
import pickle
import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
from torch.utils.data import DataLoader
import os
import time

from datasets import build_dataset, get_coco_api_from_dataset

def train_one_epoch(epo, model: torch.nn.Module,criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, file_name: str, epoch: int, max_norm: float = 0, ):
    
    label_dict = {}
    model.train()
    criterion.train()
    ex_device = torch.device("cpu")
       
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets, origin_samples, origin_targets = prefetcher.next() 
    #* samples: transformed images /targets : transformed targets / origin_samples - targets : original before transform
    
    sum_loss = 0.0
    set_tm = time.time()
    count = 0
    #low batch essential
    limit2 = [28, 32, 35, 41, 56] #photozone
    limit3 = [22, 23, 24, 25, 26, 27, 29, 31, 33, 37, 39, 40, 45, 46, 48, 49, 51, 52, 58, 59] #VE 

    loss_counts = {}
    loss_count = 0
    current_class = {}
    for idx in range(len(data_loader)): #targets 
        if utils.is_main_process():
            print(f"now working gpu :{dist.get_rank()} ")
            print("idx : ", idx, "epoch : ", epo)

        train_check = True
        samples = samples.to(ex_device)
        targets = [{k: v.to(ex_device) for k, v in t.items()} for t in targets]
        
        #TODO : one samples no over / one samples over solve this ! 
        no_use = []
        yes_use = []
        check_list = []
        if idx < 1000000:
            with torch.no_grad():
                for enum, target in enumerate(targets):

                    label_tensor = target['labels']
                    label_tensor = label_tensor.to(ex_device)
                    label_tensor_unique = torch.unique(label_tensor)
                    
                    # rate 1(did) : 4(ve + pz)
                    check_list = [idx.item() for idx in label_tensor_unique if idx.item() in label_dict  if idx.item() <= 21 and label_dict[idx.item()] > 4000] #did
                    check_list2 = [idx.item() for idx in label_tensor_unique if idx.item() in label_dict  if idx.item() in limit2 and label_dict[idx.item()] > 4000] #pz
                    check_list3 = [idx.item() for idx in label_tensor_unique if idx.item() in label_dict  if idx.item() in limit3 and label_dict[idx.item()] > 7000] #ve (more)
                    #check_list4 = [idx.item() for idx in label_tensor_unique if idx.item() in label_dict  if (label_dict[idx.item()] > 20000)] #ve(smallest)

                    #TODO : before checking overlist Process outputdim and continue iter
                    if len(check_list) > 0 or len(check_list2) > 0 or len(check_list3) > 0 :
                        if utils.is_main_process():
                            print("overlist: ", check_list, check_list2, check_list3)
                        no_use.append(enum)

                    else:
                        yes_use.append(enum)
                        label_tensor_count = label_tensor.numpy()
                        bin = np.bincount(label_tensor_count)
                        for idx in label_tensor_unique:
                            idx = idx.item()
                            if idx in label_dict.keys():
                                label_dict[idx] += bin[idx]
                            else :
                                label_dict[idx] = bin[idx]
                                

                if len(no_use) == 3:
                    train_check = False
                        
                if len(no_use) == 2: #1 training
                    new_targets = []
                    useit0 = yes_use[0]
                    ten, mask = samples.decompose()
                    
                    ten0 = torch.unsqueeze(ten[useit0], 0)
                    mask0 = torch.unsqueeze(mask[useit0], 0)
                    samples = utils.NestedTensor(ten0, mask0)
                    new_targets.append(targets[useit0])
                    targets = [{k: v for k, v in t.items()} for t in new_targets]
                    
                if len(no_use) == 1: # Two batch training if you use 3 batches, so then you should this function. because 2 no use situation. 
                    
                    new_targets = []

                    useit0 = yes_use[0]
                    useit1 = yes_use[1]
                    ten, mask = samples.decompose()
                    
                    ten0 = torch.unsqueeze(ten[useit0], 0)
                    mask0 = torch.unsqueeze(mask[useit0], 0)
                    ten1 = torch.unsqueeze(ten[useit1], 0)
                    mask1 = torch.unsqueeze(mask[useit1], 0)
                    
                    ten = torch.cat([ten0,ten1], dim = 0) 
                    mask = torch.cat([mask0,mask1], dim = 0) 
                    samples = utils.NestedTensor(ten, mask)
                    new_targets.append(targets[useit0])
                    new_targets.append(targets[useit1])
                    targets = [{k: v for k, v in t.items()} for t in new_targets]
            
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            losses_value = losses.item()
            
            if train_check == True:
                if losses_value > 0.5 and losses_value < 1.5 : #make a retraining exemplar 
                    ex_device = torch.device("cpu")
                    samples = samples.to(ex_device)
                    targets = [{k: v.to(ex_device) for k, v in t.items()} for t in targets]
                    for enum, target in enumerate(targets):
                        new_targets = []
                        label_tensor = target['labels']
                        label_tensor_unique = torch.unique(label_tensor)
                        label_tensor_count = label_tensor.numpy()
                        bin = np.bincount(label_tensor_count)
                        
                        # if len(targets) > 1:
                        #     ten, mask = samples.decompose()
                        #     ten = torch.unsqueeze(ten[enum], 0)
                        #     mask = torch.unsqueeze(mask[enum], 0)
                        #     samples = utils.NestedTensor(ten, mask)
                        #     new_targets.append(targets[enum])
                            
                        idx_count = 0 # for count images
                        image_id = target["image_id"].item()
                        #Rehearsal. 오름차순 정렬하고, Loss가 큰 값들부터 제거 -> Loss가 크면 대표성이 떨어짐.
                        for unique_idx in label_tensor_unique:
                            unique_idx = unique_idx.item()
                            if idx_count == 0:
                                if unique_idx in current_class.keys():
                                    idx_count += 1
                                    current_class[unique_idx] = sorted(current_class[unique_idx], key = lambda x : x[1]) # Loss 기준으로 asc
                                    #if sum(np.array(current_class[unique_idx], dtype = object)[:, 0]) < 75: #하나의 class에 해당하는 객체 100개 이상 금지.
                                    #! Trouble Shooting solution
                                    if sum([component[0] for component in current_class[unique_idx]]) < 75: #하나의 class에 해당하는 객체 100개 이상 금지.
                                        current_class[unique_idx].append([bin[unique_idx], losses_value, True, image_id])
                                    else :
                                        #print("/////////// count over ///////////")
                                        if current_class[unique_idx][-1][1] > losses_value:
                                            current_class[unique_idx][-1] = [bin[unique_idx], losses_value, True, image_id]
                                else :
                                    idx_count += 1
                                    current_class[unique_idx] = [[bin[unique_idx], losses_value, True, image_id]]

                            else: #idx count >= 1 
                                if unique_idx in current_class.keys():
                                    current_class[unique_idx] = sorted(current_class[unique_idx], key = lambda x : x[1]) # Loss 기준으로 정렬하기
                                    
                                    #if sum(np.array(current_class[unique_idx], dtype = object)[:, 0]) < 75: #하나의 class에 해당하는 객체 100개 이상 금지.
                                    #! Trouble Shooting solution
                                    if sum([component[0] for component in current_class[unique_idx]]) < 75: #하나의 class에 해당하는 객체 100개 이상 금지.
                                        current_class[unique_idx].append([bin[unique_idx], losses_value, False])
                                    else :
                                        #print("/////////// count over ///////////")
                                        current_class[unique_idx][-1] = [bin[unique_idx], losses_value, False]
                                        if current_class[unique_idx][-1][1] > losses_value:
                                            current_class[unique_idx][-1] = [bin[unique_idx], losses_value, False]
                                else :
                                    current_class[unique_idx] = [[bin[unique_idx], losses_value, False]]

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict, train_check)
            if loss_dict_reduced == False:
                samples, targets, origin_samples, origin_targets = prefetcher.next()
                print(f'Total GPU not working... so passed \n')
                continue
            count += 1
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()
            sum_loss += loss_value
            if utils.is_main_process(): #sum_loss가 GPU의 개수에 맞춰서 더해주고 있으니,
                print(f"losses : {loss_value:05f}, epoch_total_loss : {(sum_loss / count):05f}, count : {count}")
                print(f"total examplar counts : {sum([len(current_class[idx]) for idx in current_class])}")
                 
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                samples, targets, origin_samples, origin_targets = prefetcher.next()
                continue
            
            optimizer.zero_grad()
            losses.backward()
            
            if max_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            else:
                grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
            optimizer.step()
            samples, targets, origin_samples, origin_targets = prefetcher.next()
        else:
            break
    if utils.is_main_process():
        print("all loss: ", loss_dict_reduced)
        print("Total Time : ", time.time() - set_tm)
        
    #TODO : delete bincount, loss_value in Current class 
    for _, contents in current_class.items():
        del_list = [idx for idx, content in enumerate(contents) if content[2] == False]
        for idx in del_list[::-1]:
            del contents[idx]
    
    #TODO : bin count delete dictionary
    for dict_key, dict_values in current_class.items():
        if len(dict_values) < 1 :
            del dict_key

    #* save the capsulated dataset(Boolean, image_id:int)
    file_name = file_name + "_NEW"
    with open(file_name, 'wb') as f:
        pickle.dump(current_class, f)
        del current_class

def _new_dataLoader(saved_dict, args):
    #print(f"{dist.get_rank()}gpu training saved dict : {saved_dict}")
    dataset_idx_list = []
    for _, value in saved_dict.items():
        if len(value) > 0 :
            np_idx_list = np.array(value, dtype=object)
            dataset_idx_list.extend(np.unique(np_idx_list[:, 3]).astype(np.uint8).tolist())
    #print(f"{dist.get_rank()} gpu dataset_idx_list : {dataset_idx_list}")
    
    custom_dataset = build_dataset(image_set='train', args=args, img_ids=dataset_idx_list)
    
    custom_loader = DataLoader(custom_dataset, args.batch_size, shuffle=True, collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    return custom_loader

def train_rehearsal(epoch, model: torch.nn.Module, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    device: torch.device, file_name: str, args, max_norm: float = 0, ):
    print(f"*****{dist.get_rank()} GPU rehearsal training start ******")
    def _back_training(samples, targets, model: torch.nn.Module, criterion: torch.nn.Module):
        while True:
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            
            loss_dict_reduced = utils.reduce_dict(loss_dict, True)
            if loss_dict_reduced is not False:
                return loss_dict_reduced
    
    rehearsal_class = {}
    count = 0
    sum_loss = 0
    OLD_file_name = file_name + "_OLD"
    #okay one reading
    if os.path.isfile(OLD_file_name): 
        with open(OLD_file_name, 'rb') as f :
            rehearsal_class = pickle.load(f)
    else:
        rehearsal_class = {}
    if len(rehearsal_class.keys()) > 0:
        custom_trainloader = _new_dataLoader(rehearsal_class, args)
        prefetcher = data_prefetcher(custom_trainloader, device, prefetch=True)
        samples, targets = prefetcher.next()
        print(f"{dist.get_rank()} gpu prefetcher call complete custom datsloader len {len(custom_trainloader)}")
        
        for _ in range(len(custom_trainloader) - 1): #targets
            if samples == None or targets == None : # stop training in no dataset occasion. for avoiding error.
                break
            count += 1
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(samples)
            
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            print("loss", losses.item())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict, True)
            # if loss_dict_reduced == False: #생각해보니 필요없는 부분(Rehearsal 부분은 항상 True를 가지기 때문에 loss_dict_reduced가 False가 나올 일이 없다.)
            #     loss_dict_reduced = _back_training(samples, targets, model, criterion)
                
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                        for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()
            sum_loss += loss_value
            print(f"losses : {loss_value:05f}, epoch_total_loss : {(sum_loss / count):05f}, count : {count}")
            
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                samples, targets = prefetcher.next()
                continue

            optimizer.zero_grad()
            losses.backward()
            
            if max_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            else:
                grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
            optimizer.step()
            samples, targets = prefetcher.next()
            
    del rehearsal_class
    os.rename(file_name + "_NEW", file_name + "_OLD")
    
@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()
    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        print('output', outputs)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    #metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
