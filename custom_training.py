"""
Train and eval functions used in main.py
"""
import math
import numpy as np
import os
import sys
import random
import torch.distributed as dist
import torch
import util.misc as utils
from custom_utils import *
from custom_buffer_manager import *
from custom_prints import check_losses
import os
from typing import Tuple, Collection, Dict, List
import torch
import util.misc as utils
import numpy as np
from typing import Tuple, Dict, List, Optional

from torch.cuda.amp import autocast
from custom_fake_target import normal_query_selc_to_target
from models import inference_model
from tqdm import tqdm
from copy import deepcopy

def training(args, task_idx, last_task, epo, idx, count, sum_loss, samples, targets, 
                      model: torch.nn.Module, teacher_model, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer,  
                      current_classes, training_mode): 

    device = torch.device("cuda")
    ex_device = torch.device("cpu")
    count, sum_loss = _common_training(args, epo, idx, task_idx, last_task, count, sum_loss, 
                                        samples, targets, model, optimizer,
                                        teacher_model, criterion, device, ex_device, current_classes, training_mode)

    del samples, targets
    return sum_loss, count


#TODO: generated image filter operation. necessary filtering.
def _common_training(args, epo, idx, task_idx, last_task, count, sum_loss, 
                     samples, targets, model: torch.nn.Module, optimizer:torch.optim.Optimizer,
                     teacher_model, criterion: torch.nn.Module, device, ex_device, current_classes, t_type=None):
    model.train()
    criterion.train()

    #* teacher distllation
    if last_task and args.Distill:
        teacher_model.eval()
        teacher_model.to(device)
        teacher_attn = compute_attn_weight(teacher_model, model, samples, device, ex_device)
        teacher_model.to("cpu")

    #* fake query selection(pseudo labeling)
    if args.Fake_Query:
        teacher_model.eval()
        teacher_model.to(device)
        teacher_outputs = teacher_model(samples)
        targets = normal_query_selc_to_target(teacher_outputs, targets, current_classes)  # Adjust this line as necessary
        teacher_model.to("cpu")
        
    #* current training outputs
    outputs = inference_model(args, model, samples, targets, teacher_attn)


    loss_dict = criterion(outputs, targets)
    weight_dict = criterion.weight_dict
    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
    losses_value = losses.item()
    
    #* call every loss at each GPUs
    loss_dict_reduced = utils.reduce_dict(loss_dict, train_check=True)
    if loss_dict_reduced:

        count += 1
        loss_dict_reduced_scaled = {v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled)

        sum_loss += losses_reduced_scaled
        if utils.is_main_process(): #sum_loss가 GPU의 개수에 맞춰서 더해주고 있으니,
            check_losses(epo, idx, losses_reduced_scaled, sum_loss, count, current_classes, None)
            print(f" {t_type} \t {{ task: {task_idx}, epoch : {epo} \t Loss : {losses_value:.4f} \t Total Loss : {sum_loss/count:.4f} }}")
    
    optimizer.zero_grad()
    losses.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
    optimizer.step()

    return count, sum_loss


@torch.no_grad()
def icarl_feature_extractor_setup(args, model):
    '''
        In iCaRL, buffer manager collect samples closed to the mean of features of corresponding class.
        This function set up feature extractor for collecting.
    '''
    if args.distributed:
        feature_extractor = deepcopy(model.module.backbone)
    else:
        feature_extractor = deepcopy(model.backbone) # distributed:model.module.backbone
    
    for n, p in feature_extractor.named_parameters():
        p.requires_grad = False

    return feature_extractor


@torch.no_grad()
def icarl_prototype_setup(args, feature_extractor, device, current_classes):
    '''
        In iCaRL, buffer manager collect samples closed to the mean of features of corresponding class.
        This function set up prototype-mean of features of corresponding class-.
        Prototype can be the 'criteria' to select closest samples.
    '''
    
    feature_extractor.eval()
    proto = defaultdict(int)

    for cls in current_classes:
        _dataset, _data_loader, _sampler = IcarlDataset(args=args, single_class=cls)
        if _dataset == None:
            continue
        
        _cnt = 0
        for samples, targets, _, _ in tqdm(_data_loader, desc=f'Prototype:class_{cls}', disable=not utils.is_main_process()):
            samples = samples.to(device)
            feature, _ = feature_extractor(samples)
            feature_0 = feature[0].tensors
            proto[cls] += feature_0
            _cnt += 1
            if args.debug and _cnt == 10:
                break

        try:
            proto[cls] = proto[cls] / _dataset.__len__()
        except ZeroDivisionError:
            pass
        if args.debug and cls == 10:
            break

    return proto


@torch.no_grad()
def icarl_rehearsal_training(args, samples, targets, fe: torch.nn.Module, proto: Dict, device:torch.device,
                       rehearsal_classes, current_classes):
    '''
        iCaRL buffer collection.

        rehearsal_classes : [feature_sum, [[image_ids, difference] ...]]
        TODO: move line:200~218 to construct_rehearsal
    '''

    fe.eval()
    samples.to(device)

    feature, pos = fe(samples)
    feat_tensor = feature[0].tensors # TODO: cpu or cuda?

    for bt_idx in range(feat_tensor.shape[0]):
        feat_0 = feat_tensor[bt_idx]
        target = targets[bt_idx]
        label_tensor = targets[bt_idx]['labels']
        label_tensor_unique = torch.unique(label_tensor)
        label_list_unique = label_tensor_unique.tolist()

        for label in label_list_unique:
            try:
                class_mean = proto[label]
            except KeyError:
                print(f'label: {label} don\'t in prototype: {proto.keys()}')
                continue
            try :
                if label in rehearsal_classes: # rehearsal_classes[label] exist
                    rehearsal_classes[label][0] = rehearsal_classes[label][0].to(device)

                    exemplar_mean = (rehearsal_classes[label][0] + feat_0) / (len(rehearsal_classes[label]) + 1)
                    difference = torch.mean(torch.sqrt(torch.sum((class_mean - exemplar_mean)**2, axis=1))).item()

                    rehearsal_classes[label][0] = rehearsal_classes[label][0]                   
                    rehearsal_classes[label][0]+= feat_0
                    rehearsal_classes[label][1].append([target['image_id'].item(), difference])

                else :
                    #"initioalization"
                    difference = torch.argmin(torch.sqrt(torch.sum((class_mean - feat_0)**2, axis=0))).item() # argmin is true????
                    rehearsal_classes[label] = [feat_0, [[target['image_id'].item(), difference], ]]
            except Exception as e:
                print(f"Error opening image: {e}")
                difference = torch.argmin(torch.sqrt(torch.sum((class_mean - feat_0)**2, axis=0))).item() # argmin is true????
                rehearsal_classes[label] = [feat_0, [[target['image_id'].item(), difference], ]]
            
            rehearsal_classes[label][1].sort(key=lambda x: x[1]) # sort with difference

    # construct rehearsal (3) - reduce exemplar set
    # for label, data in tqdm(rehearsal_classes.items(), desc='Reduce_exemplar:', disable=not utils.is_main_process()):
    #     try:
    #         data[1] = data[1][:args.limit_image]
    #     except:
    #         continue

    return rehearsal_classes


def rehearsal_training(args, samples, targets, model: torch.nn.Module, criterion: torch.nn.Module, 
                       rehearsal_classes, current_classes):
    '''
        replay를 위한 데이터를 수집 시에 모델은 영향을 받지 않도록 설정
    '''
    model.eval() # For Fisher informations
    criterion.eval()
    
    device = torch.device("cuda")
    ex_device = torch.device("cpu")
    model.to(device)
    # samples, targets = _process_samples_and_targets(samples, targets, device)

    outputs = inference_model(args, model, samples, targets, eval=True)
    # TODO : new input to model. plz change dn-detr model input (self.buffer_construct_loss)
    
    _ = criterion(outputs, targets, buffer_construct_loss=True)
    
    # This is collect replay buffer
    with torch.no_grad():
        batch_loss_dict = {}
        
        # Transform tensor to scarlar value for rehearsal step
        # This values sorted by batch index so first add all loss and second iterate each batch loss for update and lastly 
        # calculate all fisher information for updating all parameters
        batch_loss_dict["loss_bbox"] = [loss.item() for loss in criterion.losses_for_replay["loss_bbox"]]
        batch_loss_dict["loss_giou"] = [loss.item() for loss in criterion.losses_for_replay["loss_giou"]]
        batch_loss_dict["loss_labels"] = [loss.item() for loss in criterion.losses_for_replay["loss_labels"]]
    
        targets = [{k: v.to(ex_device) for k, v in t.items()} for t in targets]
        rehearsal_classes = construct_rehearsal(args, losses_dict=batch_loss_dict, targets=targets,
                                                rehearsal_dict=rehearsal_classes, 
                                                current_classes=current_classes,
                                                least_image=args.least_image,
                                                limit_image=args.limit_image)
    
    
    if utils.get_world_size() > 1: dist.barrier()
    return rehearsal_classes



def _process_samples_and_targets(samples, targets, device):
    samples = samples.to(device)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    return samples, targets