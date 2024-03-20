import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
import random
# from pycocotools.coco import COCO
# from visuliazation import *?
# COCO 인스턴스 생성
# coco = COCO("/data/jjunsss/COCODIR/annotations/instances_train2017.json")

def normal_query_selc_to_target(outputs, targets, current_classes):
    out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
    
    prob = out_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 30, dim=1)
    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    boxes = torch.gather(out_bbox, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
    results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
    threshold = 0.3
    current_classes = min(current_classes)
    
    for idx, (target, result) in enumerate(zip(targets, results)):
        if target["labels"][target["labels"] < current_classes].shape[0] > 0:
            continue
        
        scores = result["scores"][result["scores"] > threshold].detach()
        labels = result["labels"][result["scores"] > threshold].detach() 
        boxes = result["boxes"][result["scores"] > threshold].detach()

        if labels[labels < current_classes].size(0) > 0:
            image_id = target["image_id"].item()
             
            addlabels = labels[labels < current_classes]
            addboxes = boxes[labels < current_classes]
            area = addboxes[:, 2] * addboxes[:, 3]
            
            targets[idx]["boxes"] = torch.cat((target["boxes"], addboxes))
            targets[idx]["labels"] = torch.cat((target["labels"], addlabels))
            targets[idx]["area"] = torch.cat((target["area"], area))
            targets[idx]["iscrowd"] = torch.cat((target["iscrowd"], torch.tensor([0], device = torch.device("cuda"))))
            
            print("fake query operation")
        
    return targets


def only_oldset_mosaic_query_selc_to_target(outputs, targets, current_classes):
    out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
    
    prob = out_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 30, dim=1)
    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    boxes = torch.gather(out_bbox, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
    results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
    threshold = 0.5
    current_classes = min(current_classes)
    for target, result in zip(targets, results):
        if target["labels"][target["labels"] >= current_classes].shape[0] > 0: #New Class에서만 동작하도록 구성
            continue
        
        scores = result["scores"][result["scores"] > threshold]
        labels = result["labels"][result["scores"] > threshold] 
        boxes = result["boxes"][result["scores"] > threshold]
        
        if labels[labels >= current_classes].size(0) > 0:
            addlabels = labels[labels >= current_classes]
            addboxes = boxes[labels >= current_classes]
            area = addboxes[:, 2] * addboxes[:, 3]   
            random_contorl = random.uniform(-1e-10, +1e-10)
            addboxes += random_contorl
            print("new fake query operation")
            target["boxes"] = torch.cat((target["boxes"], addboxes))
            target["labels"] = torch.cat((target["labels"], addlabels))            
            
    return targets

import os
def pseudo_target(outputs, count=0, min_class=0, max_class=0):
    #! target original information 
    out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
    # image_id = image_name.split(".").pop(0)
    # ann_ids = gligen_frame.getAnnIds(imgIds=int(os.path.splitext(image_id)[0]))
    num_annotations = 5 #* max number of annotations in one image
    
    
    prob = out_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_annotations, dim=1) #* select topk in each batch/ torch.Size([3, 27300])
    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2] #* learned query in each batch. // out_logits.shape[2] means that calucales object query index in 300 x 91 = 27300
    labels = topk_indexes % out_logits.shape[2]
    boxes = torch.gather(out_bbox, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4)) #* out_bbox is [batch, 300, 4]. torch.gather select index value to target. 
                                                                              #* pick the values in bbox coordinates that is correspond to the topk_boxes.
    result = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)].pop()
        
    threshold = 0.8
    if count >= 2 : #3-7 / 5=6 / 7=5 / 9 = 4(10)
        step = count - 1
        interval = 0.05
        threshold = threshold - (step * interval)
        
        if threshold <= 0.4  : threshold = 0.4
        # print(f"---- generation count : {count} - pseudo target confidence threshold : {threshold} ----")            
        
    scores = result["scores"][result["scores"] > threshold].detach()
    labels = result["labels"][result["scores"] > threshold].detach() 
    boxes = result["boxes"][result["scores"] > threshold].detach() #* cx,cy,w ,h
    
    
    addlabels = labels[labels <= max_class]
    addboxes = boxes[labels <= max_class]
    if addboxes.size(0) == 0:  # threshold를 넘는 boxes가 없는 경우
        return None, None, None, threshold
    
    #* cx, cy, w, h -> x, y, w, h
    addboxes[:, 0] = addboxes[:, 0] - (addboxes[:, 2] / 2)
    addboxes[:, 1] = addboxes[:, 1] - (addboxes[:, 3] / 2)
    addboxes = torch.clamp(addboxes, min=0)
    refined_labels, refined_boxes =  addlabels, addboxes
    
    refined_areas = refined_boxes[:, 2] * refined_boxes[:, 3]
    # draw_boxes_on_image(image, target, f'Updated_{image_id}')
        
    return refined_labels, refined_areas, refined_boxes, threshold

from torchvision.ops import box_iou
def refine_predictions(gligen_frame, image_name, labels, boxes):
    image_id = int(os.path.splitext(image_name)[0])
    ann_ids = gligen_frame.getAnnIds(imgIds=image_id)
    gt_annotations = gligen_frame.loadAnns(ann_ids)
    #* coco format x, y, w, h  + coco is original size format -> have to change normalized size
    #* boxes(predict) format : cx, cy, w ,h
    gt_boxes = torch.tensor([ann['bbox'] for ann in gt_annotations], dtype=torch.float32)  # Convert to [x, y, w, h] format
    gt_boxes = gt_boxes / 512
    
    gt_labels = torch.tensor([ann['category_id'] for ann in gt_annotations], dtype=torch.int64)
    
    boxes[:, 0] = boxes[:, 0] - (boxes[:, 2] / 2)
    boxes[:, 1] = boxes[:, 1] - (boxes[:, 3] / 2)
    boxes = boxes.detach().cpu()
    # Calculate IoU between predicted boxes and ground truth boxes
    ious = box_iou(gt_boxes, boxes)
    max_iou, max_iou_index = ious.max(dim=1)
    
    refined_boxes = []
    refined_labels = []
    
    for i, (iou, index) in enumerate(zip(max_iou, max_iou_index)):
        if iou == 0:  #*If no matching ground truth box is found, remove the GT
            continue
        
        if iou > 0.6:  #*If IoU is grrefined_areaster than threshold, refine the box
            pred_labels = gt_labels[i]
            pred_box = boxes[index]
            
            refined_boxes.append(pred_box)
            refined_labels.append(pred_labels)
            
        else:  #* If IoU is below the threshold but not zero, keep the prediction as it is
            pred_box = boxes[index]
            pred_labels = labels[index]
            
            refined_boxes.append(pred_box)
            refined_labels.append(pred_labels)

    if not refined_boxes or not refined_labels:  # If no boxes or labels are left after refinement
        return labels, boxes
    
    return torch.stack(refined_labels), torch.stack(refined_boxes)


