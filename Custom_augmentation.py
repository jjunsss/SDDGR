import torch
import numpy as np
from typing import Dict, List
import random
import matplotlib.pyplot as plt
import albumentations as A
import cv2
import copy
import matplotlib.pyplot as plt
from util.box_ops import box_cxcywh_to_xyxy_resize, box_xyxy_to_cxcywh
import datasets.transforms as T
import copy
def visualize_bboxes(img, bboxes, img_size = (1024, 1024), vertical = False):
    # min_or = img.min()
    # max_or = img.max()
    # img_uint = ((img - min_or) / (max_or - min_or) * 255.).astype(np.uint8)
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    h, w = img_size
    #img = img[..., ::-1]
    if vertical == False:
        for bbox in bboxes:
            xmin, ymin, xmax, ymax, cls = bbox * torch.tensor((w , h, w, h, 1))
            cv2.rectangle(img,(int(xmin.item()), int(ymin.item())),(int(xmax.item()), int(ymax.item())), (255, 0, 0), 3)
            label = f'Class {cls.item()}'
            cv2.putText(img, label, (int(xmin.item()), int(ymin.item()) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        cv2.imwrite("./Combined_"+str(bboxes[0][-1])+".png",img)
    else:
        img = (img * 255).astype(np.uint8)
        # #bboxes = bboxes['boxes']
        # bboxes = box_cxcywh_to_xyxy_resize(bboxes)
        # x1, y1, x2, y2 = bboxes.unbind(-1)
        # bboxes = torch.stack([x1, y1, x2, y2 ], dim=-1).tolist()
        # for bbox in bboxes:
        #     xmin, ymin, xmax, ymax = torch.tensor(bbox) * torch.tensor((w , h, w, h))
        #     cv2.rectangle(img,(int(xmin.item()), int(ymin.item())),(int(xmax.item()), int(ymax.item())), (255, 0, 0), 3)
        cv2.imwrite("./vertical_"+str(bboxes[0][-1])+".png", img)
        
import torchvision.transforms.functional as F
class CCB(object):
    def __init__(self, image_size, Continual_Batch = 2):
        self.img_size = image_size
        self.transformed = origin_transform("custom")
        self.Continual_Batch = Continual_Batch
        
    def __call__(self, image_list, target_list):
        if self.Continual_Batch == 3:
            Cur_img, Cur_lab, Dif_img, Dif_lab = self._load_mosaic(image_list, target_list)#np.array(original) / norm coord torch.tensor / np.array(original) / norm coord torch.tensor
            Cur_img, Cur_lab = self.transformed(Cur_img, Cur_lab) #Adapt Normalization(Nomalized image and ToTensor)
            Dif_img, Dif_lab = self.transformed(Dif_img, Dif_lab) #Adapt Normalization(Nomalized image and ToTensor)
            #visualize_bboxes(np.clip(Dif_img.permute(1, 2, 0).numpy(), 0, 1).copy(), Dif_lab['boxes'], self.img_size, True)
            return Cur_img, Cur_lab, Dif_img, Dif_lab
        
        if self.Continual_Batch == 2:
            while True : 
                result = self._load_mosaic(image_list, target_list)
                if result != False:
                    Cur_img, Cur_lab, _, _ = result
                    break
            Cur_img, Cur_lab = self.transformed(Cur_img, Cur_lab)
            # visualize_bboxes(Cur_img.permute(1, 2, 0).numpy().copy(), Cur_lab['boxes'], Cur_img.shape[:-1], True)
            return Cur_img, Cur_lab
    
   
    def _load_mosaic(self, image_list, target_list):
        '''
            Current_mosaic_index : For constructing masaic about current classes
            Diff_mosaic_index : For constructing mosaic abhout differenct classes (Not Now classes)
            Current_bboxes : numpy array. [cls, cx, cy, w, h] for current classes
            Diff_bboxes : numpy array. [cls, cx, cy, w, h] for different classes (Not Now classes)
        '''
        # loads images in a mosaic
        Mosaic_size = self.img_size #1024, im_w, im_h : 1024
        #self.mosaic_border = [Mosaic_size[0] // 2, Mosaic_size[1] // 2] # height . width
        Current_mosaic_img, Current_mosaic_labels = self._make_batch_mosaic(image_list, target_list, Mosaic_size)
        Current_mosaic_labels = self._make_resized_targets(Current_mosaic_labels)
        
        # Current_mosaic_img, mosaic_labels = _crop(Current_mosaic_img, temp_labels['boxes'], temp_labels['labels']) # To 480, 640 random cropiing
        # if mosaic_labels.shape[-1] != 5:
        #     return False
        # mosaic_labels = self._make_resized_targets(mosaic_labels)
        
        if self.Continual_Batch == 3: #For 3 CBB Training
            Diff_mosaic_labels = copy.deepcopy(Current_mosaic_labels)
            Diff_mosaic_img, Diff_bbox, Diff_labels  = _HorizontalFlip(Current_mosaic_img, Current_mosaic_labels['boxes'], Current_mosaic_labels['labels'])
            Diff_mosaic_labels = self._make_resized_targets(Diff_bbox, Diff_labels)
            return Current_mosaic_img, Current_mosaic_labels, Diff_mosaic_img, Diff_mosaic_labels
        return Current_mosaic_img, Current_mosaic_labels, None, None #For 2 CBB Training

    def _make_batch_mosaic(self, image_list, target_list, mosaic_size ):
        mosaic_aug_labels = []
        yc, xc = (int(random.uniform(x/4, x/4*3)) for x in mosaic_size) #normd coords
        for i, (img, target) in enumerate(zip(image_list, target_list)):

            # place img in img4(특정 center point 잡아서 할당)
            if i == 0:  # top left
                mosaic_aug_img = np.full((mosaic_size[0], mosaic_size[1], 3), 114, dtype=np.uint8)  # base image with 4 tiles
                xaxs_scale = xc
                yaxs_scale = yc
                transposed_img, transposed_bboxes = self._augment_bboxes(img, target, xaxs_scale, yaxs_scale)
                height, width, channel, = transposed_img.shape
                temp_bbox = transposed_bboxes.clone().detach()
                
                heigth_rate = height / mosaic_size[0] 
                width_rate = width  / mosaic_size[1]
                
                temp_bbox[:, :-1:2] *= width_rate
                temp_bbox[:, 1:-1:2] *= heigth_rate
                mosaic_aug_img[:yc, :xc, :] = transposed_img
                mosaic_bboxes = temp_bbox.clone().detach()
                continue
            elif i == 1:  # top right
                xaxs_scale = mosaic_size[1] - xc
                yaxs_scale = yc
                transposed_img, transposed_bboxes = self._augment_bboxes(img, target, xaxs_scale, yaxs_scale)
                height, width, channel, = transposed_img.shape
                temp_bbox = transposed_bboxes.clone().detach()
                
                heigth_rate = height / mosaic_size[0] 
                width_rate = width  / mosaic_size[1]
                temp_bbox[:, :-1:2] *= width_rate
                temp_bbox[:, 1:-1:2] *= heigth_rate
                
                mosaic_aug_img[:yc, xc:, :] = transposed_img
                temp_bbox[:, 0] += (xc / mosaic_size[1])
                temp_bbox[:, 2] += (xc / mosaic_size[1])
            elif i == 2:  # bottom left
                xaxs_scale = xc
                yaxs_scale = mosaic_size[0] - yc
                transposed_img, transposed_bboxes = self._augment_bboxes(img, target, xaxs_scale, yaxs_scale)
                height, width, channel, = transposed_img.shape
                temp_bbox = transposed_bboxes.clone().detach()                
                
                heigth_rate = height / mosaic_size[0] 
                width_rate = width  / mosaic_size[1]
                temp_bbox[:, :-1:2] *= width_rate
                temp_bbox[:, 1:-1:2] *= heigth_rate
                
                mosaic_aug_img[yc:, :xc, :] = transposed_img
                temp_bbox[:, 1] += (yc / mosaic_size[0])
                temp_bbox[:, 3] += (yc / mosaic_size[0])
            elif i == 3:  # bottom right
                xaxs_scale = mosaic_size[1] - xc
                yaxs_scale = mosaic_size[0] - yc
                transposed_img, transposed_bboxes = self._augment_bboxes(img, target, xaxs_scale, yaxs_scale)
                height, width, channel, = transposed_img.shape
                temp_bbox = transposed_bboxes.clone().detach()                
                
                heigth_rate = height / mosaic_size[0] 
                width_rate = width  / mosaic_size[1]
                temp_bbox[:, :-1:2] *= width_rate
                temp_bbox[:, 1:-1:2] *= heigth_rate
                                
                mosaic_aug_img[yc:, xc:, :] = transposed_img
                temp_bbox[:, :-1:2] += (xc / mosaic_size[1])
                temp_bbox[:, 1:-1:2] += (yc / mosaic_size[0])
                
            mosaic_bboxes = torch.vstack((temp_bbox, mosaic_bboxes))
        # visualize_bboxes(mosaic_aug_img, mosaic_bboxes, self.img_size)
        return mosaic_aug_img, mosaic_bboxes
    
    def _augment_bboxes(self, img:np.array, target:torch.tensor, xc, yc): #* Checking
        '''
            maybe index_list is constant shape in clockwise(1:origin / 2:Current Img / 3: Currnt image / 4: Current img)
        '''
        boxes = target["boxes"] #* Torch tensor
        classes = target["labels"]
        
        boxes = box_cxcywh_to_xyxy_resize(boxes)
        x1, y1, x2, y2 = boxes.unbind(-1)
        bboxes = torch.stack([x1, y1, x2, y2, classes.long()], dim=-1)
        #bboxes = torch.stack([x1, y1, x2, y2, classes.long()], dim=-1).tolist()

        transposed_img, transposed_bboxesd = self._Resize_for_batchmosaic(img, bboxes, xc, yc)
        
        return transposed_img, transposed_bboxesd
    
    def _Resize_for_batchmosaic(self, img:np.array, bboxes:torch.tensor, xc, yc): #* Checking
        """
            img : torch.tensor(Dataset[idx])
            resized : size for resizing
            BBoxes : resize image to (height, width) in a image
        """
        #이미지 변환 + Box Label 변환
        temp_img = copy.deepcopy(img)
        bboxes[:, :-1].clamp_(min = 0.0, max=1.0)
        bboxes.tolist()
        
        transform = A.Compose([
            A.Resize(yc, xc)
        ], bbox_params=A.BboxParams(format='albumentations'))

        #Annoation change
        transformed = transform(image = temp_img, bboxes = bboxes)
        transformed_bboxes = transformed['bboxes']
        transformed_img = transformed["image"]
        #visualize_bboxes(transformed_img, transformed_bboxes)
        
        transformed_bboxes = torch.tensor(transformed_bboxes)
        #transformed_img = torch.tensor(transformed_img, dtype=torch.float32).permute(2, 0, 1) #TODO: change dimension permunate for training in torch image
        
        return  transformed_img, transformed_bboxes 

    def _make_resized_targets(self, target: Dict, v_labels: torch.tensor = None)-> Dict:
        
        temp_dict = {}
        if v_labels is not None :
            boxes = target
            labels = v_labels
        else:
            boxes = target[:, :-1]
            labels = target[:, -1]
        cxcy_boxes = box_xyxy_to_cxcywh(boxes)
        temp_dict['boxes'] = cxcy_boxes.to(dtype=torch.float32)
        temp_dict['labels'] = labels.to(dtype=torch.long)
        temp_dict['images_id'] = torch.tensor(0)
        temp_dict['area'] = torch.tensor(0)
        temp_dict['iscrowd'] = torch.tensor(0)
        temp_dict['orig_size'] = torch.tensor(self.img_size)
        temp_dict['size'] = torch.tensor(self.img_size)
        
        return temp_dict
    
def _HorizontalFlip(img:np.array, bboxes, labels): #* Checking
    """
        img : torch.tensor(Dataset[idx])
        resized : size for resizing
        BBoxes : resize image to (height, width) in a image
    """
 
    boxes = box_cxcywh_to_xyxy_resize(bboxes)
    boxes[:, :-1].clamp_(min = 0.0, max=1.0)
    x1, y1, x2, y2 = boxes.unbind(-1)
    
    boxes = torch.stack([x1, y1, x2, y2, labels], dim=-1).tolist()
    
    # bboxes = bboxes.tolist()
    class_labels = labels.tolist()
    temp_img = copy.deepcopy(img)
    
    transform = A.Compose([
        A.HorizontalFlip(1),
        #A.VerticalFlip(0.1),
    ], bbox_params=A.BboxParams(format='albumentations'))
    
    #Annoation change
    transformed = transform(image = temp_img, bboxes = boxes)
    transformed_bboxes = transformed['bboxes']
    transformed_img = transformed["image"]
    
    temp = torch.tensor(transformed_bboxes)
    transformed_bboxes = temp[:, :-1]
    transformed_labels = temp[:, -1]
    transformed_img = torch.tensor(transformed_img, dtype=torch.float32).permute(2, 0, 1)
    #visualize_bboxes(np.clip(transformed_img.permute(1, 2, 0).numpy(), 0, 1).copy(), transformed_bboxes, (1024, 1024), True)
    return  transformed_img, transformed_bboxes, transformed_labels 
