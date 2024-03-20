from xmlrpc.client import Boolean
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from torch.utils.data import DataLoader, ConcatDataset
import datasets.samplers as samplers
import torch
import numpy as np
from typing import Tuple, Collection, Dict, List
import bisect
import random
import matplotlib.pyplot as plt
import albumentations as A
import cv2
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from util.box_ops import box_cxcywh_to_xyxy_resize, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from datasets.coco import CBB_transform

def visualize_bboxes(img, bboxes, img_size = 0):
    min_or = img.min()
    max_or = img.max()
    img_uint = ((img - min_or) / (max_or - min_or) * 255.).astype(np.uint8)
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    h, w = img_size
    for bbox in bboxes:
        xmin, ymin, xmax, ymax, cls = bbox * torch.tensor((w , h, w, h, 1))
        cv2.rectangle(img_uint,(int(xmin.item()), int(ymin.item())),(int(xmax.item()), int(ymax.item())), (255, 0, 0), 3)
    cv2.imwrite("./Combined_"+str(bboxes[0][-1])+".png",img_uint)
    

def Incre_Dataset(Task_Num, args, Incre_Classes):    
    current_classes = Incre_Classes[Task_Num]
    print(f"current_classes : {current_classes}")
    
    if len(Incre_Classes) == 1:
        dataset_train = build_dataset(image_set='train', args=args, class_ids=None) #* Task ID에 해당하는 Class들만 Dataset을 통해서 불러옴
    else: 
        if Task_Num == 0 : #* First Task training
            dataset_train = build_dataset(image_set='train', args=args, class_ids=current_classes)
        else:
            dataset_train = build_dataset(image_set='train', args=args, class_ids=current_classes)
    dataset_val = build_dataset(image_set='val', args=args)
    
    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        
    print(f"dataset config :{dataset_train}")
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)
    
    return dataset_train, data_loader_train, sampler_train, current_classes


def DivideTask_for_incre(Task_Counts: int, Total_Classes: int, DivisionOfNames: Boolean):
    '''
        DivisionofNames == True인 경우 Task_Counts는 필요 없어짐 Domain을 기준으로 class task가 자동 분할
        False라면 Task_Counts, Total_Classes를 사용해서 적절하게 분할
        #Task : 테스크의 수
        #Total Class : 총 클래스의 수
        #DivisionOfNames : Domain을 사용해서 분할
    '''
    if DivisionOfNames is True:
        Divided_Classes = []
        Divided_Classes.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]) #DID
        Divided_Classes.append([28, 32, 35, 41, 56]) #photozone
        Divided_Classes.append([22, 23, 24, 25, 26, 27, 29, 30, 31, 33,34,36, 37, 38, 39, 40,42,43,44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 57, 58, 59]) #VE
        return Divided_Classes
    
    classes = [idx+1 for idx in range(Total_Classes)]
    Task = int(Total_Classes / Task_Counts)
    Rest_Classes_num = Total_Classes % Task_Counts
    
    start = 0
    end = Task
    Divided_Classes = []
    for _ in range(Task_Counts):
        Divided_Classes.append(classes[start:end])
        start += Task
        end += Task
    if Rest_Classes_num != 0:
        Rest_Classes = classes[-Rest_Classes_num:]
        Divided_Classes[-1].extend(Rest_Classes)
    
    return Divided_Classes

#현재 (Samples, Targets)의 정보를 가진 형태로 데이터가 구성되어 있음(딕셔너리로 각각의 Class 정보를 가진 채로 구성됨)
#참고로 Samples -> NestedTensor, Target -> List 형태로 구성되어 있음 다만 1개의 
class CustomDataset(torch.utils.data.Dataset):
    
    def __init__(self, args, re_dict, old_classes):
        self.re_dict = re_dict
        self.keys = list(self.re_dict.keys()) #image_id
        self.old_classes = old_classes
        self.datasets = build_dataset(image_set='train', args=args, class_ids=old_classes, img_ids=self.keys) 
    
    def __len__(self):
        return len(self.datasets)
    
    def __repr__(self):
        print(f"Data key presented in buffer : {self.old_classes}")    

    def __getitem__(self, idx):
        samples, targets, new_samples, new_targets = self.datasets[idx]

        return samples, targets, new_samples, new_targets

def _Resize_for_batchmosaic(img:torch.Tensor, height_resized, width_resized, bboxes): #* Checking
    """
        img : torch.tensor(Dataset[idx])
        resized : size for resizing
        BBoxes : resize image to (height, width) in a image
    """
    transform = A.Compose([
        A.Resize(height_resized, width_resized)
    ], bbox_params=A.BboxParams(format='albumentations'))

    #Annoation change
    transformed = transform(image=img.permute(1, 2, 0).numpy(), bboxes=bboxes)
    transformed_bboxes = torch.tensor(transformed['bboxes'])
    transformed_img = torch.tensor(transformed["image"], dtype=torch.float32).permute(2, 0, 1)
    #visualize_bboxes(transformed_img, transformed_bboxes)
    
    return  transformed_img, transformed_bboxes 
    
class BatchMosaicAug(torch.utils.data.Dataset):
    def __init__(self, datasets, CurrentClasses, transform, Mosaic=False ):
        self.Datasets = datasets
        self.current_classes = CurrentClasses
        self.Confidence = 0
        self.Mosaic = Mosaic
        self.img_size = (800, 1024) #변경될 크기(이미지 변경을 위함)
        self.Rehearsal_dataset = datasets.datasets[0] #* Old Data1set
        self.Current_dataset = datasets.datasets[1] #* New Dataset
        self._transform = transform
        
    def __len__(self):
            return len(self.Datasets)    
        
    def __getitem__(self, index):
        img, target, origin_img, origin_target = self.Datasets[index]
        if random.random() > self.Confidence: #! For Randomness
            self.Mosaic = True

        if self.Mosaic == True :
            Current_mosaic_index, Diff_mosaic_index, original_index = self._Mosaic_index(index,)
            Cur_img, Cur_lab, Dif_img, Dif_lab = self.load_mosaic(Current_mosaic_index, Diff_mosaic_index, original_index)
            Cur_img, Cur_lab = self._transform(Cur_img, Cur_lab)
            Dif_img, Dif_lab = self._transform(Dif_img, Dif_lab)
            
            return img, target, origin_img, origin_target, Cur_img, Cur_lab, Dif_img, Dif_lab
        else:
            return img, target, origin_img, origin_target, None, None, None, None

    def _augment_bboxes(self, index, diff, original_id): #* Checking
        '''
            maybe index_list is constant shape in clockwise(1:origin / 2:Current Img / 3: Currnt image / 4: Current img)
        '''
        #bboxes = []
        if diff == True and index != original_id:
            boxes = self.Rehearsal_dataset[index][3]["boxes"]
            classes = self.Rehearsal_dataset[index][3]["labels"]
        else:
            _, _, _, origin_target = self.Datasets[index]
            boxes = origin_target["boxes"] #* Torch tensor
            classes = origin_target["labels"]
        boxes = box_cxcywh_to_xyxy_resize(boxes)
        
        x1, y1, x2, y2 = boxes.unbind(-1)
        bboxes = torch.stack([x1, y1, x2, y2, classes.long()], dim=-1).tolist()
        
        # for box, cls in zip(boxes, classes):
        #     x1, y1, x2, y2 = box
        #     bboxes.append([x1, y1, x2, y2, int(cls)])
        img, _, _ = self.load_image(index, diff, original_id)
        transposed_img, transposed_bboxesd = _Resize_for_batchmosaic(img, int(self.img_size[0]/2), int(self.img_size[1]/2), bboxes)
        
        return transposed_img, transposed_bboxesd

    def _Mosaic_index(self, index): #* Done
        '''
            Only Mosaic index printed 
        '''
        #*Curretn Class augmentation / Other class AUgmentation
        #Mosaic_index = random.sample(range(len(self.Current_dataset)), 3)
        Mosaic_index = random.sample(range(len(self.Rehearsal_dataset)), 3)
        Rehearsal_index = random.sample(range(len(self.Rehearsal_dataset)), 3)
        
        # for now, old in zip(Mosaic_index, Rehearsal_index):
        #     if self.Rehearsal_dataset[old][1]["boxes"].shape[-1] < 1 and self.Datasets[now][1]["boxes"] < 1:
        #         raise Exception("Error: NOT correct Dataset")
            
        Mosaic_index.insert(0, index)
        Rehearsal_index.insert(0, index)

        return random.sample(Mosaic_index, len(Mosaic_index)), random.sample(Rehearsal_index, len(Rehearsal_index)), index
    
    #* No Augmentation before BatchClassAugmentation Method
    def load_image(self, index, diff, original_id:int):#* Done
        if (diff == True) and (index != original_id):
            img = self.Rehearsal_dataset[index][2]
            h0, w0 = self.Rehearsal_dataset[index][3]["orig_size"].tolist()
        else:
            # loads 1 image from dataset, returns img, original hw, resized hw
            img = self.Datasets[index][2] #* Original Image
            h0, w0 = self.Datasets[index][3]["orig_size"].tolist()

        #h0, w0 = origin_shape[0].item(), origin_shape[1].item()  # orig hw
        return img.squeeze(), (h0, w0), img.shape[1:]  # img, hw_original, hw_resized(height, Width)
    
    def make_batch_mosaic(self, mosaic_index, mosaic_size, diff, original_id):
        for i, index in enumerate(mosaic_index):
            # Load image
            transposed_img, transposed_bboxes = self._augment_bboxes(index, diff, original_id) #! cv2.imread 통해서 불러옴. 나는 coco 사용하기에 변경해야 함.
            channel, height, width = transposed_img.shape
            temp_bbox = transposed_bboxes.clone().detach()
            temp_bbox[:, :] /= 2
            # place img in img4(특정 center point 잡아서 할당)
            if i == 0:  # top left
                mosaic_aug_img = torch.zeros((channel, mosaic_size[0], mosaic_size[1]), dtype=torch.float32)  # base image with 4 tiles
                mosaic_aug_img[:, :height, :width] = transposed_img
                # temp_bbox[:, 0] =(temp_bbox[:, 0]) / 2 #? x1 (xmin)
                # temp_bbox[:, 1] = (temp_bbox[:, 1]) / 2 #? y1 (ymin)
                # temp_bbox[:, 2] = (temp_bbox[:, 2]) / 2 #? x2 (xmax)
                # temp_bbox[:, 3] = (temp_bbox[:, 3]) / 2 #? y2 (ymax)
                mosaic_bboxes = temp_bbox.clone().detach()
            elif i == 1:  # top right
                mosaic_aug_img[:, :height, width:] = transposed_img
                temp_bbox[:, 0] += 0.5
                temp_bbox[:, 2] += 0.5
                # temp_bbox[:, 0] = (temp_bbox[:, 0] / 2) + 0.5
                # temp_bbox[:, 1] = (temp_bbox[:, 1] / 2)
                # temp_bbox[:, 2] = (temp_bbox[:, 2] / 2) + 0.5
                # temp_bbox[:, 3] = (temp_bbox[:, 3] / 2)
                mosaic_bboxes = torch.vstack((temp_bbox, mosaic_bboxes))
            elif i == 2:  # bottom left
                mosaic_aug_img[:, height:, :width] = transposed_img
                temp_bbox[:, 1] += 0.5
                temp_bbox[:, 3] += 0.5
                # temp_bbox[:, 0] = (temp_bbox[:, 0] / 2) 
                # temp_bbox[:, 1] = (temp_bbox[:, 1] / 2) + 0.5
                # temp_bbox[:, 2] = (temp_bbox[:, 2] / 2) 
                # temp_bbox[:, 3] = (temp_bbox[:, 3] / 2) + 0.5
                mosaic_bboxes = torch.vstack((temp_bbox, mosaic_bboxes))
            elif i == 3:  # bottom right
                mosaic_aug_img[:, height:, width:] = transposed_img
                temp_bbox += 0.5
                # temp_bbox[:, 0] = (temp_bbox[:, 0] / 2) + 0.5 
                # temp_bbox[:, 1] = (temp_bbox[:, 1] / 2) + 0.5
                # temp_bbox[:, 2] = (temp_bbox[:, 2] / 2) + 0.5
                # temp_bbox[:, 3] = (temp_bbox[:, 3] / 2) + 0.5
                mosaic_bboxes = torch.vstack((temp_bbox, mosaic_bboxes))
        
        #visualize_bboxes(np.clip(mosaic_aug_img.permute(1, 2, 0).numpy(), 0, 1).copy(), mosaic_bboxes, self.img_size)
        return mosaic_aug_img, mosaic_bboxes
        
    def load_mosaic(self, Current_mosaic_index:List[int], Diff_mosaic_index:List[int], original_id:int ):
        '''
            Current_mosaic_index : For constructing masaic about current classes
            Diff_mosaic_index : For constructing mosaic abhout differenct classes (Not Now classes)
            Current_bboxes : numpy array. [cls, cx, cy, w, h] for current classes
            Diff_bboxes : numpy array. [cls, cx, cy, w, h] for different classes (Not Now classes)
        '''
        # loads images in a mosaic
        Mosaic_size = self.img_size #1024, im_w, im_h : 1024
            
        Current_mosaic_img, Current_mosaic_labels = self.make_batch_mosaic(Current_mosaic_index, Mosaic_size, True, original_id)
        Diff_mosaic_img, Diff_mosaic_labels = self.make_batch_mosaic(Diff_mosaic_index, Mosaic_size, True, original_id)
        Current_mosaic_labels = self._make_resized_targets(Current_mosaic_labels)
        Diff_mosaic_labels = self._make_resized_targets(Diff_mosaic_labels)
        
        return Current_mosaic_img, Current_mosaic_labels, Diff_mosaic_img, Diff_mosaic_labels
    
    def _make_resized_targets(self, target: Dict)-> Dict:
        cxcy_boxes = box_xyxy_to_cxcywh(target[:, :-1])
        temp_dict = {}
        #boxes = target[:, :-1]
        labels = target[:, -1].to(dtype=torch.long)
        
        temp_dict['boxes'] = cxcy_boxes
        temp_dict['labels'] = labels
        temp_dict['images_id'] = torch.tensor(0)
        temp_dict['area'] = torch.tensor(0)
        temp_dict['iscrowd'] = torch.tensor(0)
        temp_dict['orig_size'] = torch.tensor(self.img_size)
        temp_dict['size'] = torch.tensor(self.img_size)
        
        return temp_dict
    
#For Rehearsal
def CombineDataset(args, RehearsalData, CurrentDataset, Worker, Batch_size, old_classes):
    OldDataset = CustomDataset(args, RehearsalData, old_classes) #oldDatset[idx]:
    class_ids = CurrentDataset.class_ids
    CombinedDataset = ConcatDataset([OldDataset, CurrentDataset]) #Old : previous, Current : Now
    MosaicBatchDataset = BatchMosaicAug(CombinedDataset, class_ids, CBB_transform("mosaic"), args.Mosaic) #* if Mosaic == True -> 1 batch(divided three batch/ False -> 3 batch (only original)
    
    print(MosaicBatchDataset[0])

    print(f"current Dataset length : {len(CurrentDataset)} -> Rehearsal + Current length : {len(MosaicBatchDataset)}")
    
    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(MosaicBatchDataset)
        else:
            sampler_train = samplers.DistributedSampler(MosaicBatchDataset)
    else:
        sampler_train = torch.utils.data.RandomSampler(MosaicBatchDataset)
        
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, Batch_size, drop_last=True)
    CombinedLoader = DataLoader(MosaicBatchDataset, batch_sampler=batch_sampler_train,
                        collate_fn=utils.collate_fn, num_workers=Worker,
                        pin_memory=True)
    
    
    return MosaicBatchDataset, CombinedLoader