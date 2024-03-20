from xmlrpc.client import Boolean
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from torch.utils.data import DataLoader, ConcatDataset
import datasets.samplers as samplers
import torch
import numpy as np
from termcolor import colored


def Incre_Dataset(task_num, args, incremental_classes, pseudo_dataset = False):
    """Create dataset and data loader for a given task, either for training or evaluation."""
    
    current_classes = incremental_classes[task_num]
    all_classes = sum(incremental_classes[:task_num + 1], [])
    previous_all_classes = sum(incremental_classes[:task_num], [])
    is_eval_mode = args.eval
    is_distributed = args.distributed
    
    #* For Training
    if not is_eval_mode:
        if pseudo_dataset :
            train_dataset = build_dataset(image_set='train', args=args, class_ids=previous_all_classes, pseudo=True)
            return train_dataset, None, None, previous_all_classes
        else :
            train_dataset = build_dataset(image_set='train', args=args, class_ids=current_classes)
            
        print(f"Current classes for training: {current_classes}")
        train_sampler = samplers.DistributedSampler(train_dataset, shuffle=True) if is_distributed else torch.utils.data.RandomSampler(train_dataset)
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True)
        train_loader = DataLoader(
            train_dataset, batch_sampler=train_batch_sampler,
            collate_fn=utils.collate_fn, num_workers=args.num_workers,
            pin_memory=True, prefetch_factor=args.prefetch)
        
        return train_dataset, train_loader, train_sampler, current_classes
    
    #* For Evaluation
    else:
        target_classes = all_classes
        print(colored(f"Current classes for evaluation: {target_classes}", "blue", "on_yellow"))
        
        val_dataset = build_dataset(image_set='val', args=args, class_ids=target_classes)
        
        val_sampler = samplers.DistributedSampler(val_dataset) if is_distributed else torch.utils.data.SequentialSampler(val_dataset)
        
        val_loader = DataLoader(
            val_dataset, args.batch_size, sampler=val_sampler,
            drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
            pin_memory=True, prefetch_factor=args.prefetch)
        
        return val_dataset, val_loader, val_sampler, all_classes

def make_class(test_file):
    #####################################
    ########## !! Edit here !! ##########
    #####################################
    class_dict = {
        'file_name': ['did', 'pz', 've'],
        'class_idx': [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], # DID
            [28, 32, 35, 41, 56], # photozone
            [24, 29, 30, 39, 40, 42] # 야채칸 중 일부(mAP 높은 일부)
        ]
    }
    #####################################
    
    # case_1) file name에 VE가 포함되어 있지 않은 경우
    if test_file.lower() in ['2021', 'multisingle', '10test']:
        test_file = 've' + test_file
    # case_2) 혼합 데이터셋
    if '+' in test_file:
        task_list = test_file.split('+')
        tmp = []
        for task in task_list:
            idx = [name in task.lower() for name in class_dict['file_name']].index(True)
            tmp.append(class_dict['class_idx'][idx])
        res = sum(tmp, [])
        return res  # early return
    
    idx = [name in test_file.lower() for name in class_dict['file_name']].index(True)
    return class_dict['class_idx'][idx]


def create_dataset_for_incremental(args, eval_config=False):

    gen_dataset = data_setting(ratio=args.divide_ratio, random_setting=False)

    # if eval_config :
    #     #! can set the testing setting
    #     classes = [idx+1 for idx in range(args.Test_Classes)]
    #     Divided_Classes = [classes]
        
    return gen_dataset

    
        
    return Divided_Classes

from collections import defaultdict
import numpy as np


def img_id_config_no_circular_training(args, re_dict):
    if args.Sampling_strategy == 'icarl':
        keys = []
        for cls, val in re_dict.items():
            img_ids = np.array(val[1])
            keys.extend(list(img_ids[:, 0].astype(int)))
            
        no_duple_keys = list(set(keys))
        print(f"not duple keys :{len(no_duple_keys)}")
        return no_duple_keys
    else:
        return list(re_dict.keys())

import copy
from sklearn.preprocessing import QuantileTransformer
import numpy as np
class CustomDataset(torch.utils.data.Dataset):
    '''
        replay buffer configuration
        1. Weight based Circular Experience Replay (WCER)
        2. Fisher based Circular Experience Replay (FCER)
        3. Fisher based ER
    '''
    def __init__(self, args, re_dict, old_classes):
        self.re_dict = copy.deepcopy(re_dict)
        self.old_classes = old_classes
        
        if args.CER == "uniform" and args.AugReplay:
            self.weights = None
            self.keys = list(self.re_dict.keys())
            self.datasets = build_dataset(image_set='train', args=args, class_ids=self.old_classes, img_ids=self.keys)
            self.fisher_softmax_weights = None
            
        else :
            self.weights = None
            self.fisher_softmax_weights = None
            self.keys = img_id_config_no_circular_training(args, re_dict)
            self.datasets = build_dataset(image_set='train', args=args, class_ids=self.old_classes, img_ids=self.keys)
            

    def __len__(self):
        return len(self.datasets)
    
    def __repr__(self):
        print(f"Data key presented in buffer : {self.old_classes}")    

    def __getitem__(self, idx):
        samples, targets = self.datasets[idx]

        return samples, targets

import os
from glob import glob
from PIL import Image
from datasets.coco import make_coco_transforms
import json
from pycocotools.coco import COCO
class PseudoDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, args, gen_json_dir= None, pseudo_path=None, existing_ids=None, regen=False):
        """
        folder_path: 이미지들이 있는 폴더의 경로
        transform: 이미지에 적용할 변환 (예: torchvision.transforms)
        """
        self.folder_path = folder_path
        self.image_paths = glob(os.path.join(self.folder_path, '*.jpg'))
        self.image_ids = list(map(lambda x: os.path.basename(x), self.image_paths)) #* list
        if existing_ids is not None:
            self.image_ids = [img_id for img_id in self.image_ids if img_id not in existing_ids]
        self.transform = make_coco_transforms("val")
        self.generate_data = self.coco_loader(os.path.join(args.coco_path, "annotations/instances_val2017.json")) #* coco format
        if regen :
            with open(pseudo_path, 'r') as f:
                self.indicated_data = json.load(f)
        
        if gen_json_dir is not  None and pseudo_path is not None:
            self.original_data = COCO(gen_json_dir) #* Generated images format
            self.pseudo_data = COCO(pseudo_path) #* GLIGEN format
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.folder_path, image_id)
        image = Image.open(image_path).convert('RGB')  # 이미지를 RGB로 변환
        
        if self.transform:
            image, _ = self.transform(image)
        return image, image_id  # 타겟이 없으므로 이미지만 반환합니다.

    def coco_loader(self, coco_dir):
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

import copy
class ExtraDataset(torch.utils.data.Dataset):
    '''
        replay buffer configuration
        1. Weight based Circular Experience Replay (WCER)
        2. Fisher based Circular Experience Replay (FCER)
        3. Fisher based ER
    '''
    def __init__(self, args, re_dict, old_classes):
        self.re_dict = copy.deepcopy(re_dict)
        self.old_classes = old_classes
        self.keys = list(self.re_dict.keys())
        self.datasets = build_dataset(image_set='extra', args=args, class_ids=self.old_classes, img_ids=self.keys)
            
    def __len__(self):
        return len(self.datasets)
    
    def __repr__(self):
        print(f"Data key presented in buffer : {self.old_classes}")    

    def __getitem__(self, idx):
        samples, targets, new_samples, new_targets = self.datasets[idx]

        return samples, targets, new_samples, new_targets

import random
import collections

import torch.distributed as dist
class NewDatasetSet(torch.utils.data.Dataset):
    def __init__(self, args, datasets, OldDataset, AugReplay=False, Mosaic=False):
        self.args = args
        self.Datasets = datasets #now task
        self.Rehearsal_dataset = OldDataset
        self.AugReplay = AugReplay
        if self.AugReplay == True:
            self.old_length = len(self.Rehearsal_dataset) if dist.get_world_size() == 1 else int(len(self.Rehearsal_dataset) // dist.get_world_size()) # 4
            
    def __len__(self):
        return len(self.Datasets)

    def __getitem__(self, index): 
        img, target = self.Datasets[index] #No normalize pixel, Normed Targets
        if self.AugReplay == True :
            if self.args.CER == "uniform": # weight CER
                index = np.random.choice(np.arange(len(self.Rehearsal_dataset)))
                O_img, O_target, _, _ = self.Rehearsal_dataset[index] #No shuffle because weight sorting.
                return img, target, O_img, O_target
            
        return img, target

#For Rehearsal
from Custom_augmentation import CCB
def CombineDataset(args, OldData, CurrentDataset, 
                   Worker, Batch_size, old_classes, pseudo_training=False):
    '''
        MixReplay arguments is only used in MixReplay. If It is not args.MixReplay, So
        you can ignore this option.
    '''
    if pseudo_training is False: #* just original rehearsal
        OldDataset = CustomDataset(args, OldData, old_classes) #oldDatset[idx]:
        
        if args.AugReplay and not args.MixReplay :
            NewTaskdataset = NewDatasetSet(args, CurrentDataset, OldDataset, AugReplay=True)
                
        if args.Replay and not args.AugReplay and not args.MixReplay and not args.Mosaic:
            CombinedDataset = ConcatDataset([OldDataset, CurrentDataset])
            NewTaskdataset = NewDatasetSet(args, CombinedDataset, OldDataset, AugReplay=False)
        
    else : #* pseudo generation dataset, shuffle new dataset + gen dataset for training
        NewTaskdataset = ConcatDataset([OldData, CurrentDataset])
        
    print(colored(f"current Dataset length : {len(CurrentDataset)}", "blue"))
    print(colored(f"Total Dataset length : {len(CurrentDataset)} +  old dataset length : {len(OldData)}", "blue"))
    print(colored(f"********** sucess combined Dataset ***********", "blue"))
    
    if args.distributed:
        sampler_train = samplers.DistributedSampler(NewTaskdataset, shuffle=True)
    else:
        sampler_train = torch.utils.data.RandomSampler(NewTaskdataset)
        
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, Batch_size, drop_last=True)
    CombinedLoader = DataLoader(NewTaskdataset, batch_sampler=batch_sampler_train,
                    collate_fn=utils.collate_fn, num_workers=Worker,
                    pin_memory=True, prefetch_factor=args.prefetch) #worker_init_fn=worker_init_fn, persistent_workers=args.AugReplay)

    return NewTaskdataset, CombinedLoader, sampler_train


    
def IcarlDataset(args, single_class:int):
    '''
        For initiating prototype-mean of the feature of corresponding, single class-, dataset composed to single class is needed.
    '''
    dataset = build_dataset(image_set='extra', args=args, class_ids=[single_class])
    if len(dataset) == 0:
        return None, None, None
    
    if args.distributed:
        if args.cache_mode:
            sampler = samplers.NodeDistributedSampler(dataset)
        else:
            sampler = samplers.DistributedSampler(dataset)  
    else:
        sampler = torch.utils.data.RandomSampler(dataset)
        
    batch_sampler = torch.utils.data.BatchSampler(sampler, args.batch_size, drop_last=True)
    data_loader = DataLoader(dataset, batch_sampler=batch_sampler,
                                collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                pin_memory=True)
    
    return dataset, data_loader, sampler


def fisher_dataset_loader(args, RehearsalData, old_classes):
    print(colored(f"fisher loading classes : {old_classes}", "blue", "on_yellow"))
    buffer_dataset = ExtraDataset(args, RehearsalData, old_classes)
    
    sampler_train = torch.utils.data.SequentialSampler(buffer_dataset)
        
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, batch_size=1, drop_last=False)
    
    data_loader = DataLoader(buffer_dataset, batch_sampler=batch_sampler_train,
                                collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                pin_memory=True, prefetch_factor=args.prefetch)
    
    return data_loader

def _divide_classes_randomly(total_classes, ratios):
    random.shuffle(total_classes)
    divided_classes = []
    start_idx = 0

    for ratio in ratios:
        end_idx = start_idx + ratio
        divided_classes.append(total_classes[start_idx:end_idx])
        start_idx = end_idx

    return divided_classes

def data_setting(ratio: str, random_setting: bool=False):
    def flatten_list(nested_list):
        return [item for sublist in nested_list for item in sublist]
    
    total_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, \
                    11, 13, 14, 15, 16, 17, 18, 19, 20, \
                    21, 22, 23, 24, 25, 27, 28, 31, 32, \
                    33, 34, 35, 36, 37, 38, 39, 40, 41, \
                    42, 43, 44, 46, 47, 48, 49, 50, 51, \
                    52, 53, 54, 55, 56, 57, 58, 59, 60, \
                    61, 62, 63, 64, 65, 67, 70, 72, 73, \
                    74, 75, 76, 77, 78, 79, 80, 81, 82, \
                    84, 85, 86, 87, 88, 89, 90]
    
    # Initialize Divided_Classes
    Divided_Classes = [
        list(range(1, 46)),  # 45 classes
        list(range(46, 56)), # 10 classes
        list(range(56, 66)), # 10 classes
        list(range(66, 80)), # 14 classes
        list(range(80, 91))  # 11 classes
    ]
    
    ratio_to_classes = {
            '4040': [Divided_Classes[0], flatten_list(Divided_Classes[1:])],
            '402020': [Divided_Classes[0], flatten_list(Divided_Classes[1:3]), flatten_list(Divided_Classes[3:])],
            '4010101010': Divided_Classes,
            '7010': [flatten_list(Divided_Classes[:-1]), Divided_Classes[-1]],
            '8000': [flatten_list(Divided_Classes), []],
            '1010': [list(range(1, 11)), list(range(11, 22))],
            '20': [list(range(1, 22))]
    }
    Divided_Classes_detail = ratio_to_classes.get(ratio, total_classes)

    #* for various order testing in CL-DETR 
    if random_setting :
        # 나눌 비율: 40/10/10/10/10
        ratios = [40, 10, 10, 10, 10]

        # 랜덤으로 카테고리 ID를 섞고 나눈다.
        Divided_Classes_detail = _divide_classes_randomly(total_classes, ratios)
    
    print(colored(f"Divided_Classes :{Divided_Classes_detail}", "blue", "on_yellow"))
    return Divided_Classes_detail