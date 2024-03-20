import pickle
import copy
from ast import arg
from xmlrpc.client import Boolean
import torch
import util.misc as utils
import numpy as np
from typing import Tuple, Dict, List, Optional
import os
import torch.distributed as dist
import random
from util.misc import get_world_size
from termcolor import colored


from Custom_Dataset import *
from custom_prints import *
from engine import extra_epoch_for_replay
from custom_utils import buffer_checker

#TODO : Change calc each iamage loss and tracking each object loss avg.
def _replacment_strategy(args, loss_value, targeted, rehearsal_classes, label_tensor_unique_list, image_id, num_bounding_boxes):
    if args.Sampling_strategy == "hierarchical" or args.Sampling_strategy == "hier_highlabels": 
        if ( targeted[1][0] > loss_value ): #Low buffer construct
            print(colored(f"hierarchical based buffer change strategy", "blue"))
            del rehearsal_classes[targeted[0]]
            rehearsal_classes[image_id] = [loss_value, label_tensor_unique_list, num_bounding_boxes]
            return rehearsal_classes
        
    if args.Sampling_strategy == "hier_highloss" or args.Sampling_strategy == "highlabels_highloss" or args.Sampling_strategy == "hier_highunique_highloss" :
        if ( targeted[1][0] < loss_value ): # high loss buffer construct
            print(colored(f"hier_highloss based buffer change strategy", "blue"))
            del rehearsal_classes[targeted[0]]
            rehearsal_classes[image_id] = [loss_value, label_tensor_unique_list, num_bounding_boxes]
            return rehearsal_classes
        
    if args.Sampling_strategy == "low_loss" : 
        if ( targeted[1][0] > loss_value ): # high loss buffer construct
            print(colored(f"low_loss based buffer change strategy", "blue"))
            del rehearsal_classes[targeted[0]]
            rehearsal_classes[image_id] = [loss_value, label_tensor_unique_list, num_bounding_boxes]
            return rehearsal_classes
        
    if args.Sampling_strategy  == "RODEO": # This is same that "RODEO sampling strategy"
        if ( len(targeted[1][1]) < len(label_tensor_unique_list) ): #Low buffer construct
            print(colored(f"high-unique counts based buffer change strategy", "blue"))
            del rehearsal_classes[targeted[0]]
            rehearsal_classes[image_id] = [loss_value, label_tensor_unique_list, num_bounding_boxes]
            return rehearsal_classes
        
    random_prob = random.random() 
    if args.Sampling_strategy  == "random" :
        if random_prob > 0.5 :
            print(colored(f"random counts based buffer change strategy", "blue"))
            key_to_delete = random.choice(list(rehearsal_classes.keys()))
            del rehearsal_classes[key_to_delete]
            rehearsal_classes[image_id] = [loss_value, label_tensor_unique_list, num_bounding_boxes]
            return rehearsal_classes
    
    if args.Sampling_strategy == "hard":
        # This is same as "hard sampling"
        if targeted[1][2] < num_bounding_boxes:  # Low buffer construct
            print(colored(f"hard sampling based buffer change strategy", "blue"))
            del rehearsal_classes[targeted[0]]
            rehearsal_classes[image_id] = [loss_value, label_tensor_unique_list, num_bounding_boxes]
            return rehearsal_classes

    print(f"no changed")
    return rehearsal_classes

def _change_available_list_mode(mode, rehearsal_dict, need_to_include, least_image, current_classes):
    '''
        각 유니크 객체의 개수를 세어 제한하는 것은 그에 맞는 이미지가 존재해야만 모일수도 있기때문에 모두 모을 수 없을 수도 있게되는 불상사가 있다.
        따라서 객체의 개수를 제한하는 것보다는 그에 맞게 비율을 따져서 이미지를 제한하는 방법이 더 와닿을 수 있다.
        
        영어로 아래 설명을 작성한 부분은 Notion에서 제공됩니다. 
    '''
    if mode == "normal":
        '''
            CIL의 방법을 직접적으로 CIOD 모델에 가져오는 것이 모델 전체의 분포를 가져오는 것에 문제가 있다고 판단하였기 때문에 Replay를 
            사용해서 연구하던 사람들은 이러한 전체 데이터셋 내의 객체 비중을 따지지 않고 데이터를 수집하는 방법을 선택하였습니다.
            Random, Hard, Adaptive, RODEO(에서 제안된 다중 객체) 등에서는 하나의 이미지 내에 여러개의 객체가 존재한다는 OD의 특성을 반영하여
            랜덤으로 추출하거나, 객체의 개수가 많은 순으로 추출하거나, 독립적인 객체의 수가 많은 순으로 추출하거나 등의 동작을 진행하였습니다.
            이런 방법은 각 클래스 객체의 수집 량을 제한 하지 않는 방법으로 여기서는 'normal' 옵션으로 적어서 사용하였습니다.
            
            이러한 방법들은 제법 괜찮게 동작하는 것처럼 보였지만, 클래스의 일부분이 본 데이터셋 내에서 많은 객체를 가지고 있음에도 불구하고 적게 추출되거나
            거의 추출되지 않는 객체도 있을 수 있다는 가능성이 있었습니다. Replay는 이전의 Task의 일부분을 가지고 계속해서 상기시켜주는 동작을 하는데
            기존에 많은 양이 있었음에도 불구하고 적은 양의 데이터가 버퍼내에 추출되게 되면 이는 데이터의 분포를 적극적으로 버퍼에 반영할 수 없었습니다.
            이는 기존의 데이터 분포와는 동떨어지게 버퍼가 구성되거나, 이를 가지고 incremental task 상황에서 반복적으로 훈련을 진행할 때 적은 양의 데이터들은 
            지속적으로 성능이 떨어지는 문제도 보였습니다.
        '''
        # no limit and no least images
        changed_available_dict = rehearsal_dict
        
    if mode == "GM": # GM 모드
        '''
            'normal'과 'classification'에서의 버퍼 분할 방법에서 발생할 수 있는 여러 문제를 해결하기위해 우리는 GuaranteeMinimum 이라는 방법을 제안합니다.
            우선 우리는 최소한으로 보장할 이미지의 개수를 하이퍼 파라미터로 가집니다. 이는 특정 클래스에 해당하는 객체가 데이터 내에 하나라도 존재한다면(Unique) 해당 객체를 포함하는 데이터라고 가정합니다.
            중복되는 이미지로 인해서 버퍼의 용량을 최대한으로 사용하지 못하는 것을 방지하기 위해 우리는 이미지 전체의 개수를 제한하고 버퍼 관리자 내에서 Image ID를 관리합니다.
            버퍼 관리자는 Image_ID가 중복될 때에는 하나의 데이터(이미지)만 저장하도록 하고, 해당 데이터 내에 있는 모든 고유 객체 인덱스 개수를 버퍼 내의 각 클래스를 보장하기 위한 용도로 사용합니다. 
            
            이는 기존의 CIOD Replay에서 발생하던 버퍼를 최대한 활용하지 못하는 문제를 해결함과 동시에 각 고유 클래스를 포함하고 있는 최소한의 데이터를 보장할 수 있도록 데이터를 수집할 수 있습니다.
            정확한 동작 과정은
            1. 버퍼 용량이 초과하기 이전에는 모든 데이터를 수집합니다. (버퍼용량 : 1000 가정)
            2. 1000이 초과하고 나서는 버퍼 내의 각각 고유 클래스 객체가 버퍼 내에 포함되어 있는 양을 구합니다. ( class 1 : 100장, class 2: 200장, ...)
            3. 미리 설정하였던 최소한의 데이터 개수를 넘어선 고유 클래스 번호들을 따로 모으고 이들이 포함되어 있는 데이터는 교체 가능한 것으로 간주합니다. 이 때, 최대한 최소 데이터 개수를 넘지 못한
               데이터를 제거하는 일을 줄이기 위해서 최소 데이터 수를 초과하는 객체만 가지고 있는 데이터를 교체 대상으로 지정합니다.
               
            4. 이 때 교체 대상으로 지정된 여러 개의 데이터(버퍼 내의)들은 Hierarchical Sampling 전략을 통해서 최대 고유 개체를 반영하면서 동시에 현재 모델(optimized T_all)에 가장 인접한 데이터; 적은 loss의 데이터를
               우선적으로 버퍼로 가져올 수 있도록 합니다. 이는 GM mode와 다른 3.2 절에서 설명합니다.
               
            참고로 최소 용량을 초과하는 데이터에 대해서는 언제든 교체 목록의 대상이 될 수 있으며 최대한의 제한은 없기 때문에 원본 데이터셋의
            특정 고유 클래스가 많은 데이터(이미지)내에 분포해 있다면 버퍼에도 이와 같은 속성이 반영될 것입니다.
            이를 통해 GM mode는 데이터의 전체 비율을 반영할 수 있으며 모든 고유 클래스들의 최소 데이터의 양도 보장함으로써 replay 훈련시에
            점점 Imbalancing한 상황을 완화할 수 있습니다.
        '''
        image_counts_in_rehearsal = {class_label: sum(class_label in classes for _, (_, classes, _) in rehearsal_dict.items()) for class_label in current_classes}
        print(f"replay counts : {image_counts_in_rehearsal}")
        
        changed_available_dict = {key: (losses, classes, bboxes) for key, (losses, classes, bboxes) in rehearsal_dict.items() if all(image_counts_in_rehearsal.get(class_label, 0) > least_image for class_label in classes)}

        if len(changed_available_dict.keys()) == 0 :
            # this process is protected to generate error messages
            # include classes that have at least one class in need_to_include
            print(colored(f"no changed available dict, suggest to reset your least image count", "blue"))
            temp_dict = {key: len([c for c in items[1] if c in need_to_include]) for key, items in rehearsal_dict.items() if any(c in need_to_include for c in items[1])}

            # sort the temporary dictionary by values (counts of classes from need_to_include)
            sorted_temp_dict = dict(sorted(temp_dict.items(), key=lambda item: item[1]))
            
            # get the first key in the sorted dictionary as min_key
            try :
                min_key = next(iter(sorted_temp_dict))

                # create the new changed_available_dict with entries that have the minimum number of classes from need_to_include
                changed_available_dict = {key:items for key, items in rehearsal_dict.items() if len([c for c in items[1] if c in need_to_include]) == sorted_temp_dict[min_key]}
            except:
                print(colored(f"no changed available items, so now all rehearsal dictionary used to changing. recommend to you change the least_image value", "red", "on_yellow"))
                changed_available_dict = rehearsal_dict
    
    # TODO:  in CIL method, {K / |C|} usage
    # if mode == "classification":
    '''
        CIL의 방법은 모든 클래스의 데이터를 보장할 수는 있지만 OD의 특성상 중복되는 이미지가 버퍼내에 저장될 수 있다. 이는 동일한 데이터를 두번 
        사용하는 것으로써 버퍼를 최종적으로 구성하였을 때 최대한으로 버퍼를 구성했는지 확신할 수 없다.
        
        또한 특정 클래스의 객체들은 데이터셋 전체에서 극히 일부분만 차지하는 경향이 있을 수 있고 반대로 많은 양을 가지고 있을 수 있는데
        CIL 방법으로는 이러한 데이터셋 전체의 비중을 반영하지 못한채로 모든 클래스에 해당하는 데이터를 버퍼에 균등하게 가지게 된다.
        이는 real 환경과 비교해서 살펴보아도 모든 객체가 균등하게 존재하지 않는 다는 점에서 크게 어긋나기 때문에 균등한 분포로 버퍼를 구성하는것은
        이상적이지 않다.
    '''
    #     num_classes = len(classes)
    #     initial_limit = limit_image // num_classes
    #     limit_memory = {class_index: initial_limit for class_index in classes}
        
    return changed_available_dict


def construct_rehearsal(args, losses_dict: dict, targets, rehearsal_dict: List, 
                       current_classes: List[int], least_image: int = 3, limit_image:int = 100) -> Dict:

    loss_value = 0.0
    for enum, target in enumerate(targets): #! 배치 개수 ex) 4개 
        loss_value = losses_dict["loss_bbox"][enum] + losses_dict["loss_giou"][enum] + losses_dict["loss_labels"][enum]
        if loss_value > 10.0 :
            "너무 높은 loss를 가지는 객체들은 모일 필요없음."
            continue
        # Get the unique labels and the count of each label
        label_tensor = target['labels']
        bbox_counts = label_tensor.shape[0]
        image_id = target['image_id'].item()
        label_tensor_unique = torch.unique(label_tensor)
        label_tensor_unique_list = label_tensor_unique.tolist()

        if len(rehearsal_dict.keys()) <  limit_image :
            # when under the buffer 
            rehearsal_dict[image_id] = [loss_value, label_tensor_unique_list, bbox_counts]
        else :
            if args.Sampling_mode == "normal": # Hard, RODEO strategy is not using GM mode.
                    targeted = _calc_target(rehearsal_classes=rehearsal_dict, replace_strategy=args.Sampling_strategy, )
                    rehearsal_dict = _replacment_strategy(args=args, loss_value=loss_value, targeted=targeted, 
                                                            rehearsal_classes=rehearsal_dict, label_tensor_unique_list=label_tensor_unique_list,
                                                            image_id=image_id, num_bounding_boxes=bbox_counts)
                    
                
                
            if args.Sampling_mode == "GM":    
                # First, generate a dictionary with counts of each class label in rehearsal_classes
                image_counts_in_rehearsal = {class_label: sum(class_label in classes for _, classes, _ in rehearsal_dict.values()) for class_label in label_tensor_unique_list}

                # Then, calculate the needed count for each class label and filter out those with a non-positive needed count
                need_to_include = {class_label: count - least_image for class_label, count in image_counts_in_rehearsal.items() if count - least_image <= 0}

                if len(need_to_include) > 0:
                    changed_available_dict = _change_available_list_mode(mode=args.Sampling_mode, rehearsal_dict=rehearsal_dict,
                                                need_to_include=need_to_include, least_image=least_image, current_classes=current_classes)
                    
                    # all classes dont meet L requirement
                    targeted = _calc_target(rehearsal_classes=changed_available_dict, replace_strategy=args.Sampling_strategy, )
                    
                    del rehearsal_dict[targeted[0]]
                    rehearsal_dict[image_id] = [loss_value, label_tensor_unique_list, bbox_counts]
                else :
                    changed_available_dict = _change_available_list_mode(mode=args.Sampling_mode, rehearsal_dict=rehearsal_dict,
                                                need_to_include=need_to_include, least_image=least_image, current_classes=current_classes)
                    
                    # all classes dont meet L requirement
                    targeted = _calc_target(rehearsal_classes=changed_available_dict, replace_strategy=args.Sampling_strategy, )
                    rehearsal_dict = _replacment_strategy(args=args, loss_value=loss_value, targeted=targeted, 
                                            rehearsal_classes=rehearsal_dict, label_tensor_unique_list=label_tensor_unique_list,
                                            image_id=image_id, num_bounding_boxes=bbox_counts)
    
    return rehearsal_dict


def _check_rehearsal_size(limit_memory_size, rehearsal_classes, unique_classes_list, ):
    if len(rehearsal_classes.keys()) == 0:
        return True
    
    check_list = [len(list(filter(lambda x: index in x[1], list(rehearsal_classes.values())))) for index in unique_classes_list]
    
    check = all([value < limit_memory_size for value in check_list])
    return check


def _calc_target(rehearsal_classes, replace_strategy="hierarchical", ): 

    if replace_strategy == "hierarchical":
        # ours for effective, mode is "GuaranteeMinimum"
        min_class_length = min(len(x[1]) for x in rehearsal_classes.values())
        
        # first change condition: low unique based change
        changed_list = [(index, values) for index, values in rehearsal_classes.items() if len(values[1]) == min_class_length]
    
        # second change condition: low loss based change
        sorted_result = max(changed_list, key=lambda x: x[1][0])
        
    elif replace_strategy == "hier_highunique_highloss":
        # ours for effective, mode is "GuaranteeMinimum"
        min_class_length = min(len(x[1]) for x in rehearsal_classes.values())
        
        # first change condition: low unique based change
        changed_list = [(index, values) for index, values in rehearsal_classes.items() if len(values[1]) == min_class_length]
    
        # second change condition: high loss based change
        sorted_result = min(changed_list, key=lambda x: x[1][0])
        
    elif replace_strategy == "hier_highlabels":
        # ours for effective, mode is "GuaranteeMinimum"
        # x[2] = the number of bbox labels[int]
        min_class_length = min(x[2] for x in rehearsal_classes.values())
        
        # first change condition: low unique based change
        changed_list = [(index, values) for index, values in rehearsal_classes.items() if len(values[1]) == min_class_length]
    
        # second change condition: high loss based change
        sorted_result = max(changed_list, key=lambda x: x[1][0])
        
    elif replace_strategy == "highlabels_highloss":
        # ours for effective, mode is "GuaranteeMinimum"
        # x[2] = the number of bbox labels[int]
        min_class_length = min(x[2] for x in rehearsal_classes.values())
        
        # first change condition: target low labels
        changed_list = [(index, values) for index, values in rehearsal_classes.items() if len(values[1]) == min_class_length]
    
        # second change condition: target low loss sample
        sorted_result = min(changed_list, key=lambda x: x[1][0])
        
    elif replace_strategy == "RODEO": # RODEO == delete high unqiue classes
        # only high unique based change, mode is "normal" or "random"
        sorted_result = min(rehearsal_classes.items(), key=lambda x: len(x[1][1]))
        
    elif replace_strategy == "random":
        # only random change, mode is "normal" or "random"
        sorted_result = None
        
    elif replace_strategy == "low_loss":
        # only low loss based change, mode is "normal" or "random"
        sorted_result = max(rehearsal_classes.items(), key=lambda x: x[1][0])
        
    elif replace_strategy == "hard":
        # only high bounding box count based change, mode is "normal" or "random"
        sorted_result = min(rehearsal_classes.items(), key=lambda x: x[1][2])

    return sorted_result


def _save_rehearsal_for_combine(task, dir, rehearsal, epoch):
    backupdir = os.path.join(dir, "backup")
    #* save the capsulated dataset(Boolean, image_id:int)
    if not os.path.exists(dir) and utils.is_main_process():
        os.makedirs(dir, exist_ok=True)
        print(f"Directory created")

    if not os.path.exists(backupdir) and utils.is_main_process():
        os.makedirs(backupdir, exist_ok=True)
        print(f"Backup directory created")    
        
    if utils.get_world_size() > 1: dist.barrier()

    temp_dict = copy.deepcopy(rehearsal)
    for key, value in rehearsal.items():
        if len(value[1]) == 0:
            del temp_dict[key]
    
    try:
        dist_rank = dist.get_rank()
    except:
        dist_rank = 0
        
    backup_dir = os.path.join(
        dir + "/backup/", str(dist_rank) + "_gpu_rehearsal_task_" + str(task) + "_ep_" + str(epoch)
    )
    dir = os.path.join(
        dir, str(dist_rank) + "_gpu_rehearsal_task_" + str(task) + "_ep_" + str(epoch)
    )
    with open(dir, 'wb') as f:
        pickle.dump(temp_dict, f)
        
    with open(backup_dir, 'wb') as f:
        pickle.dump(temp_dict, f)


import pickle
import os
def _save_rehearsal(rehearsal, dir, task, memory):
    all_dir = os.path.join(dir, "Buffer_T_" + str(task) +"_" + str(memory))
    if not os.path.exists(dir):
        os.mkdir(dir)
        print(f"Directroy created")

    with open(all_dir, 'wb') as f:
        pickle.dump(rehearsal, f)
        print(colored(f"Save task buffer", "light_red", "on_yellow"))


def load_rehearsal(dir, task=None, memory=None):
    if dir is None:
        return None
    
    if task==None and memory==None:
        all_dir = dir
    else:
        all_dir = os.path.join(dir, "Buffer_T_" + str(task) + "_" + str(memory))
    print(f"load replay file name : {all_dir}")
    if os.path.exists(all_dir) :
        with open(all_dir, 'rb') as f :
            temp = pickle.load(f)
            print(colored(f"********** Loading {task} tasks' buffer ***********", "blue", "on_yellow"))
            return temp
    else:
        print(colored(f"not exist file. plz check your replay file path or existence", "blue", "on_yellow"))


def _handle_rehearsal(args, dir, limit_memory_size, gpu_counts, task, epoch, least_image, list_CC, include_all=False):

    def load_dicts_from_files(dir_list):
        merged_dict = {}
        for dictionary_dir in dir_list:
            with open(dictionary_dir, 'rb') as f :
                temp = pickle.load(f)
                merged_dict = {**merged_dict, **temp}
        return merged_dict

    # TODO: in incremental setting, class overlapping is exist?
    def icarl_load_dicts_from_files(dir_list):
        merged_dict = {}
        for dictionary_dir in dir_list:
            with open(dictionary_dir, 'rb') as f :
                temp = pickle.load(f)
                merged_dict = {**merged_dict, **temp}
        return merged_dict

    dir_list = [
        os.path.join(
            dir,
            str(num) +"_gpu_rehearsal_task_" + str(task) + "_ep_" + str(epoch)
        ) for num in range(gpu_counts)
    ]

    for each_dir in dir_list:
        if not os.path.exists(each_dir):
            raise Exception("No rehearsal file")   
    
    print(colored(f"Total memory : {len(dir_list)} ", "blue"))
    # For only one GPU processing, becuase effective buffer constructing
    print(colored(f"New buffer dictionary genrating for optimizing replay dataset", "dark_grey", "on_yellow"))
    new_buffer_dict = {}

    if args.Sampling_strategy != 'icarl':
        merge_dict = load_dicts_from_files(dir_list)
        
        if args.Sampling_strategy == "random" :
            keys = random.sample(list(merge_dict.keys()), limit_memory_size-1)
            new_buffer_dict = {key: merge_dict[key] for key in keys}
            return new_buffer_dict
        
        for img_idx in merge_dict.keys():
            loss_value = merge_dict[img_idx][0]
            unique_classes_list = merge_dict[img_idx][1]
            bbox_counts = merge_dict[img_idx][2]
                                                    # 0 -> loss value
                                                    # 1 -> unique classes list

            if len(new_buffer_dict.keys()) <  limit_memory_size :                                        
                new_buffer_dict[img_idx] = merge_dict[img_idx]
            else : 
                if args.Sampling_mode == "normal":
                    targeted = _calc_target(rehearsal_classes=new_buffer_dict, replace_strategy=args.Sampling_strategy, )
                    new_buffer_dict = _replacment_strategy(args=args, loss_value=loss_value, targeted=targeted, 
                                        rehearsal_classes=new_buffer_dict, label_tensor_unique_list=unique_classes_list,
                                        image_id=img_idx, num_bounding_boxes=bbox_counts)
                    
                elif args.Sampling_mode == "GM":    
                    # First, generate a dictionary with counts of each class label in rehearsal_classes
                    image_counts_in_rehearsal = {class_label: sum(class_label in classes for _, classes, _ in new_buffer_dict.values()) for class_label in unique_classes_list}

                    # Then, calculate the needed count for each class label and filter out those with a non-positive needed count
                    need_to_include = {class_label: count - least_image for class_label, count in image_counts_in_rehearsal.items() if (count - least_image) <= 0}
                    if len(need_to_include) > 0:
                        changed_available_dict = _change_available_list_mode(mode=args.Sampling_mode, rehearsal_dict=new_buffer_dict,
                                                    need_to_include=need_to_include, least_image=least_image, current_classes=list_CC)
                        
                        # all classes dont meet L requirement
                        targeted = _calc_target(rehearsal_classes=changed_available_dict, replace_strategy=args.Sampling_strategy, )
                        
                        del new_buffer_dict[targeted[0]]
                        new_buffer_dict[img_idx] = [loss_value, unique_classes_list, bbox_counts]
                            
                    else :
                        changed_available_dict = _change_available_list_mode(mode=args.Sampling_mode, rehearsal_dict=new_buffer_dict,
                                                    need_to_include=need_to_include, least_image=least_image, current_classes=list_CC)
                    
                        # all classes meet L requirement
                        # Just sampling strategy and replace strategy
                        targeted = _calc_target(rehearsal_classes=changed_available_dict, replace_strategy=args.Sampling_strategy,)

                        new_buffer_dict = _replacment_strategy(args=args, loss_value=loss_value, targeted=targeted, 
                                                                rehearsal_classes=new_buffer_dict, label_tensor_unique_list=unique_classes_list,
                                                                image_id=img_idx, num_bounding_boxes=bbox_counts)

    else:
        merged_dict = icarl_load_dicts_from_files(dir_list)
        # for cls, val in merge_dict.items():
        #     mean_feat = val[0]
        #     img_ids = val[1] # with difference

        #     if len(img_ids) <= limit_memory_size:
        #         new_buffer_dict[cls] = val
        #     else:
        new_buffer_dict = merged_dict
            
    print(colored(f"Complete generating new buffer", "dark_grey", "on_yellow"))
    return new_buffer_dict


def _multigpu_rehearsal(args, dir, limit_memory_size, gpu_counts, task, epoch, least_image, list_CC):
    return _handle_rehearsal(args, dir, limit_memory_size, gpu_counts, task, epoch, least_image, list_CC, include_all=False)


def _merge_replay_for_multigpu(args, dir, limit_memory_size, gpu_counts, task, epoch, least_image, list_CC):
    return _handle_rehearsal(args, dir, limit_memory_size, gpu_counts, task, epoch, least_image, list_CC, include_all=True)
    

def merge_rehearsal_process(args, task:int ,dir:str ,rehearsal:dict ,epoch:int 
                                 ,limit_memory_size:int , list_CC:list, gpu_counts:int, ) -> dict:
    least_image = args.least_image
    # total_size = limit_memory_size * get_world_size()
    all_dir = os.path.join(dir, "Buffer_T_" + str(task) +"_" + str(limit_memory_size))
    
    #file save of each GPUs
    _save_rehearsal_for_combine(task, dir, rehearsal, epoch)
    
    # All GPUs ready replay buffer combining work(protecting some errors)
    if utils.get_world_size() > 1: dist.barrier()
        
    if utils.is_main_process() : 
        rehearsal_classes = _multigpu_rehearsal(args, dir, limit_memory_size, gpu_counts, task, epoch, least_image, list_CC)
        # save combined replay buffer data for next training
        # _save_rehearsal output : save total buffer dataset to dir
                    
        _save_rehearsal(rehearsal_classes, dir, task, limit_memory_size) 
        buffer_checker(args, task, rehearsal=rehearsal_classes)
    
    # wait main process to synchronization
    if utils.get_world_size() > 1: dist.barrier()

    # All GPUs ready replay dataset
    rehearsal_classes = load_rehearsal(all_dir)
    return rehearsal_classes


def construct_replay_extra_epoch(args, Divided_Classes, model, criterion, device, rehearsal_classes={}, task_num=0):
    
    # 0. Initialization
    extra_epoch = True
    print(f"already buffer state number : {len(rehearsal_classes)}")
    
    # 0.1. If you are not use the construct replay method, so then you use the real task number of training step.
    if args.Construct_Replay :
        task_num = args.start_task    
    
    # 1. 현재 테스크에 맞는 적절한 데이터 셋 호출 (학습한 테스크, 0번 테스크에 해당하는 내용을 가져와야 함)
    #    하나의 GPU로 Buffer 구성하기 위해서(더 정확함) 모든 데이터 호출
    # list_CC : collectable class index
    dataset_train, data_loader_train, _, list_CC = Incre_Dataset(task_num, args, Divided_Classes, extra_epoch) 
    
    # 2. Extra epoch, 모든 이미지들의 Loss를 측정
    rehearsal_classes = extra_epoch_for_replay(args, dataset_name="", data_loader=data_loader_train, model=model, criterion=criterion, 
                                                device=device, current_classes=list_CC, rehearsal_classes=rehearsal_classes)

    # 3. 수집된 Buffer를 특정 파일에 저장
    if args.Rehearsal_file is None:
        args.Rehearsal_file = args.output_dir
    # Rehearsal_file 경로의 폴더가 없을 경우 생성
    os.makedirs(os.path.dirname(args.Rehearsal_file), exist_ok=True)
    rehearsal_classes = merge_rehearsal_process(args=args, task=task_num, dir=args.Rehearsal_file, rehearsal=rehearsal_classes,
                                                    epoch=0, limit_memory_size=args.limit_image, gpu_counts=utils.get_world_size(), list_CC=list_CC)
    
    print(colored(f"Complete constructing buffer","red", "on_yellow"))
    
    return rehearsal_classes
