import os
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from typing import Optional, List
    
import copy
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from termcolor import colored

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
    elif 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = ntasks
        args.rank = proc_id
        args.gpu = proc_id % num_gpus
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
    
    
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
def is_main_process():
    return get_rank() == 0

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ['LOCAL_RANK'])

def get_local_size():
    if not is_dist_avail_and_initialized():
        return 1
    return int(os.environ['LOCAL_SIZE'])

def cleanup():
    dist.destroy_process_group()

def save_each_file(base_path, file_name, data):
    #* save the capsulated dataset(Boolean, image_id:int)
    if not os.path.exists(base_path) and is_main_process():
        os.makedirs(base_path, exist_ok=True)
        print(f"Directory created")
    # sync
    if get_world_size() > 1:
        dist.barrier()

    try:
        dist_rank = dist.get_rank()
    except:
        dist_rank = 0
        
    dir = os.path.join(base_path, str(dist_rank))
    save_file_path = get_unique_filename(dir, file_name, ".json")
    static_dir = dir + "_" + save_file_path
    
    with open(static_dir, 'w') as f:
        json_str = json.dumps(data, indent=4)
        f.write(json_str)
    print(colored(f"Save {dist_rank} gpu's {file_name} save complete", "red", "on_yellow"))    
    # sync
    if get_world_size() > 1:
        dist.barrier()
        
def get_unique_filename(base_path, filename, extension):
    counter = 1
    new_filename = f"{filename}{extension}"
    while os.path.exists(base_path + "_" + new_filename):
        new_filename = f"{filename}({counter}){extension}"
        counter += 1
    return new_filename

def load_meta_files(load_path):
    
    with open(load_path, 'r') as f:
        load_file = json.load(f)
        
    return load_file 
    
def merge_meta_files(file_paths, output_path):
    '''
        merge json files and save to file
    '''
    merged_list = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist. Skipping.")
            continue
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                merged_list.extend(data)
            else:
                print(f"File {file_path} does not contain a list. Skipping.")
    
    with open(output_path, 'w') as f:
        json.dump(merged_list, f, indent=4)
        print(colored(f"Save integration meta information ver. save here: {output_path}", "red", "on_yellow"))
        
    return merged_list
        
def merge_coco_like_jsons(file_paths, output_path):
    '''
        load, integrate, and save total file
    '''
    # 첫 번째 파일에서 키를 읽어서 병합된 결과를 저장할 딕셔너리 초기화
    if not os.path.exists(file_paths[0]):
        print(f"File {file_paths[0]} does not exist. Exiting.")
        return
    
    with open(file_paths[0], 'r') as f:
        data = json.load(f)
    
    # key initialization
    merged_dict = {key: [] for key in data.keys()}
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist. Skipping.")
            continue
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            
            if not all(key in data for key in merged_dict.keys()):
                print(f"File {file_path} does not contain all required keys. Skipping.")
                continue
            
            for key in merged_dict.keys():
                if isinstance(data[key], list):
                    merged_dict[key].extend(data[key])
                else:
                    print(f"Value for {key} in file {file_path} is not a list. Skipping.")
    
    with open(output_path, 'w') as f:
        json.dump(merged_dict, f, indent=4)
        print(colored(f"Save integration coco train information version save here: {output_path}", "light_red", "on_yellow"))    

from PIL import Image, ImageDraw
import numpy as np
def save_image(img_name, sample, output_folder):
    # if gen_id != 0:
    #     img_name = str(gen_id) + img_name[1:] + ".jpg"
    # else:
    img_name = f"{img_name}.jpg"
    
    sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
    sample = sample.cpu().numpy().transpose(1,2,0) * 255 
    sample = Image.fromarray(sample.astype(np.uint8))
    
    sample.save(os.path.join(output_folder, img_name))