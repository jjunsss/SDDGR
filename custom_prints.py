from typing import Tuple, Dict, List, Optional
import os
import sys
import torch
from datetime import datetime

def write_to_addfile(filename):
    def decorator(func):
        def wrapper(*args, **kwargs):
            original_stdout = sys.stdout
            if os.path.exists("./" + filename.split('/')[1]) is False:
                os.makedirs(filename.split('/')[1],)                
            with open(filename, "a") as f:
                sys.stdout = f
                func(*args, **kwargs)
                sys.stdout = original_stdout
        return wrapper
    return decorator

from collections import Counter
@write_to_addfile("./check/check_replay_limited.txt")
def check_components(args, task, rehearsal_classes: Dict, print_stat: bool=False):
    '''
        1. check each instance usage capacity
        2. print each classes counts
        3. Instance Summary 
        4. Save information
    '''
    if len(rehearsal_classes) == 0:
        raise Exception("No replay classes")
    
    if print_stat == True:
        # check each instance usage capacity
        
        # To print the current time
        if args.Sampling_strategy == 'icarl':
            class_counts = {cls: len(item[1]) for cls, item in rehearsal_classes.items()}
            print(f"--------------------------------------------------------\n")
            print("Current Time =", datetime.now(), "Task = ", task)
            print(f"output file : {args.output_dir}")
            print(f"The number of buffer: {len(rehearsal_classes.keys())}")
            for key in sorted(class_counts):
                print(f"{key}: {class_counts[key]}")  
        else:
            class_counts = Counter(cls for _ ,(_, classes, _) in rehearsal_classes.items() for cls in classes)
            # To print the current time
            print(f"--------------------------------------------------------\n")
            print("Current Time =", datetime.now(), "Task = ", task)
            print(f"output file : {args.output_dir}")
            print(f"The number of buffer: {len(rehearsal_classes.keys())}")
            for key in sorted(class_counts):
                print(f"{key}: {class_counts[key]}")         
            
def Memory_checker():
    '''
        To check memory capacity
        To check memory cache capacity
    '''
    print(f"*" * 50)
    print(f"allocated Memory : {torch.cuda.memory_allocated()}")
    print(f"max allocated Memory : {torch.cuda.max_memory_allocated()}")
    print(f"*" * 50)
    print(f"cache allocated Memory : {torch.cuda.memory_allocated()}")
    print(f"max allocated Memory : {torch.cuda.max_memory_cached()}")
    print(f"*" * 50)
    
def over_label_checker(check_list:List , check_list2:List = None, check_list3:List = None, check_list4:List = None):
    if check_list2 is None:
        print("Only one overlist: ", check_list)    
    else :
        print("overlist: ", check_list, check_list2, check_list3, check_list4)

@write_to_addfile("./check/loss_check.txt")
def check_losses(epoch, index, losses, epoch_loss, count, training_class, rehearsal=None, dtype=None):
    '''
        protect to division zero Error.
        print (epoch, losses, losses of epoch, training count, training classes now, rehearsal check, CBB format check)
    '''

    try :
        epoch_total_loss = epoch_loss / count
    except ZeroDivisionError:
        epoch_total_loss = 0
            
    if index % 30 == 0: 
        print(f"epoch : {epoch}, losses : {losses:05f}, epoch_total_loss : {epoch_total_loss:05f}, count : {count}")
        if rehearsal is not None:
            print(f"total examplar counts : {len(list(rehearsal.keys()))}")
        if dtype is not None:
            print(f"Now, CBB is {dtype}")    
        
    if index % 30 == 0:
        print(f"current classes is {training_class}")