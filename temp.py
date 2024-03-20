def contruct_rehearsal(losses_value: float, lower_limit: float, upper_limit: float, samples, targets,
                       origin_samples: torch.Tensor, origin_targets: Dict, rehearsal_classes: List = [], Current_Classes: List[int], Rehearsal_Memory: int = 300) \
                           -> Dict:
    # Check if losses_value is within the specified range
    if losses_value > lower_limit and losses_value < upper_limit : 
        ex_device = torch.device("cpu")
        
        for enum, target in enumerate(targets): #! 배치 개수 ex) 4개 
            # Get the unique labels and the count of each label
            label_tensor = target['labels']
            image_id = target['image_id'].item()
            label_tensor_unique = torch.unique(label_tensor)
            if set(label_tensor_unique.tolist()).issubset(Current_Classes) is False: #if unique tensor composed by Old Dataset, So then Continue iteration
                continue
            
            label_tensor_count = label_tensor.numpy()
            bin = np.bincount(label_tensor_count)
            if image_id in rehearsal_classes.keys():
                continue
            label_tensor_unique_list = label_tensor_unique.tolist()
            
            if check_rehearsal_size(Rehearsal_Memory, rehearsal_classes, *label_tensor_unique_list) == True:
                rehearsal_classes[image_id] = [losses_value, label_tensor_unique]
            else:
                print(f"**** Memory over ****")
                
            for index, unique_idx in enumerate(label_tensor_unique):
                unique_idx = unique_idx.item()
                rehearsal_size = len(list(filter(lambda x: x in list(rehearsal_classes.values())[1], unique_idx)))
                
                if rehearsal_size <= Rehearsal_Memory:  #TODO : Memory size 개수를 셀 방법을 찾아야한다. (label_tensor가 있어서 사용하면 될듯 ) dict.values()에 filter를 걸어서 계산
                    rehearsal_classes[image_id] = [losses_value, label_tensor_unique]
                    return rehearsal_classes
                else :
                    print(f"**** Memory over ****")
                    if change_rehearsal_size(Rehearsal_Memory, rehearsal_classes, )
                    if rehearsal_classes[image_id][0] > losses_value:
                        rehearsal_classes[image_id] = [losses_value, label_tensor_unique]
                    return rehearsal_classes
    
    return rehearsal_classes

def check_rehearsal_size(limit_memory_size, rehearsal_classes, *args, ): 
    check_list = [len(list(filter(lambda x: index in x, list(rehearsal_classes.values())[1]))) for index in len(args)]
    
    check = all([value < limit_memory_size for value in check_list])
    return check

def change_rehearsal_size(limit_memory_size, rehearsal_classes, *args, ): 
    check_list = [len(list(filter(lambda x: index in x, list(rehearsal_classes.values())[1]))) for index in len(args)]
    temp_array = np.array(check_list)
    temp_array = temp_array < limit_memory_size 
    
    non_over_list = []
    for t, arg in zip(temp_array, args):
        if t == False:
            non_over_list.appned(arg)
            
    check_list = list(filter(lambda x: all(item in x[1][1] for item in non_over_list), list(rehearsal_classes.items())))
    sorted_result  = sorted(check_list, key = lambda x : x[1][0])
    return sorted_result
    