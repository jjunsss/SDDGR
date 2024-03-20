import random
from pathlib import Path
import os
import numpy as np
import torch
import util.misc as utils

import re

from Custom_Dataset import *
from custom_utils import *
from custom_prints import *
from custom_buffer_manager import *

from datasets import get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, pseudo_process
from models import get_models
from glob import glob
import torch.backends.cudnn as cudnn
import wandb
import json
def init(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)
    
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    
    if not args.eval and utils.is_main_process() and args.wandb:
        wandb.login()
        experiment_name = f"{args.run_name}_{args.model_name}_{args.divide_ratio}_batch={args.batch_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run = wandb.init(project=args.prj_name, name=experiment_name, config=vars(args))


class TrainingPipeline:
    def __init__(self, args):
        init(args)
        self.set_directory(args)
        self.args = args
        self.device = torch.device(args.device)
        self.Divided_Classes, self.dataset_name, self.start_epoch, self.start_task, self.tasks = self._incremental_setting()
        if self.args.eval:
            self.args.start_task = 0
        self.model, self.model_without_ddp, self.criterion, self.postprocessors, self.teacher_model = self._build_and_setup_model(task_idx=self.args.start_task)
        if self.args.Branch_Incremental and not args.eval and args.pretrained_model is not None:
            self.make_branch(self.start_task, self.args, is_init=True)
        self.optimizer, self.lr_scheduler = self._setup_optimizer_and_scheduler()
        self.output_dir = Path(args.output_dir)
        self.load_replay, self.rehearsal_classes = self._load_replay_buffer()
        self.DIR = os.path.join(self.output_dir, 'mAP_TEST.txt')
        self.Task_Epochs = args.Task_Epochs
    
    def set_directory(self, args):
        '''
            pretrained_model and rehearsal file should be contrained "checkpoints" and "replay", respectively.
        '''        
        if args.pretrained_model_dir is not None:
            if 'checkpoints' not in args.pretrained_model_dir and not args.eval :
                args.pretrained_model_dir = os.path.join(args.pretrained_model_dir, 'checkpoints')
                print(colored(f"args.pretrained_model_dir : {args.pretrained_model_dir}", "red", "on_yellow"))
        if args.Rehearsal_file is not None:
            if 'replay' not in args.Rehearsal_file and not args.eval :
                args.Rehearsal_file = os.path.join(args.Rehearsal_file, 'replay')
                print(colored(f"args.Rehearsal_file : {args.Rehearsal_file}", "red", "on_yellow"))
    
    def set_task_epoch(self, args, idx):
        epochs = self.Task_Epochs
        if len(epochs) > 1:
            args.Task_Epochs = epochs[idx]
        else:
            args.Task_Epochs = epochs[0]
    

    def make_branch(self, task_idx, args, is_init=False):
        self.update_class(task_idx)
        
        ## 고려하고 있는 case ##
        # case 1) start_task=0부터 시작해서 차근차근 task를 진행하는 경우
        #    이 경우는 task가 끝날 때마다 해당 task의 weight가 output_dir에 저장되므로,
        #    previous_weight를 output_dir에서 불러옴
        #
        # case 2) start_task=1, args.Rehearsal_file에서 task 0의 데이터 불러오는 경우 (전체 task 2개)
        #    이 경우는 previous_weight가 현재 output_dir에 없음
        #    따라서 args.pretrained_model에서 previous_weight를 불러옴
        #
        # case 3) start_task=1, args.Rehearsal_file에서 task 0의 데이터 불러오는 경우 (전체 task 3개 이상)
        #    이 경우, 초기에는 args.pretrained_model에서 previous_weight를 불러와야 하지만,
        #    task가 변할 경우 args.output_dir에서 previous_weight를 불러와야 함
        #
        # case 1, 2, 3를 모두 충족시키는 방법)
        #    main_component에서 make_branch가 참조되는 경우 is_init을 True로, main에서 참조되는 경우 False로 설정함.
        #    case 1의 경우는 어차피 args.pretrained_model이 선언되어 있지 않기 때문에 is_init이 항상 False임
        #    case 2,3의 경우 args.pretrained_model이 존재하기 때문에, is_init이 True인 경우와 False인 경우가 둘 다 존재함
        #        1) is_init==True
        #              해당 경우는 args.pretrained_model에서 previous_weight를 불러옴
        #        2) is_init==False
        #              해당 경우는 args.output_dir에서 previous_weight를 불러옴
        
        if is_init:
            weight_path = args.pretrained_model[0]
        else:
            weight_path = os.path.join(args.output_dir, f'checkpoints/cp_{self.tasks:02}_{task_idx:02}.pth')
            self.model, self.model_without_ddp, self.criterion, self.postprocessors, self.teacher_model = \
                self._build_and_setup_model(task_idx=task_idx)
            self.model = self.model_without_ddp = load_model_params("main", self.model, weight_path)
            
        previous_weight = torch.load(weight_path)
        print(colored(f"Branch_incremental weight path : {weight_path}", "red", "on_yellow"))

        try:
            if args.model_name == 'deform_detr':
                for idx, class_emb in enumerate(self.model.class_embed):
                    init_layer_weight = torch.nn.init.xavier_normal_(class_emb.weight.data)
                    previous_layer_weight = previous_weight['model'][f'class_embed.{idx}.weight']
                    previous_class_len = previous_layer_weight.size(0)

                    init_layer_weight[:previous_class_len] = previous_layer_weight
                    
            elif args.model_name == 'dn_detr':
                class_emb = self.model.class_embed
                label_enc = self.model.label_enc
                
                init_class_weight = torch.nn.init.xavier_normal_(class_emb.weight.data)
                init_label_weight = torch.nn.init.xavier_normal_(label_enc.weight.data)
                previous_class_weight = previous_weight['model']['class_embed.weight']
                previous_label_weight = previous_weight['model']['label_enc.weight']
                previous_class_len = previous_class_weight.size(0)
                previous_label_len = previous_label_weight.size(0)
                init_class_weight[:previous_class_len] = previous_class_weight
                init_label_weight[:previous_label_len] = previous_label_weight
        except:
            # LG pretrained model이 아니라 coco pretrained model을 사용할 때는 class, label weight 안가져옴
            print(colored(f"Num of class does not matched! : {weight_path}", "yellow", "on_red"))

    def update_class(self, task_idx):
        if self.args.Branch_Incremental is False:
            # Because original classes(whole classes) is 60 to LG, COCO is 91.
            num_classes = 60 if self.args.LG else 91
            current_class = None
        else:
            idx = len(self.Divided_Classes) if self.args.LG and self.args.eval else task_idx+1
            current_class = sum(self.Divided_Classes[:idx], [])
            num_classes = len(current_class) + 1
            
        previous_classes = sum(self.Divided_Classes[:task_idx], []) # For distillation options.
        self.previous_classes = previous_classes
        self.current_class = current_class
        self.num_classes = num_classes

    def _build_and_setup_model(self, task_idx):
        args = self.args
        self.update_class(task_idx)

        model, criterion, postprocessors = get_models(args.model_name, args, self.num_classes, self.current_class)
        
        if args.Distill or args.pseudo_labeling or args.Fake_Query:
            pre_model, _, _ = get_models(args.model_name, args, self.num_classes, self.current_class)
    
        if args.pretrained_model is not None and not args.eval:
            model = load_model_params("main", model, args.pretrained_model, args.Branch_Incremental)
        model_without_ddp = model
        
        teacher_model = None
        if args.Distill or args.pseudo_labeling or args.Fake_Query:
            teacher_model = load_model_params("teacher", pre_model, args.teacher_model, args.Branch_Incremental)
            print(f"teacher model load complete !!!!")
            return model, model_without_ddp, criterion, postprocessors, teacher_model
            
        return model, model_without_ddp, criterion, postprocessors, teacher_model
    

    def _setup_optimizer_and_scheduler(self):
        args = self.args
        
        def match_name_keywords(n, name_keywords):
            out = False
            for b in name_keywords:
                if b in n:
                    out = True
                    break
            return out
        
        param_dicts = [
            {
                "params":
                    [p for n, p in self.model_without_ddp.named_parameters()
                        if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
                "lr": args.lr,
            },
            {
                "params": [p for n, p in self.model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
                "lr": args.lr * args.lr_linear_proj_mult,
            },
            {
                "params": [p for n, p in self.model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]
            
        if args.sgd:
            optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                        weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                        weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

        return optimizer, lr_scheduler


    def load_ddp_state(self):
        args = self.args
        # For extra epoch training, because It's not affected to DDP.
        self.model = self.model.to(self.device)
        if args.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[args.gpu])
            self.model_without_ddp = self.model.module

        if args.frozen_weights is not None:
            checkpoint = torch.load(args.frozen_weights, map_location='cpu')
            self.model_without_ddp.detr.load_state_dict(checkpoint['model'])
            
            
    def _incremental_setting(self):
        args = self.args
        Divided_Classes = []
        start_epoch = 0
        start_task = 0
        tasks = args.Task
        Divided_Classes = create_dataset_for_incremental(args, args.eval)
        if args.Total_Classes_Names == True :
            # If you use the Total Classes names, you don't need to write args.tasks(you can use the any value)
            tasks = len(Divided_Classes)    
        
        if args.start_epoch != 0:
            start_epoch = args.start_epoch
        
        if args.start_task != 0:
            start_task = args.start_task
            
        dataset_name = "Original"
        if args.AugReplay :
            dataset_name = "AugReplay"

        return Divided_Classes, dataset_name, start_epoch, start_task, tasks
    

    def _load_replay_buffer(self):
        '''
            you should check more then two task splits. because It is used in incremental tasks
            1. criteria : tasks >= 2
            2. args.Rehearsal : True
            3. args.
        '''
        load_replay = []
        rehearsal_classes = {}
        args = self.args
        for idx in range(self.start_task):
            load_replay.extend(self.Divided_Classes[idx])
        
        load_task = 0 if args.start_task == 0 else args.start_task - 1
        
        #* Load for Replay
        if args.Rehearsal:
            rehearsal_classes = load_rehearsal(args.Rehearsal_file, load_task, args.limit_image)
            try:
                if len(list(rehearsal_classes.keys())) == 0:
                    print(f"No rehearsal file. Initialization rehearsal dict")
                    rehearsal_classes = {}
                else:
                    print(f"replay keys length :{len(list(rehearsal_classes.keys()))}")
            except:
                print(f"Rehearsal File Error. Generate new empty rehearsal dict.")
                rehearsal_classes = {}

        return load_replay, rehearsal_classes


    def evaluation_only_mode(self,):
        '''evaluation mode'''
        
        args = self.args
        print(colored(f"evaluation only mode start !!", "red"))
        
        def load_all_files(directory):
            all_files = []
            for root, _, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    all_files.append(file_path)
            return all_files
        
        def extract_last_number(filepath):
            filename = filepath.split('/')[-1]  # get the last part after '/'
            numbers = re.findall(r'\d+', filename)  # find all number sequences
            if numbers:
                return int(numbers[-1])  # return the last number
            else:
                return 0  # return 0 if there are no numbers
            

        # load all files in data
        if self.args.pretrained_model_dir is not None:
            self.args.pretrained_model = load_all_files(self.args.pretrained_model_dir)
            print(f"test directory list : {len(self.args.pretrained_model)}")
            args.pretrained_model.sort(key=extract_last_number)
            print(f"test directory examples : {self.args.pretrained_model}")
            
        for enum, predefined_model in enumerate(self.args.pretrained_model):
            print(colored(f"current predefined_model : {enum}, defined model name : {predefined_model}", "red"))
            
            if predefined_model is not None:
                self.model = load_model_params("eval", self.model, predefined_model)
            
            with open(self.DIR, 'a') as f:
                f.write(f"\n-----------------------pth file----------------------\n")
                f.write(f"file_name : {os.path.basename(predefined_model)}\n")  # 파일 이름
                f.write(f"file_path : {os.path.abspath(os.path.dirname(predefined_model))}\n")  # 파일 절대 경로
        
            test_epoch = 1 if args.Total_Classes != args.Test_Classes else args.Task
            Divided_Classes = create_dataset_for_incremental(args, False)
            for task_idx in range(test_epoch) :
                print(colored(f"evaluation task number {task_idx + 1} / {test_epoch}", "blue", "on_yellow"))
                
                dataset_val, data_loader_val, _, _  = Incre_Dataset(task_idx, args, Divided_Classes)
                base_ds = get_coco_api_from_dataset(dataset_val)
                
                with open(self.DIR, 'a') as f:
                    f.write(f"-----------------------task working----------------------\n")
                    f.write(f"NOW TASK num : {task_idx + 1} / {test_epoch}, checked classes : {sum(Divided_Classes[:task_idx+1], [])} \t ")
                    
                evaluate(self.model, self.criterion, self.postprocessors, data_loader_val, base_ds, self.device, args.output_dir, self.DIR, args)
                

    def incremental_train_epoch(self, task_idx, last_task, dataset_train, data_loader_train, sampler_train, list_CC, first_training=False):
        args = self.args
        self.list_cc = list_CC
        T_epochs = args.Task_Epochs[0] if isinstance(args.Task_Epochs, list) else args.Task_Epochs
        
        for epoch in range(self.start_epoch, T_epochs): 
            if args.distributed:
                sampler_train.set_epoch(epoch) 
            print(colored(f"task id : {task_idx} / {self.tasks-1}", "blue"))
            print(colored(f"each epoch id : {epoch} , Dataset length : {len(dataset_train)}, current classes :{list_CC}", "blue"))
            print(colored(f"Task is Last : {last_task} \t task is first : {first_training}", "blue"))
            
            #* Training process
            train_one_epoch(args, task_idx, last_task, epoch, self.model, self.teacher_model, self.criterion, dataset_train,
                            data_loader_train, self.optimizer, self.lr_scheduler,
                            self.device, self.dataset_name, list_CC, self.rehearsal_classes, first_training)
            
            #* set a lr scheduler.
            self.lr_scheduler.step()

            #* Save model each epoch
            save_model_params(model_without_ddp=self.model_without_ddp, optimizer=self.optimizer, lr_scheduler=self.lr_scheduler,
                            args=args, output_dir=args.output_dir, task_index=task_idx, total_tasks=int(self.tasks), epoch=epoch)
        
        #* If task change, training epoch should be zero.
        self.start_epoch = 0
        
        #* for task information at end training course
        save_model_params(model_without_ddp=self.model_without_ddp, optimizer=self.optimizer, lr_scheduler=self.lr_scheduler,
                        args=args, output_dir=args.output_dir, task_index=task_idx, total_tasks=int(self.tasks))
        
        #TODO: maybe delete the code here
        self.load_replay.extend(self.Divided_Classes[task_idx])
        
        #* distillation task and reload new teacher model.
        self.teacher_model = self.model_without_ddp
        self.teacher_model = teacher_model_freeze(self.teacher_model)

        if utils.get_world_size() > 1: dist.barrier()

    # No incremental learning process    
    def only_one_task_training(self):
        dataset_train, data_loader_train, sampler_train, list_CC = Incre_Dataset(0, self.args, self.Divided_Classes)
        
        print(f"Normal Training Process \n \
                Classes : {self.Divided_Classes}")
        
        # Normal training with each epoch
        self.incremental_train_epoch(task_idx=0, last_task=True, dataset_train=dataset_train,
                                        data_loader_train=data_loader_train, sampler_train=sampler_train,
                                        list_CC=list_CC)
        
    # No incremental learning process    
    def pseudo_work(self, re_gen=False, insufficient_objects=None, count=0):
        args = self.args
        generated_path = args.generator_path #* absolute_path
        json_file_name = 'annotations/pseudo_data.json'
        json_dir = os.path.join(generated_path, json_file_name)
        
        incremental_classes = self.Divided_Classes
        all_classes = sum(incremental_classes[:self.start_task], [])
        if args.new_gen :
            all_classes = incremental_classes[self.start_task] 
        max_class = max(all_classes)
        min_class = min(all_classes)
        print(colored(f"generating min classes : {min_class} / max classes : {max_class}.", "blue", "on_yellow"))
        
        if os.path.exists(json_dir) and re_gen==False:
            print(colored(f"{json_dir} already exists. Skipping making pseudo dataset.", "blue", "on_yellow"))
            return
        
        if utils.is_main_process():
            if os.path.exists(json_dir) and re_gen:
                existing_image_ids, insufficient_cats = get_existing_image_ids(json_dir, insufficient_objects)
            else:
                existing_image_ids = None
                insufficient_cats = None
            
            generated_image_path = os.path.join(generated_path, "images")   
            gen_dataset = PseudoDataset(generated_image_path, args, pseudo_path=json_dir, existing_ids=existing_image_ids, regen=re_gen)
            dataset_frame = gen_dataset.generate_data
            if re_gen:
                indicate_frame = gen_dataset.indicated_data
            
            sampler_train = torch.utils.data.SequentialSampler(gen_dataset)
            batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, batch_size=1, drop_last=False)
            
            data_loader = DataLoader(gen_dataset, batch_sampler=batch_sampler_train,
                                    collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                    pin_memory=True, prefetch_factor=args.prefetch)
            
            dataset_frame = pseudo_process(args=args, dataset_frame=dataset_frame, data_loader=data_loader, 
                                        image_paths=generated_image_path, model=self.teacher_model, device=self.device , insufficient_cats=insufficient_cats, count=count, min_class=min_class, max_class=max_class)
            
            if re_gen:
                indicate_frame['images'].extend(dataset_frame['images'])
                indicate_frame['annotations'].extend(dataset_frame['annotations'])
                dataset_frame = indicate_frame
                
            # 딕셔너리를 JSON 파일로 저장합니다.
            with open(json_dir, 'w') as f:
                json.dump(dataset_frame, f, indent=4)
                
            print(colored(f"{json_dir} has been successfully created.", "blue", "on_yellow"))
        
        #* if use MultiGPU, so then you should sync each GPUs
        if utils.get_world_size() > 1 : dist.barrier()
        
        return
            
    def generator(self):
        args = self.args
        incremental_classes = self.Divided_Classes
        all_classes = sum(incremental_classes[:self.start_task], []) #! previous classes (all previous)
        if args.new_gen :
            all_classes = incremental_classes[self.start_task] #! target classes (only current)
        max_class = max(all_classes)
        min_class = min(all_classes)
        print(colored(f"| generating | min classes : {min_class} / max classes : {max_class}.", "blue", "on_yellow"))
        
        
        #* blip, SD, GLIGEN processing. all generation samples generate here.
        print(colored(f"generating processing start.", "blue", "on_yellow"))
        generation_process(args, max_class, min_class)
        return
    
    # No incremental learning process    
    def labeling_check(self):
        '''
            gen_path = pseudo_data (refined data coco format)
            original_path = genarated coco path (initial gen path)
        '''
        # if utils.is_main_process():
        args = self.args
        generated_path = args.generator_path #* absolute_path
        refined_json = 'annotations/pseudo_data.json'
        refined_json_dir = os.path.join(generated_path, refined_json)
        
        incremental_classes = self.Divided_Classes
        all_classes = sum(incremental_classes[:self.start_task], []) #! previous classes (all previous)
        if args.new_gen :
            all_classes = incremental_classes[self.start_task] #! target classes (only current)
        max_class = max(all_classes)
        min_class = min(all_classes)
        print(colored(f"| label check | min classes : {min_class} / max classes : {max_class}.", "blue", "on_yellow"))
        
        if os.path.exists(refined_json_dir) is not True:
            print(colored(f"{refined_json_dir} is not exists. Can't check the pseudo labeling work.", "red", "on_yellow"))
            return
        
        initial_gen_json = "annotations/Total_coco.json"
        initial_gen_json_dir = os.path.join(generated_path, initial_gen_json) #* genarated coco path (initial gen path)
        generated_image_path = os.path.join(generated_path, "images")
        gen_dataset = PseudoDataset(generated_image_path, args, initial_gen_json_dir, refined_json_dir)
        
        #* check work for visualization
        # pseudo_data_for_check = gen_dataset.pseudo_data
        # original_data_for_check = gen_dataset.original_data
        # check_and_copy_different_annotations(pseudo=pseudo_data_for_check, origin=original_data_for_check, gen_path=generated_path)
        
        coco_json_name = 'annotations/instances_train2017.json'
        origin_coco_json_dir = os.path.join(args.coco_path, coco_json_name)
        insufficient_objects = gen_ratio_check(original_data_path=origin_coco_json_dir, gen_data_path=refined_json_dir, target_ratio=args.object_counts,min_c=min_class, max_c=max_class)
    
        # #* if use MultiGPU, so then you should sync each GPUs
        if utils.get_world_size() > 1 :
            dist.barrier()
        
        #* end
        print(colored(f"labeling check finish.", "blue", "on_yellow"))
        return insufficient_objects
    
    def regeneration(self, insufficient_objects, count = 0):
        '''
            re-generate the image as many objects as it lacks the standard (args.target_ratio)
        '''
        args = self.args
        check_insufficient = insufficient_objects
        
        if dist.get_world_size() > 1:
            gpus = dist.get_world_size()
            insufficient = {}
            for key, value in insufficient_objects.items():
                if value >= gpus:
                    insufficient[key] = int(value / gpus)
                else:
                    insufficient[key] = value
        else:
            insufficient = insufficient_objects.copy()
            
        args = self.args
        incremental_classes = self.Divided_Classes
        all_classes = sum(incremental_classes[:self.start_task], [])
        if args.new_gen :
            all_classes = incremental_classes[self.start_task] 
        max_class = max(all_classes)
        min_class = min(all_classes)
        print(colored(f"generating min classes : {min_class} / max classes : {max_class}.", "blue", "on_yellow"))
        
        print(colored(f"re-generating processing start.", "blue", "on_yellow"))
        generation_process(args, max_class, min_class, insufficient, count)
        
        #* end
        print(colored(f"re-generating processing finish.", "blue", "on_yellow"))
        if utils.get_world_size() > 1 :
            dist.barrier()
        return check_insufficient


        
from copy import deepcopy
def generate_dataset(first_training, task_idx, args, pipeline):
    #* Generate new dataset(current classes)
    #TODO: generate new dataset for Diff-DDETR (T1 Task)
    new_dataset, new_loader, new_sampler, list_CC = Incre_Dataset(task_idx, args, pipeline.Divided_Classes)

    if not first_training and args.Rehearsal:
        
        #* Ready for replay training strategy
        temp_replay_dataset = deepcopy(pipeline.rehearsal_classes)
        replay_dataset = dict(sorted(temp_replay_dataset.items(), key=lambda x: x[0]))
        previous_classes = sum(pipeline.Divided_Classes[:task_idx], []) # Not now current classe
        
        
        if args.AugReplay:
            fisher_dict = None
            AugRplay_dataset, AugRplay_loader, AugRplay_sampler = CombineDataset(
                args, replay_dataset, new_dataset, args.num_workers, args.batch_size, 
                old_classes=previous_classes, pseudo_training=False)
        else:
            fisher_dict = None
            AugRplay_dataset, AugRplay_loader, AugRplay_sampler = None, None, None
        
        #* re-check for valid option    
        assert (args.Mosaic and not args.AugReplay) or (not args.Mosaic and args.AugReplay) or (not args.Mosaic and not args.AugReplay)
            

        #* Combine dataset for original and AugReplay(Circular)
        original_dataset, original_loader, original_sampler = CombineDataset(
            args, replay_dataset, new_dataset, args.num_workers, args.batch_size, 
            old_classes=previous_classes, pseudo_training=False)

        aug_set = (AugRplay_dataset, AugRplay_loader, AugRplay_sampler)
        real_set = (original_dataset, original_loader, original_sampler)

        # Set a certain configuration
        new_dataset, new_loader, new_sampler = dataset_configuration(args, real_set, aug_set)
        
    if not first_training and args.pseudo_training :
        gen_dataset_train, _, _, previous_classes = Incre_Dataset(task_idx, args, pipeline.Divided_Classes, pseudo_dataset=True)
        new_dataset, new_loader, new_sampler = CombineDataset(
                args, gen_dataset_train, new_dataset, args.num_workers, args.batch_size, 
                old_classes=previous_classes, pseudo_training=True)

    return new_dataset, new_loader, new_sampler, list_CC