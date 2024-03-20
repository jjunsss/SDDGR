import argparse
import numpy as np

import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Parent parser', add_help=False)

    #* Model
    parser.add_argument('--model_name', type=str, default='deform_detr', choices=['deform_detr', 'dn_detr']) # set model name
    parser.add_argument('--frozen_weights', type=str, default=None, help="Path to the pretrained model. If set, only the mask head will be trained")    

    # lr
    parser.add_argument('--clip_max_norm', default=0.1, type=float,help='gradient clipping max norm')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--sgd', action='store_true')

    # * Backbone
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--backbone', default='resnet50', type=str, help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true', help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'), help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')    

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--hidden_dim', default=256, type=int, help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--nheads', default=8, type=int, help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int, help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--two_stage', default=False, action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true', help="Train segmentation head if the flag is provided")

    #* Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false', help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--set_cost_bbox', default=5, type=float, help="L1 box coefficient in the matching cost")    
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    #* dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='/data/LG/real_dataset/total_dataset/didvepz/plustotal/', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--output_dir', default='./result/DIDPZ+VE', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)

    
    #* Setting 
    parser.add_argument('--LG', default=False, action='store_true', help="for LG Dataset process")
    parser.add_argument('--file_name', default='./saved_rehearsal', type=str)
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--prefetch', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--pretrained_model', default=None, type=str, nargs='+', help='resume from checkpoint')
    parser.add_argument('--pretrained_model_dir', default=None, type=str, help='test all parameters')
    parser.add_argument('--orgcocopath', action='store_true', help='for original coco directory path')

    #* Continual Learning 
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--start_task', default=0, type=int, metavar='N', help='start task, if you set the construct_replay method, \
                                                                                so then you should set the start_task value. becuase start_task is task number of construct replay options ')
    parser.add_argument('--Task', default=2, type=int, help='The task is the number that divides the entire dataset, like a domain.') #if Task is 1, so then you could use it for normal training.
    parser.add_argument('--Task_Epochs', default=[16], type=int, nargs='+', help='each Task epoch, e.g. 1 task is 5 of 10 epoch training.. ')
    parser.add_argument('--Total_Classes', default=59, type=int, help='number of classes in custom COCODataset. e.g. COCO : 80 / LG : 59')
    parser.add_argument('--Total_Classes_Names', default=False, action='store_true', help="division of classes through class names (DID, PZ, VE). This option is available for LG Dataset")
    parser.add_argument('--CL_Limited', default=0, type=int, help='Use Limited Training in CL. If you choose False, you may encounter data imbalance in training.')
    parser.add_argument('--divide_ratio', default='4040', type=str, help='Adjusting ratio of task classes. 4040 = 40:40 class; 7010 = 70:10 class; 1070 = 10:70 class')

    #* Replay strategy
    parser.add_argument('--Rehearsal', default=False, action='store_true', help="use Rehearsal strategy in diverse CL method")
    parser.add_argument('--AugReplay', default=False, action='store_true', help="use Our augreplay strategy in step 2")
    parser.add_argument('--MixReplay', default=False, action='store_true', help="1:1 Mix replay solution, First Circular Training. Second Original Training")
    parser.add_argument('--Mosaic', default=False, action='store_true', help="mosaic augmentation for autonomous training")
    parser.add_argument('--Rehearsal_file', default=None, type=str)
    parser.add_argument('--Construct_Replay', default=False, action='store_true', help="For cunnstructing replay dataset")
    
    parser.add_argument('--Sampling_strategy', default='hierarchical', type=str, help="hierarchical(ours), RODEO(del low unique labels), random \
                                                                                     , hier_highloss, hier_highlabels, hier_highlabels_highloss, hard(high labels)")

    parser.add_argument('--Sampling_mode', default='GM', type=str, help="normal, GM(GuaranteeMinimum, ours), ")
    parser.add_argument('--least_image', default=0, type=int, help='least image of each class, must need to exure_min mode')
    parser.add_argument('--limit_image', default=100, type=int, help='maximum image of all classes, must need to exure_min mode')
    parser.add_argument('--icarl_limit_image', default=1200, type=int, help='maximum image of icarl')
    parser.add_argument('--CER', default='fisher', type=str, help="fisher(ours), original, weight. This processes are used with \
                                                                   Augreplay ER")
    
    #* CL Strategy
    parser.add_argument('--Fake_Query', default=False, action='store_true', help="retaining previous task target through predict query")
    parser.add_argument('--Distill', default=False, action='store_true', help="retaining previous task target through predict query")
    parser.add_argument('--Branch_Incremental', default=False, action='store_true', help="MLP or something incremental with class")
    parser.add_argument('--teacher_model', default=None, type=str)
    parser.add_argument('--Continual_Batch_size', default=2, type=int, help='continual batch traiing method')
    parser.add_argument('--fisher_model', default=None, type=str, help='fisher model path')
    
    #* 정완 디버그
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--num_debug_dataset', default=10, type=int) # 디버그 데이터셋 개수

    #* eval
    parser.add_argument('--all_data', default=False, action='store_true', help ="save your model output image") # I think this option is depreciated, so temporarily use for 79 path, and modify later ... .
    parser.add_argument('--FPP', default=False, action='store_true', help="Forgetting metrics")
    parser.add_argument('--Test_Classes', default=45, type=int, help="2 task eval(coco) : T1=45 / T2=90, 3task eval(coco) T1=30 T2=60 T3=90\
                                                                      this value be used to config model architecture in the adequate task")
    
    #* WandB
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--prj_name', default='Diff_DDETR', type=str)
    parser.add_argument('--run_name', default='finetune', type=str)
    
    #* Generative Replay mode
    parser.add_argument('--pseudo_generation', default=False, action='store_true', help="retaining previous task target through predict query")
    parser.add_argument('--pseudo_labeling', default=False, action='store_true', help="retaining previous task target through predict query")
    parser.add_argument('--labeling_check', default=False, action='store_true', help="retaining previous task target through predict query")
    parser.add_argument('--pseudo_training', default=False, action='store_true', help="retaining previous task target through predict query")
    parser.add_argument("--gen_target_ratio", type=float,  default=0.2, help="")
    
    #* GLIGEN
    parser.add_argument("--no_plms", action='store_true', help="use DDIM instead. WARNING: I did not test the code yet")
    parser.add_argument("--blip2", action='store_true', help="use DDIM instead. WARNING: I did not test the code yet")
    parser.add_argument("--gen_batch", type=int, default=1, help="")
    parser.add_argument("--guidance_scale", type=float,  default=7.5, help="")
    parser.add_argument("--negative_prompt", type=str,  default="blurry, overlapping objects, distorted proportions, (monochrome), (grayscale), bad hands, deformed, lowres, error, normal quality,\
                                                                watermark, duplicate, worst quality, obscured faces, low visibility, unnatural colors, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, \
                                                                fewer digits, cropped, worst quality, low quality, (NSFW), extra limb, extra arms, rawing, painting, crayon, sketch, graphite,", help="")
    parser.add_argument("--max_length", type=int, default=5, help="glgien generation limits")
    parser.add_argument("--gen_length", type=int, default=25000, help="")
    parser.add_argument("--coco_generator", action='store_true', help="use DDIM instead. WARNING: I did not test the code yet")
    parser.add_argument("--generator_path", type=str,  default="/data/gen_dataset", help="")
    parser.add_argument("--pseudo_path", type=str,  default="/data/gen_dataset", help="")
    parser.add_argument("--gligen_path", type=str,  default="GLIGEN/gligen_checkpoints/checkpoint_generation_text.pth", help="")
    
    #* generative sufficient annotations for balance generation
    parser.add_argument('--balance_gen', default=False, action='store_true', help="generate instances for balancing in each categories")
    parser.add_argument('--new_gen', default=False, action='store_true', help="generate instances for balancing in each categories")
    parser.add_argument("--sufficient_instances", type=int, default=3, help="")
    parser.add_argument("--sufficient_box_size", type=float,  default=0.2, help="")
    parser.add_argument("--object_counts", type=int,  default=500, help="")
    
    return parser    


def deform_detr_parser(parser):
    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')

    # lr
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')

    # * Backbone
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float, help="position / size * scale")   

    # * Transformer f
    parser.add_argument('--dim_feedforward', default=1024, type=int, help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")    

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float, help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float, help="giou box coefficient in the matching cost")    

    # * Loss coefficients
    parser.add_argument('--cls_loss_coef', default=2, type=float)

    return parser