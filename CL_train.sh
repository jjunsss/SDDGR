#!/usr/bin/env bash

set -x

# Define variables
PUS_PER_NODE=4
BATCH_SIZE=8
MODEL_NAME="dn_detr"
PREFETCH_SIZE=2
COCO_PATH=
OUTPUT_DIR=
START_TASK=
START_EPOCH=
TASK_EPOCHS=30
NUM_WORKERS=12
TOTAL_CLASSES=59
TEST_FILE_LIST="did pz ve"
PRETRAINED_MODEL=

# replay method
LIMIT_IMAGE=1200
LEAST_IMAGE=12 # 1%
TASK=3
REHEARSAL_FILE=
SAMPLING_STRATEGY="hierarchical"
SAMPLING_MODE="GM"

# Fisher 
FISHER_MODEL=
CER_MODE="fisher"

# Prepare the command
CMD="PUS_PER_NODE=4 ./tools/run_dist_launch.sh $PUS_PER_NODE ./configs/r50_dn_detr.sh \
    --batch_size $BATCH_SIZE \
    --prefetch $PREFETCH_SIZE \
    --model_name $MODEL_NAME \
    --coco_path $COCO_PATH \
    --output_dir $OUTPUT_DIR \
    --test_file_list $TEST_FILE_LIST \
    --orgcocopath \
    --use_dn \
    
    --Branch_Incremental \
    --Total_Classes $TOTAL_CLASSES \
    --Total_Classes_Names \
    --Task_Epochs $TASK_EPOCHS \
    --start_task $START_TASK \
    --start_epoch $START_EPOCH \
    --num_workers $NUM_WORKERS \
    --Task $TASK \
    --verbose \
    --LG \
    --Rehearsal_file $REHEARSAL_FILE \
    
    --Rehearsal \
    --limit_image $LIMIT_IMAGE \
    --least_image $LEAST_IMAGE \
    --Sampling_strategy $SAMPLING_STRATEGY \
    --Sampling_mode $SAMPLING_MODE \
    
    --AugReplay \
    --CER $CER_MODE \
    --fisher_model $FISHER_MODEL \
    $@"

# Print the command
echo $CMD

# Run the command
eval $CMD

