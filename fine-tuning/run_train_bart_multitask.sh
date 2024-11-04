#!/usr/bin/env bash


export PYTHONIOENCODING=utf-8;
HOME_DIR=`realpath ../`


# Confirm GPUs, model, and dataset before training
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 #0,1,2,3 #

MODEL_NAME_OR_PATH="facebook/bart-large" #"facebook/bart-base" also change per device batch size accordingly
MODEL_NAME_SHORT="bart-large"    # for creating the output dir

DATASET=kpbiomed
ROUND=final_with_neg # ablation_reject  final_with_neg  final
REJNEG=1
REJSTRATEGY=sample #

# final_with_neg is for testing abstain scores
if [ "$ROUND" = "final_with_neg" ]; then
    DATA_DIR="json_with_rejection"
# final is for testing keyphrase semantic scores
# this is for ablation study purposes, 
# if training on keyphrase only, need to turn off the add_na_token flag
elif [ "$ROUND" = "final" ]; then
    DATA_DIR="json_seq2seq"
elif [ "$ROUND" = "ablation_reject" ]; then
    DATA_DIR="json_with_rejection"
fi

# remember to modify OUT_DIR to store checkpoints in some specific result folder
function train () {
    BATCH_SIZE_PER_GPU=1 #4 #  # 16 for bart-large
    GRAD_ACCUMULATION_STEPS=8 # 4 # 16
    N_EPOCHS=15
    N_WARMUP_STEPS=100  # 2000
    N_EVAL_STEPS=200
    LR=6e-5
    LR_schedule='polynomial'    # 'linear'
    SEED=1234
    
    OUT_DIR=$(date +'%Y%m%d-%H%M')_${DATASET}-${ROUND}_checkpoints_from_${MODEL_NAME_SHORT}_lr${LR}_${LR_schedule}_seed${SEED}_with_rejection_${REJNEG}negperpos_strategy${REJSTRATEGY}/
    mkdir -p ${OUT_DIR}/code_backup
    cp *.py *.sh ${OUT_DIR}/code_backup

    # give the argument --num_gpus=X to deepspeed if we don't use CUDA_VISIBLE_DEVICES   
    deepspeed run_train.py \
        --output_dir $OUT_DIR \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --do_train \
        --do_eval \
        --train_file "${DATA_DIR_PREFIX}/${DATASET}/${DATA_DIR}/train.json" \
        --validation_file "${DATA_DIR_PREFIX}/${DATASET}/${DATA_DIR}/valid.json" \
        --src_column "src" \
        --tgt_column "tgt" \
        --per_device_train_batch_size ${BATCH_SIZE_PER_GPU} \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps ${GRAD_ACCUMULATION_STEPS} \
        --num_train_epochs ${N_EPOCHS} \
        --learning_rate ${LR} \
        --lr_scheduler_type ${LR_schedule} \
        --warmup_steps ${N_WARMUP_STEPS} \
        --logging_strategy 'steps' \
        --logging_steps 50 \
        --evaluation_strategy 'steps' \
        --eval_steps ${N_EVAL_STEPS} \
        --save_strategy 'steps' \
        --save_steps ${N_EVAL_STEPS} \
        --save_total_limit 5 \
        --load_best_model_at_end true \
	--metric_for_best_model "f1" \
	--greater_is_better true \
        --overwrite_output_dir \
        --predict_with_generate \
        --seed ${SEED} \
        --deepspeed `realpath deepspeed_configs/normal_fp16_gpuonly.json` \
        --fp16 true \
        --half_precision_backend "auto" \
        --add_end_goal_token true \
        --add_na_token true \
        --mask_mkp_before_na true 

}


# training
DATA_DIR_PREFIX="${HOME_DIR}/data/${ROUND}/"
train
