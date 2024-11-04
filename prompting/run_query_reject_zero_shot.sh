#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
HOME_DIR=`realpath ../`


# Confirm GPUs, model, and dataset
export CUDA_VISIBLE_DEVICES=0,1,2,3  


DATA_DIR_PREFIX="${HOME_DIR}/data"
ROUND=final_with_neg
METHOD=zero_shot
NUM_SENT=5

function query_zero_shot () {
    DATASET=$1
    MODEL=$2

    python -W ignore ${HOME_DIR}/prompting/query_rejection_zero_shot.py \
	   --data_file "${HOME_DIR}/data/final_with_neg/${DATASET}/test.rejection_augmented_release.json" \
	   --out_path "${HOME_DIR}/prompting/${DATASET}_${METHOD}_${MODEL}_num_sent=${NUM_SENT}_rejection.txt" \
       --num_sentence "${NUM_SENT}" \
       --model "${MODEL}" \
       --dataset "${DATASET}" \


}


function evaluate () {
    EVAL_DATASET=$1
    MODEL=$2

    python -W ignore ${HOME_DIR}/eval/evaluate_abstain_scores.py \
	   --data_path "${DATA_DIR_PREFIX}/${ROUND}/${EVAL_DATASET}/test.rejection_augmented_release.json" \
	   --prediction_path "${HOME_DIR}/prompting/${EVAL_DATASET}_${METHOD}_${MODEL}_num_sent=${NUM_SENT}_rejection.txt" \
    | tee "${EVAL_DATASET}_${METHOD}_${MODEL}_num_sent=${NUM_SENT}_rejection.abstain.eval.results.txt"
}



for Model in gpt-3.5-turbo-0125; do 
    for DS in kptimes ; do  #kptimes duc2001 kpbiomed pubmed
        query_zero_shot ${DS} ${Model}
        evaluate ${DS} ${Model}
    done
done