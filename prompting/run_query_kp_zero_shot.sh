#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
HOME_DIR=`realpath ../`


# Confirm GPUs, model, and dataset before training
export CUDA_VISIBLE_DEVICES=0,1,2,3  #0,1,2,3 #


DATA_DIR_PREFIX="${HOME_DIR}/data"
ROUND=final
METHOD=zero_shot
NUM_TOKEN=4000
MAX_TOKEN=30



function query_zero_shot () {
    DATASET=$1
    MODEL=$2

    python -W ignore ${HOME_DIR}/prompting/query_kp_zero_shot.py \
	   --data_file "${HOME_DIR}/data/${ROUND}/${DATASET}/test.humanvalid_processed_release.json" \
	   --out_path "${HOME_DIR}/prompting/${DATASET}_${METHOD}_${MODEL}_num_token=${NUM_TOKEN}_keyphrase.txt" \
       --num_token "${NUM_TOKEN}" \
       --model "${MODEL}" \
       --dataset "${DATASET}" \
       --maxtoken "${MAX_TOKEN}" \
    | tee "${HOME_DIR}/prompting/${DATASET}_${METHOD}_${MODEL}_num_token=${NUM_TOKEN}_keyphrase.output.txt"

}


function evaluate () {
    EVAL_DATASET=$1
    MODEL=$2
    
    python -W ignore ${HOME_DIR}/eval/evaluate_semantic_scores.py \
	   --data_path "${DATA_DIR_PREFIX}/${ROUND}/${EVAL_DATASET}/test.humanvalid_processed_release.json" \
	   --prediction_path "${HOME_DIR}/prompting/${EVAL_DATASET}_${METHOD}_${MODEL}_num_token=${NUM_TOKEN}_keyphrase.txt" \
       --output_path "${EVAL_DATASET}_${METHOD}_${MODEL}_num_token=${NUM_TOKEN}_keyphrase.eval.log" \
	| tee "${EVAL_DATASET}_${METHOD}_${MODEL}_num_token=${NUM_TOKEN}_keyphrase.eval.results.txt"    
}




for Model in  gpt-3.5-turbo-0125 ; do #gpt-4o-2024-05-13
    for DS in  kptimes ; do  #kptimes duc2001 kpbiomed pubmed
        query_zero_shot ${DS} ${Model}
        evaluate ${DS} ${Model}
    done
done


