#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;

export CUDA_VISIBLE_DEVICES=0

HOME_DIR=`realpath ../`
DATA_DIR_PREFIX="${HOME_DIR}/data/"
OUT_DIR=$1
DS=$2

function evaluate () {
    EVAL_DATASET=$1
    
    python -W ignore ${HOME_DIR}/eval/evaluate_semantic_scores.py \
	   --data_path "${DATA_DIR_PREFIX}/final/${EVAL_DATASET}/test.humanvalid_processed.json" \
	   --prediction_path "$OUT_DIR/${EVAL_DATASET}_hypotheses.txt" \
	   --output_path "$OUT_DIR/${EVAL_DATASET}_hypotheses.eval.log" \
	| tee "$OUT_DIR/${EVAL_DATASET}_hypotheses.eval.results.txt"
    
}

evaluate ${DS}
