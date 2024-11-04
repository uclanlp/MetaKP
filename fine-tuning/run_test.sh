#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;

export CUDA_VISIBLE_DEVICES=0,1,2,3

HOME_DIR=`realpath ../`
DATA_DIR_PREFIX="${HOME_DIR}/data/"

OUT_DIR=$1  # specify OUT_DIR to where results will be stored
ROUND=$2   # final_with_neg final ablation_reject

# final_with_neg is for testing abstain scores
if [ "$ROUND" = "final_with_neg" ]; then
    DATA_DIR="json_with_rejection"
    FILE_NAME="hypotheses_full"
# final is for testing keyphrase semantic scores
elif [ "$ROUND" = "final" ]; then
    DATA_DIR="json_seq2seq"
    FILE_NAME="hypotheses_posonly"
elif [ "$ROUND" = "ablation_reject" ]; then
    DATA_DIR="json_with_rejection"
    FILE_NAME="hypotheses_ablation_reject"
fi

function decode-prefix-with-na () {
    EVAL_DATASET=$1

    # generation params
    NBEAMS=1

    BATCH_SIZE_PER_GPU=8
    
    python run_decode_with_prefix_vary_threshold.py \
           --model_name_or_path $OUT_DIR \
           --tokenizer_name $OUT_DIR \
           --test_file "${DATA_DIR_PREFIX}/${ROUND}/${EVAL_DATASET}/${DATA_DIR}/test.json" \
           --src_column "src" \
           --tgt_column "tgt" \
           --per_device_eval_batch_size $BATCH_SIZE_PER_GPU \
           --output_dir $OUT_DIR \
           --output_file_name "$OUT_DIR/${EVAL_DATASET}_${FILE_NAME}.txt" \
           --add_end_goal_token true \
	   --add_na_token true  \
           --num_beams ${NBEAMS} \
           --na_filename "na_probability_${EVAL_DATASET}.txt"  
           ### na file stores the <n/a> probability, change according to training data negperpos
           
}

function decode-prefix () {
    EVAL_DATASET=$1

    # generation params
    NBEAMS=1

    BATCH_SIZE_PER_GPU=8

    python run_decode_with_prefix_vary_threshold.py \
           --model_name_or_path $OUT_DIR \
           --tokenizer_name $OUT_DIR \
           --test_file "${DATA_DIR_PREFIX}/${ROUND}/${EVAL_DATASET}/${DATA_DIR}/test.json" \
           --src_column "src" \
           --tgt_column "tgt" \
           --per_device_eval_batch_size $BATCH_SIZE_PER_GPU \
           --output_dir $OUT_DIR \
           --output_file_name "$OUT_DIR/${EVAL_DATASET}_${FILE_NAME}.txt" \
           --add_end_goal_token true \
	   --add_na_token true \
           --remove_na_token \
           --num_beams ${NBEAMS} \
           --na_filename "na_probability_${EVAL_DATASET}.txt"  
           ### change according to training data negperpos
           
}

function abstain-evaluate () {
    EVAL_DATASET=$1
    
    python -W ignore ${HOME_DIR}/eval/evaluate_abstain_scores.py \
	   --data_path "${DATA_DIR_PREFIX}/${ROUND}/${EVAL_DATASET}/test.rejection_augmented_release.json" \
	   --prediction_path "$OUT_DIR/${EVAL_DATASET}_${FILE_NAME}.txt" \
    | tee "$OUT_DIR/${EVAL_DATASET}_hypotheses.abstain.eval.results.txt"
}


function evaluate () {
    EVAL_DATASET=$1
    
    python -W ignore ${HOME_DIR}/eval/evaluate_semantic_scores.py \
	   --data_path "${DATA_DIR_PREFIX}/${ROUND}/${EVAL_DATASET}/test.humanvalid_processed_release.json" \
	   --prediction_path "$OUT_DIR/${EVAL_DATASET}_${FILE_NAME}.txt" \
	   --output_path "$OUT_DIR/${EVAL_DATASET}_${FILE_NAME}.eval.log" \
	| tee "$OUT_DIR/${EVAL_DATASET}_${FILE_NAME}.eval.results.txt"
    
}


# inference

if [ "$ROUND" = "final_with_neg" ]; then
    for ds in kptimes duc2001 ; do  #duc2001 kpbiomed pubmed 
        decode-prefix-with-na ${ds}
        abstain-evaluate ${ds}
    done
elif [ "$ROUND" = "final" ]; then
    for ds in kptimes duc2001 kpbiomed pubmed ; do  #duc2001 kpbiomed pubmed 
        decode-prefix ${ds}
        evaluate ${ds}
    done
elif [ "$ROUND" = "ablation_reject" ]; then
    for ds in kptimes duc2001 kpbiomed pubmed ; do  
        decode-prefix-with-na ${ds}
        abstain-evaluate ${ds}
    done
fi


