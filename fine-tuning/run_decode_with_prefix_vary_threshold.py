#!/usr/bin/env python
# coding=utf-8
# based on https://github.com/huggingface/transformers/blob/v4.16.2/examples/pytorch/summarization/run_summarization_no_trainer.py
"""
Decoding a 🤗 Transformers model on keyphrase generation.
"""

import argparse
import logging
import math
import json
import os
import sys
import random
from pathlib import Path

import datasets
import nltk
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from filelock import FileLock
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartTokenizer,
    BartTokenizerFast,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    set_seed,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TemperatureLogitsWarper,
    LogitsProcessorList
)
from transformers.file_utils import get_full_repo_name, is_offline_mode
from transformers.utils.versions import require_version
from logits_warpers import TypicalLogitsWarper, NoRepeatKeyphraseLogitsProcessor, NoPresentKeyphraseLogitsProcessor
# from scibart.source.tokenization_scibart import SciBartTokenizer

            
logger = logging.getLogger(__name__)
consoleHandler = logging.StreamHandler()
logger.addHandler(consoleHandler)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


kpgen_name_mapping = {
    "kp20k": ("src", "tgt"),
    "kptimes": ("src", "tgt"),
}


def postprocess_text_kpgen(preds, labels, predsonly=False, verbose=False):
    preds = [list(set([x.strip() for x in pred.strip().split(';')])) for pred in preds]
    if not predsonly and labels:
        labels = [list(set([x.strip() for x in label.strip().split(';')])) for label in labels]
    if verbose:
        print(*preds, sep='\n')
    return preds, labels


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on keyphrase generation")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to " "padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text " "(useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=None, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help="The maximum total sequence length for validation "
        "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
        "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
        "param of ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--src_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the source texts (for keyphrase generation).",
    )
    parser.add_argument(
        "--tgt_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the keyphrases (for keyphrase generation).",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Dir for saving eval log.")
    parser.add_argument("--output_file_name", type=str, required=True, help="Output hypothesis file name.")
    parser.add_argument("--write_raw", type=bool, default=True, help="Write raw text lines (True) or json with phrases splitted (False).")

    # special arguments for decoding experiments
    parser.add_argument(
        "--decode_mode_api",
        type=str,
        default='generate',
        choices=['generate', 'greedy_search', 'sample', 'beam_search', 'beam_sample', 
                 'group_beam_search', 'constrained_beam_search'],
        help="Huggingface decode API: 'generate', 'greedy_search', 'sample', 'beam_search', "
        "'beam_sample', 'group_beam_search', 'constrained_beam_search'",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Number of sequences to return. The resulting phrases will be pooled together for "
        "each example.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``.",
    )
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=1,
        help="Used for group_beam_search decode mode (https://arxiv.org/pdf/1610.02424.pdf)",
    )
    # currently, don't need to try 'no_bad_words', 'prefix_constrained' for logis_processor
    # parser.add_argument(
    #     "--logits_processor",
    #     nargs='*',
    #     type=str,
    #     default=None,
    #     choices=['minlen', 'repetition_penalty', 'no_repeat_ngram', 'hamming_diversity'],
    #     help="Add Huggingface logits processor: ",
    # )
    # parser.add_argument(
    #     "--logits_warper",
    #     nargs='*',
    #     type=str,
    #     default=None,
    #     choices=['temperature', 'top_p', 'top_k', 'typical'],
    #     help="Add Huggingface logits warper: ",
    # )
    parser.add_argument(
        "--spm_model_file",
        type=str,
        help="Sentencepiece model file corresponding to the scibart model.",
        default=None,
    )
    
    # logit warper
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="temperature for temperature logits warper",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="k for top k logits warper",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="p for top p logits warper",
    )
    parser.add_argument(
        "--typical_mass",
        type=float,
        default=None,
        help="mass for typical decoding ",
    )

    # logit processors
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=-1,
        help="n for ngram that are not allow to repeat",
    )
    parser.add_argument(
        "--no_repeat_kp",
        default=False,
        action='store_true',
        help="do not allow repeated keyphrases in the generated string",
    )
    parser.add_argument(
        "--encoder_no_repeat_ngram_size",
        type=int,
        default=-1,
        help="n for ngram that are not allow to repeat in the encoder's inputs",
    )

    # for contrastve decoding
    parser.add_argument(
        "--penalty_alpha",
        type=float,
        default=None,
        help="penalty alpha for contrastive decoding (use with top k)",
    )

    # for blocking keyphrases 
    parser.add_argument(
        "--sep_token_str",
        type=str,
        default=' ;',
        help="The separator token (please be careful about white spaces)",
    )
    parser.add_argument(
        "--no_pkp",
        default=False,
        action='store_true',
        help="do not allow keyphrase to repeat ngrams in the input text",
    )
    parser.add_argument(
        "--encoder_max_ngram_per_kp",
        type=int,
        default=-1,
        help="n for ngram that are not allow to repeat in the encoder's inputs",
    )

    # for collecting the entropy
    parser.add_argument(
        "--record_entropy",
        default=False,
        action='store_true',
        help="Record the entropy of generated phrases. Only support 1 sample at a time.",
    )
    parser.add_argument("--entropy_file_name", type=str, help="Output entropy file name.")
    
    # for adding noise before decoding
    parser.add_argument(
        "--add_noise",
        default=False,
        action='store_true',
        help="add Gaussian noise on the encoder representations (only supporting BART for now).",
    )
    parser.add_argument("--noise_stdev", type=float, default=None, help="Standard deviation for the noise.")
    
    parser.add_argument("--decode_verbose", action='store_true', help="print out postprocessed predictions.")

    # specific to MetaKP
    parser.add_argument(
        "--add_end_goal_token",
        type=bool,
        default=None,
        help="Add the special <end_goal> token to the vocab.",
    )

    parser.add_argument(
        "--add_na_token",
        type=bool,
        default=None,
        help="Add the special <n/a> token to the vocab.",
    )

    parser.add_argument(
        "--remove_na_token",
        default=False,
        action='store_true',
        help="Remove the special <n/a> token while decoding with prefix.",
    )

    parser.add_argument(
        "--na_filename",
        type=str,
        default="na_probability.txt",
        help="The name of the na_probability file to use.",
    )
    
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.test_file is None:
        raise ValueError("Need either a dataset name or a testing file.")
    else:
        if args.test_file is not None:
            extension = args.test_file.split(".")[-1]
            assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."

    if args.record_entropy:
        assert args.num_return_sequences == 1

    if args.add_noise:
        assert args.noise_stdev is not None

    return args


def get_test_dataset(args, accelerator, tokenizer):
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        assert args.test_file is not None
        data_files = {}
        data_files["test"] = args.test_file
        extension = args.test_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)

    prefix = args.source_prefix if args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["test"].column_names

    # Get the column names for input/target.
    dataset_columns = kpgen_name_mapping.get(args.dataset_name, None)
    if args.src_column is None:
        src_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        src_column = args.src_column
        if src_column not in column_names:
            raise ValueError(
                f"--src_column' value '{args.src_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.tgt_column is None:
        tgt_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        tgt_column = args.tgt_column
        if tgt_column not in column_names:
            raise ValueError(
                f"--tgt_column' value '{args.tgt_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        inputs = examples[src_column]
        targets = examples[tgt_column]        
        
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    return processed_datasets["test"]


def entropy(scores):
    normalized = nn.functional.log_softmax(scores, dim=-1)
    p = torch.exp(normalized)
    ent = -(normalized * p).nansum(-1, keepdim=True)
    return ent


def main():
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Inference on device: " + str(device))

    # log to both file and stderr
    fileHandler = logging.FileHandler("{}/test.log".format(args.output_dir))
    logger.addHandler(fileHandler)

    if args.source_prefix is None and args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    if args.tokenizer_name:
        if 'scibart' in args.tokenizer_name:
            raise NotImplementedError
            #assert args.spm_model_file is not None
            #tokenizer = SciBartTokenizer(args.spm_model_file)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    if args.add_end_goal_token:
        tokenizer.add_tokens(['<end_goal>'])
        print('Added <end_goal> token, resizing model embedding to {}'.format(len(tokenizer)))
    if args.add_na_token:
        tokenizer.add_tokens(['<n/a>'])
        print('Added <n/a> token, resizing model embedding to {}'.format(len(tokenizer)))
    model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    
    handle = None
    if args.add_noise:
        assert 'bart' in args.model_name_or_path.lower()
        def add_gaussian_noise_hook(module, input, output):
            noise = torch.randn_like(output[0]) * args.noise_stdev
            out = (output[0].clone() + noise,)
            return out
        handle = model.model.encoder.layers[-1].register_forward_hook(add_gaussian_noise_hook)
        print('Hook registered')
        
    model = model.to(device)
    logger.info("Loaded config, tokenizer, and model from " + args.model_name_or_path)
    logger.info("Tokenizer: " + str(type(tokenizer)))

    model.resize_token_embeddings(len(tokenizer))
    logger.info("Final embedding size: {}".format(len(tokenizer)))
    
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # get datasets
    test_dataset = get_test_dataset(args, accelerator, tokenizer)

    # Log a few random samples from the test set:
    for index in random.sample(range(len(test_dataset)), 1):
        logger.info(f"Sample {index} of the test set: {test_dataset[index]}.")

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    eval_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    logger.info("***** Running decode *****")
    logger.info(f"  Num examples = {len(test_dataset)}")
    logger.info(f"  Total eval batch size = {args.per_device_eval_batch_size}")

    model.eval()
    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length
    

    # basic arguments for each decode mode
    if args.decode_mode_api == 'generate':
        gen_kwargs = {
            "max_length": args.val_max_target_length,
            "num_beams": args.num_beams,
            "num_return_sequences": args.num_return_sequences,
        }
        if args.penalty_alpha is not None and args.penalty_alpha > 0:
            assert args.top_k is not None
            gen_kwargs['penalty_alpha'] = args.penalty_alpha
    elif args.decode_mode_api == 'greedy_search':
        gen_kwargs = {
            "max_length": args.val_max_target_length,
            "num_beams": 1,
        }
    elif args.decode_mode_api == 'sample':
        gen_kwargs = {
            "max_length": args.val_max_target_length,
            "num_beams": 1,
            "do_sample": True,
            "num_return_sequences": args.num_return_sequences,
        }
    elif args.decode_mode_api == 'beam_search':
        assert args.num_beams > 1 and args.num_beams >= args.num_return_sequences
        gen_kwargs = {
            "num_beams": args.num_beams,
            "num_return_sequences": args.num_return_sequences,
        }
    elif args.decode_mode_api == 'beam_sample':
        assert args.num_beams > 1
        gen_kwargs = {
            "num_beams": args.num_beams,
            "do_sample": True,
            "num_return_sequences": args.num_return_sequences,
        }
    elif args.decode_mode_api == 'group_beam_search':
        assert args.num_beams > 1 and args.num_beam_groups > 1
        gen_kwargs = {
            "num_beams": args.num_beams,
            "num_beam_groups": args.num_beam_groups,
            "do_sample": True,
        }
    elif args.decode_mode_api == 'constrained_beam_search':
        raise NotImplementedError
    else:
        raise NotImplementedError
    

    def get_custom_logits_processors_warpers_per_batch(args, enc_input_ids, tokenizer=None):
        # logits processors
        logit_processors = []
        if args.no_repeat_ngram_size is not None and args.no_repeat_ngram_size < 0:
            # let generate associate the logit processor
            args.no_repeat_ngram_size = None
        if args.encoder_no_repeat_ngram_size is not None and args.encoder_no_repeat_ngram_size < 0:
            # let generate associate the logit processor
            args.encoder_no_repeat_ngram_size = None
        if args.no_repeat_kp:
            sep_token = tokenizer.encode(args.sep_token_str, add_special_tokens=False)
            if len(sep_token) > 1:
                raise NotImplementedError('Currently only length 1 separator token is supported')
            if type(tokenizer) in [BartTokenizer, BartTokenizerFast]: #, SciBartTokenizer]:
                prefix_len = 2
            else:
                raise NotImplementedError
            logit_processors.append(
                NoRepeatKeyphraseLogitsProcessor(sep_token=sep_token[0], prefix_len=prefix_len,
                                                 eos_token=tokenizer.eos_token_id)
            )
        if args.no_pkp:
            sep_token = tokenizer.encode(args.sep_token_str, add_special_tokens=False)
            assert args.encoder_max_ngram_per_kp != -1
            logit_processors.append(
                NoPresentKeyphraseLogitsProcessor(args.encoder_max_ngram_per_kp, enc_input_ids,
                                                  sep_token=sep_token[0],
                                                  eos_token=tokenizer.eos_token_id,
                                                  prefix_len=prefix_len,
                                                  tokenizer=tokenizer)
            )
            
        # logits warpers
        logit_warpers = []
        if args.temperature is not None and args.temperature > 0:
            logit_warpers.append(TemperatureLogitsWarper(args.temperature))
        if args.top_k is not None and args.top_k > 0:
            if args.penalty_alpha is None or args.penalty_alpha <= 0:
                logit_warpers.append(TopKLogitsWarper(args.top_k))
            else:
                # contrastive search specified. Let generate handle top_k
                pass
        if args.top_p is not None and args.top_p > 0:
            logit_warpers.append(TopPLogitsWarper(args.top_p))
        if args.typical_mass is not None and args.typical_mass > 0:
            logit_warpers.append(TypicalLogitsWarper(args.typical_mass))
            
        # final list of processors
        if len(logit_processors + logit_warpers) > 0:        
            logit_processors_and_warpers = LogitsProcessorList(logit_processors + logit_warpers)
        else:
            logit_processors_and_warpers = []
        
        print("logit processors:", logit_processors_and_warpers)
        return logit_processors_and_warpers

    print(args)
    

    # run generate
    valid_preds = []
    valid_entropy_preds = []
    valid_preds_unmerged = []   # for returning multiple sequences
    na_probability = []
    for step, batch in tqdm(enumerate(eval_dataloader), desc='Decoding'):
        with torch.no_grad():
            batch = batch.to(device)


            processors = get_custom_logits_processors_warpers_per_batch(args,
                                                                        batch["input_ids"],
                                                                        tokenizer)
            
            print(args.remove_na_token)                                                
            if args.remove_na_token:
                na_token_id = tokenizer.encode('<n/a>', add_special_tokens=False)[0]
                all_ids = [i for i in range(len(tokenizer)) if i != na_token_id] 
                print("<n/a> token removed")
            else:
                all_ids = [i for i in range(len(tokenizer))]  
                print("<n/a> token not removed")
            print(len(all_ids))

            def prefix_allowed_tokens_fn(batch_id, prefix):
                """
                MetaKP's proposal to prefix on goals and selectively generate keyphrases.
                User goals are the prefix for constrained decoding, and is followed by <end_goal> token,
                afterwards, the model needs to decide if it the goal is relevant, and then selectively generate keyphrases corresponding to the goal.
                If the model rejects the goal, then an <n/a> token is generated.
                """
                
                idx = step * len(batch) + batch_id
        
                end_goal_token = [tokenizer.encode('<end_goal>', add_special_tokens=False)[0]]
                if args.decode_mode_api == 'sample':
                    original_batch_id = batch_id // args.num_return_sequences
                    label_tokens = batch['labels'].detach().cpu().numpy().tolist()[original_batch_id]
                else:
                    label_tokens = batch['labels'].detach().cpu().numpy().tolist()[batch_id]

                if 't5' in args.model_name_or_path.lower():
                    cur_goal_tokens = label_tokens[:label_tokens.index(end_goal_token[0])]
                else:
                    cur_goal_tokens = label_tokens[1:label_tokens.index(end_goal_token[0])]
                
                prefix_list = prefix.detach().cpu().numpy().tolist()

                if 't5' in args.model_name_or_path.lower():
                    assert prefix_list[0] == 0
                    if len(prefix_list) < 1 + len(cur_goal_tokens):
                        idx = len(prefix_list) - 1
                        return [cur_goal_tokens[idx]]
                    elif len(prefix_list) == 1 + len(cur_goal_tokens):
                        return end_goal_token
                    else:
                        return all_ids
                else:
                    if len(prefix_list) <= 1:
                        assert prefix_list[0] == 2
                        return [0]
                    elif len(prefix_list) < 2 + len(cur_goal_tokens):
                        idx = len(prefix_list) - 2
                        return [cur_goal_tokens[idx]]
                    elif len(prefix_list) == 2 + len(cur_goal_tokens):
                        return end_goal_token
                    else:
                        return all_ids
                    
                return all_ids
            
            # accumulate output for f1 score calculation
            outputs = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                encoder_no_repeat_ngram_size=args.encoder_no_repeat_ngram_size,
                logits_processor=processors,
                output_scores=True,
                return_dict_in_generate=True,
                **gen_kwargs,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn  ## PREFIX
            )

            generated_tokens, logits = outputs.sequences, outputs.scores

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            ### note generated_tokens shape is one large than logits, bc the first decoder start token has not probabilities
            

            probabilities = [torch.softmax(logit, dim=-1) for logit in logits]
            # print("sum of logits", torch.sum(logits[0], dim=-1))
            # print("sum of probabilities", torch.sum(probabilities[0], dim=-1))
            
            end_goal_token_id = tokenizer.encode('<end_goal>', add_special_tokens=False)[0]
            na_token_id = tokenizer.encode('<n/a>', add_special_tokens=False)[0]

            # 'generated_tokens' is the output from the generation process
            # and has shape (batch_size, sequence_length)

            batch_size, sequence_length = generated_tokens.shape

            # initialize a list to store the probabilities of <n/a> after <end_goal>
            na_probabilities_after_end_goal = [-1] * batch_size  # Default value indicating <end_goal> not found

            for i in range(batch_size):
                # Find the index of <end_goal> in the sequence
                end_goal_index = (generated_tokens[i] == end_goal_token_id).nonzero(as_tuple=True)[0]

                if len(end_goal_index) > 0:
                    # Get the first occurrence of <end_goal>
                    end_goal_index = end_goal_index[0].item()

                    if end_goal_index + 1 < sequence_length:
                        # Get the probability of <n/a> at the position following <end_goal>
                        na_prob_after_end_goal = probabilities[end_goal_index][i][na_token_id].item()
                        na_probabilities_after_end_goal[i] = na_prob_after_end_goal

            # Print the probabilities of <n/a> after <end_goal> for each sequence in the batch
            print("Probabilities of <n/a> after <end_goal>:", na_probabilities_after_end_goal)
            
            na_probability.extend(na_probabilities_after_end_goal)
            # sys.exit()

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]

            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()

            # record phrase-wise entropy
            if args.record_entropy:
                scores = torch.stack([x for x in scores], 1).detach()
                entropy_scores = entropy(scores)
                decoded_preds_for_entropy = tokenizer.batch_decode(generated_tokens,
                                                                   skip_special_tokens=False)
                tokens_list = generated_tokens.tolist()
                for b in range(len(tokens_list)):
                    cur_tokens = tokens_list[b]
                    cur_entropy = {}
                    cur_phrase_word_list, cur_phrase_entropy_list = [], []
                    for i_token, token in enumerate(cur_tokens):
                        cur_word = tokenizer.decode(token)
                        if cur_word in ['</s>', '<pad>']:
                            continue
                        elif cur_word.strip() == ';':
                            if cur_phrase_word_list:
                                cur_phrase = ''.join(cur_phrase_word_list)
                                cur_entropy[cur_phrase] = cur_phrase_entropy_list
                                cur_phrase_word_list, cur_phrase_entropy_list = [], []
                        else:
                            cur_phrase_entropy_list.append(entropy_scores[b][i_token-1].item())
                            cur_phrase_word_list.append(cur_word)
                        
                    valid_entropy_preds.append(cur_entropy)

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            print(decoded_preds)

            assert len(decoded_preds) == args.per_device_eval_batch_size * args.num_return_sequences or step == len(eval_dataloader) - 1
            decoded_preds_unmerged = None
            if args.num_return_sequences > 1:
                # merge all results into a single sequence
                decoded_preds_unmerged = []
                decoded_preds_merged = []
                # for b in range(args.per_device_eval_batch_size):
                for b in range(len(decoded_preds) // args.num_return_sequences):
                    decoded_preds_merged.append(' ; '.join(decoded_preds[args.num_return_sequences*b:args.num_return_sequences*(b+1)]))
                    decoded_preds_unmerged.append('<samplesep>'.join(decoded_preds[args.num_return_sequences*b:args.num_return_sequences*(b+1)]))
                decoded_preds = decoded_preds_merged
                assert len(decoded_preds) == args.per_device_eval_batch_size or step == len(eval_dataloader) - 1

            decoded_preds, _ = postprocess_text_kpgen(decoded_preds, None, predsonly=True, verbose=args.decode_verbose)
            if args.write_raw:
                decoded_preds = [' ; '.join(x) for x in decoded_preds]

            valid_preds.extend(decoded_preds)
            if decoded_preds_unmerged is not None:
                valid_preds_unmerged.extend(decoded_preds_unmerged)
            

    accelerator.wait_for_everyone()
    with open(args.output_file_name, 'w') as f:
        if args.num_return_sequences > 1:
            f_unmerged = open(args.output_file_name + '.unmerged', 'w')

        for i, pred in enumerate(valid_preds):
            if args.write_raw:
                f.write(pred)
            else:
                f.write(json.dumps(pred))
            f.write('\n')

            if args.num_return_sequences > 1:
                f_unmerged.write(valid_preds_unmerged[i])
                f_unmerged.write('\n')

    with open(args.na_filename, 'w') as f:
        for i, prob in enumerate(na_probability):
            if args.write_raw:
                f.write(str(prob))
            else:
                f.write(json.dumps(str(prob)))
            f.write('\n')

    if args.record_entropy and args.entropy_file_name:
        with open(args.entropy_file_name, 'w') as f:
            for entry in valid_entropy_preds:
                print(json.dumps(entry), file=f)
                

if __name__ == "__main__":
    main()
