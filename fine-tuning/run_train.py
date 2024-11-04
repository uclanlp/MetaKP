#!/usr/bin/env python
# coding=utf-8
# based on https://github.com/huggingface/transformers/blob/v4.16.2/examples/pytorch/summarization/run_summarization.py
"""
Fine-tuning a 🤗 Transformers model on keyphrase generation.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import nltk
from nltk.stem.porter import *
import torch
import numpy as np
import random
from datasets import load_dataset

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers.integrations import TensorBoardCallback
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy
from transformers.deepspeed import is_deepspeed_zero3_enabled  

from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from dataclasses import dataclass

stemmer = PorterStemmer()

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.15.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)
consoleHandler = logging.StreamHandler()
logger.addHandler(consoleHandler)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


@dataclass
class DataCollatorForMKPSeq2Seq:
    ''' Adapted from DataCollatorForSeq2Seq '''
    
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    
    # For MetaKP
    mask_tokens_before_na: bool = False
    mask_mkp_tokens: bool = False
    end_goal_token_id: int = -100
    na_token_id: int = -100

    def __call__(self, features, return_tensors=None):
        
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        
        # masking for metakp
        labels_masked = []
        for label in labels:
            label = torch.tensor(label)
            
            if self.mask_mkp_tokens:
                # mask everything to the left of <end_goal> 
                end_goal_pos_mask = label.eq(self.end_goal_token_id).float()
                mask = end_goal_pos_mask.cumsum(axis=-1)
                mask = mask.logical_and(1 - end_goal_pos_mask).float()   # also mask <end_goal>
                
                # make sure do not mask anything if <end_goal> is not in the sequence
                clean_mask = torch.logical_not(torch.any(label == self.end_goal_token_id)).float()
                mask[clean_mask == 1] = 1
                
                label[mask == 0] = self.label_pad_token_id
            
            elif self.mask_tokens_before_na:
                # mask everything to the left of <n/a>
                na_pos_mask = label.eq(self.na_token_id).float()
                mask = na_pos_mask.cumsum(axis=-1)
                
                # make sure do not mask anything if <n/a> is not in the sequence
                clean_mask = torch.logical_not(torch.any(label == self.na_token_id)).float()
                mask[clean_mask == 1] = 1
                
                label[mask == 0] = self.label_pad_token_id
        
            labels_masked.append(label.tolist())
                
        labels = labels_masked
        
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features
    
class MKPSeq2SeqTrainer(Seq2SeqTrainer):
    '''
    Trainer with prefix-controlled generation for validation
    '''
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs):
        all_ids = [i for i in range(len(self.tokenizer))] 
        target_strings = self.tokenizer.batch_decode(inputs['decoder_input_ids'])
        def prefix_allowed_tokens_fn(batch_id, prefix):
            cur_prefix_list = prefix.detach().cpu().numpy().tolist()            
            target_prefix_string = target_strings[batch_id].split('<end_goal>')[0] + '<end_goal>'
            target_prefix_list = self.tokenizer.encode(target_prefix_string, add_special_tokens=False)            
            if len(cur_prefix_list) < len(target_prefix_list):
                return target_prefix_list[len(cur_prefix_list)]
            else:
                return all_ids

        #######################################################################
        # Original logic; the only change is we do prefix-controlled gen
        #######################################################################
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        gen_kwargs = {
            # "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "max_length": 200,
            #"num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "num_beams": 1,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
        }
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )
        
        inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}
        
        # only change is here
        generation_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs,
                                               prefix_allowed_tokens_fn=prefix_allowed_tokens_fn)
        
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, generated_tokens, labels)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
            "the model's position embeddings."
        },
    )
    
    # specific to MetaKP
    add_end_goal_token: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Add the special <end_goal> token to the vocab."
        },
    )
    add_na_token: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Add the special <n/a> token to the vocab."
        },
    )
    mask_mkp_before_na: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Mask out the MKP before special <n/a> token for loss calculation."
        },
    )
    mask_mkp_tokens: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Mask out all the MKP parts."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    src_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the source texts (for keyphrase generation)."},
    )
    tgt_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the keyphrases (for keyphrase generation)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    valid_without_contraint: bool = field(
        default=False,
        metadata={
            "help": "Run validation without constained decoding."
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


kpgen_name_mapping = {
    "kp20k": ("src", "tgt"),
}


def load_and_preprocess_datasets(model_args, data_args, training_args, tokenizer, model):
    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # keyphrases (unless you specify column names for this with the `src_column` and `tgt_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    dataset_columns = kpgen_name_mapping.get(data_args.dataset_name, None)
    if data_args.src_column is None:
        src_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        src_column = data_args.src_column
        if src_column not in column_names:
            raise ValueError(
                f"--src_column' value '{data_args.src_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.tgt_column is None:
        tgt_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        tgt_column = data_args.tgt_column
        if tgt_column not in column_names:
            raise ValueError(
                f"--tgt_column' value '{data_args.tgt_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        # remove pairs where at least one record is None
        inputs, targets = [], []
        for i in range(len(examples[src_column])):
            if examples[src_column][i] is not None and examples[tgt_column][i] is not None:
                inputs.append(examples[src_column][i])
                targets.append(examples[tgt_column][i])

        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset, eval_dataset, predict_dataset = None, None, None
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            # train_dataset = train_dataset.select(range(data_args.max_train_samples))
            rand_data_idx = random.sample(list(range(len(train_dataset))), data_args.max_train_samples)
            train_dataset = train_dataset.select(rand_data_idx)
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    return train_dataset, eval_dataset, predict_dataset


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # log to both file and stderr
    fileHandler = logging.FileHandler("{}/train.log".format(training_args.output_dir))
    logger.addHandler(fileHandler)
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
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

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    #if 't5' not in model_args.model_name_or_path:
    if model_args.add_end_goal_token:
        tokenizer.add_tokens(['<end_goal>'])
        print('Added <end_goal> token')
    if model_args.add_na_token:
        tokenizer.add_tokens(['<end_goal>', '<n/a>'])
        print('Added <n/a> token.')
    print('Resizing model embedding to {}'.format(len(tokenizer)))
    model.resize_token_embeddings(len(tokenizer))
    #else:
    #    raise NotImplementedError

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )

    # load and preprocess the datasets
    train_dataset, eval_dataset, predict_dataset = load_and_preprocess_datasets(model_args, data_args, training_args, tokenizer, model)

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForMKPSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        mask_tokens_before_na=model_args.mask_mkp_before_na,
        mask_mkp_tokens=model_args.mask_mkp_tokens, 
        end_goal_token_id=-100 if not model_args.add_end_goal_token else tokenizer.encode('<end_goal>', add_special_tokens=False)[0],
        na_token_id=-100 if not model_args.add_na_token else tokenizer.encode('<n/a>', add_special_tokens=False)[0]
    )
    
    # Metric
    def postprocess_text_metakp(preds, labels):
        print(list(zip(preds, labels))[:5])
        preds = [[x.strip() for x in pred.strip().split('<end_goal>')[-1].split(';')] for pred in preds]
        labels = [[x.strip() for x in label.strip().split('<end_goal>')[-1].split(';')] for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text_metakp(decoded_preds, decoded_labels)

        # batch-wise token-level f1 with stemming
        
        # for cur_label in set([l for case in decoded_labels for l in case]):
        #     kp2scores[cur_label] = {"tp": 0, "fp": 0, "fn": 0, "label_count": 0, "pred_count": 0, "precision": 0, "recall": 0, "f1": 0}
        overall_p, overall_r = [], []
        for cur_labels, cur_preds in zip(decoded_labels, decoded_preds):
            kp2scores = {}
            
            cur_preds = set(cur_preds)
            cur_labels = set(cur_labels)
            cur_preds_tokens = set([stemmer.stem(x) for p in cur_preds for x in p.split()])
            cur_labels_tokens = set([stemmer.stem(x) for p in cur_labels for x in p.split()])
            for cur_label in cur_labels_tokens:
                if cur_label not in kp2scores:
                    kp2scores[cur_label] = {"precision_match": 0, "recall_match": 0, "label_count": 0, "pred_count": 0}
                kp2scores[cur_label]["label_count"] += 1
                if cur_label in cur_preds_tokens:
                    kp2scores[cur_label]["recall_match"] += 1
            for cur_pred in cur_preds_tokens:
                if cur_pred not in kp2scores:
                    kp2scores[cur_pred] = {"precision_match": 0, "recall_match": 0, "label_count": 0, "pred_count": 0}
                kp2scores[cur_pred]["pred_count"] += 1
                if cur_label in cur_labels_tokens:
                    kp2scores[cur_label]["precision_match"] += 1
                    
            for cur_label in kp2scores.keys():
                kp2scores[cur_label]['precision'] = (kp2scores[cur_label]["precision_match"] / kp2scores[cur_label]["pred_count"]) if kp2scores[cur_label]["pred_count"] != 0 else 0
                kp2scores[cur_label]['recall'] = (kp2scores[cur_label]["recall_match"] / kp2scores[cur_label]["label_count"]) if kp2scores[cur_label]["label_count"] != 0 else 0 
                # kp2scores[cur_label]['f1'] = ((kp2scores[cur_label]['precision'] + kp2scores[cur_label]['recall']) / 2 * kp2scores[cur_label]['recall'] * kp2scores[cur_label]['precision']) if kp2scores[cur_label]['recall'] * kp2scores[cur_label]['precision'] != 0 else 0

            overall_p += [x['precision'] for x in kp2scores.values() if x['pred_count'] > 0]
            overall_r += [x['recall'] for x in kp2scores.values() if x['label_count'] > 0]
            
        overall_p = np.mean(overall_p) if len(overall_p) > 0 else 0
        overall_r = np.mean(overall_r) if len(overall_r) > 0 else 0
        result = {'f1': (overall_p+overall_r)/2*overall_p*overall_r if overall_r*overall_p != 0 else 0}
        
        return result

    # Initialize our Trainer
    if data_args.valid_without_contraint: 
        trainer_class = Seq2SeqTrainer
    else:
        trainer_class = MKPSeq2SeqTrainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        callbacks=[TensorBoardCallback()],
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


    return results


if __name__ == "__main__":
    main()
