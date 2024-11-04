import openai
import backoff
import json
import random
import sys
import pandas as pd
from tqdm import tqdm
import os
import nltk
import torch
import tiktoken
from transformers import AutoTokenizer
import argparse
import numpy as np

home_dir = os.path.realpath('../..')


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on keyphrase generation")
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="The file of the dataset to use.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="The file of the output.",
    )
    parser.add_argument(
        "--num_token",
        type=int,
        default=5,
        help="Number of tokens to keep.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The model used to query.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="The name of the dataset to query.",
    )
    parser.add_argument(
        "--maxtoken",
        type=int,
        default=30,
        help="Maximum number of tokens allowed to generate.",
    )


    args = parser.parse_args()

    return args



def main():
    args = parse_args()

    data_file = args.data_file
    OUT_PATH = args.out_path
    model = args.model
    dataset = args.dataset
    NUM_TOKEN = args.num_token
    max_token = args.maxtoken

    assert '/' not in model
    backend = 'openai' if 'gpt' in model else 'vllm'

    print('Using backend "{}" for rejection.'.format(backend))
    
    llm = sampling_params = None
    if backend == 'openai':
        def init_openai():
            openai.organization = ### fill with your organization
            openai.api_key = ### fill with your api key
        init_openai()
        encoding = tiktoken.encoding_for_model(model)
    else:
        raise NotImplementedError


    @backoff.on_exception(backoff.expo, (openai.error.RateLimitError,
                                    openai.error.ServiceUnavailableError,
                                    openai.error.APIError))
    def chat_completions_with_backoff(**kwargs):
        return openai.ChatCompletion.create(**kwargs)


    # build data
    data = []
    with open(data_file) as f:
        for line in f.readlines():
            entry = json.loads(line)
            title = entry['title']
            abstract = entry['document']
            goal = entry['goal']
            keyphrases = entry['keyphrases']
            num_tokens = len(encoding.encode(abstract) + encoding.encode(title) + encoding.encode(goal))
            if num_tokens > NUM_TOKEN:
                new_abstract_encoding = encoding.encode(abstract)[:NUM_TOKEN]
                abstract = encoding.decode(new_abstract_encoding)
            data.append({'title': title, 'abstract': abstract, 'goal': goal, 'keyphrases': keyphrases})

    
    token_tot = 0
    all_outs = []
    summary_cache = {}
    for entry in tqdm(data, desc="Querying"):
        
        # prompt 1

        if dataset == 'duc2001':
            prompt = 'Generate present and absent keyphrases belonging to the high-level category from the given text, separated by commas. Do not output anything else.'
            prompt += '\nDocument Abstract: {}\nHigh-level category: {}\nKeyphrases (Must be of category \'{}\'):'.format(entry['abstract'], entry['goal'], entry['goal'])
        else:
            prompt = 'Generate present and absent keyphrases belonging to the high-level category from the given text, separated by commas. Do not output anything else.'
            prompt += '\nDocument Title: {}\nDocument Abstract: {}\nHigh-level category: {}\nKeyphrases (Must be of category \'{}\'):'.format(entry['title'], entry['abstract'], entry['goal'], entry['goal'])

        if backend == 'openai':

            cur_prompt = {
                "model": model, 
                "messages": [{"role": "system", 
                            "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}],
                "n": 1,
                "temperature": 1,
                "max_tokens": max_token
            }
            response = chat_completions_with_backoff(**cur_prompt)

            if response['choices'][0]['finish_reason'] == "length":
                all_outs.append(','.join(response['choices'][0]['message']['content'].strip().split(',')[:-1]))
            else:
                all_outs.append(response['choices'][0]['message']['content'].strip())


    
    with open(OUT_PATH, "w") as file:
        for line in all_outs:
            print(json.dumps({'output': line}), file=file)

            
if __name__ == '__main__':
    main()