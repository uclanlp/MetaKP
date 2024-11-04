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

home_dir = os.path.realpath('../..')


nltk.download('punkt_tab')


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
        "--num_sentence",
        type=int,
        default=5,
        help="Number of sentences to keep.",
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


    args = parser.parse_args()

    return args



def main():
   
    args = parse_args()

    data_file = args.data_file
    OUT_PATH = args.out_path
    num_sentence = args.num_sentence
    model = args.model
    dataset = args.dataset

    assert '/' not in model
    backend = 'openai'

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
            new_abstract = ' '.join(nltk.sent_tokenize(abstract)[:num_sentence])             
            data.append({'title': title, 'abstract': new_abstract, 'goal': goal})

    token_tot = 0
    all_outs = []
    for entry in tqdm(data, desc="Querying"):
        # prompt
        
        if dataset == 'duc2001':
            instruction = "In this task you will need to decide if you should reject the high-level category given the abstract of a document. One could use the high-level category to write keyphrases from the document. If you decide the category is relevant to the document, generate yes; if the category is not relevant, generate no. Do not output anything else.\n"
            prompt = 'Abstract: {}\nHigh-level category: {}\nRelevant? (yes or no):'.format(entry['abstract'], entry['goal'])
        else:
            instruction = "In this task you will need to decide if you should reject the high-level category given the title and abstract of a document. One could use the high-level category to write keyphrases from the document. If you decide the category is relevant to the document, generate yes; if the category is not relevant, generate no. Do not output anything else.\n"
            prompt = 'Title: {}\nAbstract: {}\nHigh-level category: {}\nRelevant? (yes or no):'.format(entry['title'], entry['abstract'], entry['goal'])
        
        if backend == 'openai':
            cur_prompt = {
                "model": model, 
                "messages": [{"role": "system", 
                            "content": instruction},
                            {"role": "user", "content": prompt}],
                "n": 1,
                "temperature": 0,
                "max_tokens": 10
            }
            response = chat_completions_with_backoff(**cur_prompt)
            token_tot += response['usage']['prompt_tokens']
            all_outs.append(response['choices'][0]['message']['content'].strip())

            
    
    with open(OUT_PATH, "w") as file:
        for line in all_outs:
            print(json.dumps({'output': line}), file=file)

            
if __name__ == '__main__':
     main()