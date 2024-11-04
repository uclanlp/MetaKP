import json
import argparse
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from nltk.stem.porter import PorterStemmer
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--prediction_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--lower_case", type=bool, default=True)
    parser.add_argument("--truncate", type=bool, default=False)
    parser.add_argument(
        "--top_k",
        type=int,
        default=1,
        help="Truncate to top k number of keyphrases.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Weight to rerank keyphrases.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold to rerank sampled llm keyphrases.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=10,
        help="The number of samples used to rerank sampled llm keyphrases.",
    )

    
    return parser.parse_args()


stemmer = PorterStemmer()
def stem_phrase(x):
    return ' '.join([stemmer.stem(y) for y in x.split()])


def sem_matching(model, preds, preds_stemmed, labels, labels_stemmed, similarity_threshold=0,
                 truncate_preds_to_O=False):
    if truncate_preds_to_O:
        preds = preds[:len(labels)]
    n_labels, n_preds = len(labels), len(preds)
    input_tokens = labels + preds
    
    phrase_embed_mean_pool = model.encode(input_tokens)
    label_embeds = phrase_embed_mean_pool[:n_labels]
    pred_embeds = phrase_embed_mean_pool[n_labels:]

    ###########################################################
    # 1 - calculate semantic precision scores for each phrase
    ###########################################################
    if n_labels == 0 or n_preds == 0:
        cur_p = 0
        cur_cg, cur_dcg, cur_ndcg = 0, 0, 0
    else:
        all_cos_sim = util.cos_sim(pred_embeds, label_embeds)
        top_sim_values, top_sim_indices = torch.topk(all_cos_sim, min(3, n_labels))
        match_results = {}
        rel_values = []
        for pred_i in range(n_preds):
            cur_pred, cur_pred_stemmed = preds[pred_i], preds_stemmed[pred_i]
            
            cur_support_label = None
            # if exact match, then give a score of 1
            if cur_pred_stemmed in labels_stemmed:
                cur_pred_score = 1
            elif top_sim_values[pred_i][0] > similarity_threshold:
                cur_pred_score = top_sim_values[pred_i][0].item()
                cur_support_label = labels[top_sim_indices[pred_i][0]]
            else:
                cur_pred_score = 0
            match_results[cur_pred] = [cur_pred_score, cur_support_label]
            rel_values.append(cur_pred_score)# / np.log2(pred_i + 1))
        cur_p = np.mean([x[0] for x in match_results.values()]).item()

        # ranking metrics
        cur_cg = np.mean(rel_values).item()
        cur_dcg = np.mean([x / np.log2(j + 2)
                            for j, x in enumerate(rel_values)]).item()
        ideal_rel_values = sorted(rel_values, reverse=True)
        cur_idcg = np.mean([x / np.log2(j + 2)
                            for j, x in enumerate(ideal_rel_values)]).item()
        cur_ndcg = cur_dcg / cur_idcg if cur_idcg != 0 else 0

    ###########################################################
    # 2 - calculate semantic recall scores for each phrase
    ###########################################################
    if n_labels == 0 or n_preds == 0:
        cur_r = 0
    else:
        all_cos_sim = util.cos_sim(label_embeds, pred_embeds)
        top_sim_values, top_sim_indices = torch.topk(all_cos_sim, min(3, n_preds))
        match_results = {}
        for label_i in range(n_labels):
            cur_label, cur_label_stemmed = labels[label_i], labels_stemmed[label_i]
            
            cur_support_pred = None
            # if exact match, then give a score of 1
            if labels_stemmed in preds_stemmed:
                cur_label_score = 1
            elif top_sim_values[label_i][0] > similarity_threshold:
                cur_label_score = top_sim_values[label_i][0].item()
                cur_support_pred = preds[top_sim_indices[label_i][0]]
            else:
                cur_label_score = 0
            match_results[cur_label] = [cur_label_score, cur_support_pred]
        cur_r = np.mean([x[0] for x in match_results.values()]).item()

    cur_f1 = 0 if cur_p * cur_r == 0 else 2 * cur_p * cur_r / (cur_p + cur_r)
    
    return {'p': cur_p, 'r': cur_r, 'f1': cur_f1, 'cg': cur_cg, 'dcg': cur_dcg, 'ndcg': cur_ndcg}
        

def main(args):
    data = []
    with open(args.data_path) as f:
        for line in f.readlines():
            entry = json.loads(line)
            data.append({
                'id': entry['id'],
                'goal': entry['goal'],
                'target': entry['keyphrases']
            })
    
    preds = []
    with open(args.prediction_path) as f:
        for line in f.readlines():
            if "sample" in args.prediction_path:
                """ Preprocess for self-consistency sampling methods.
                After extraction, ach phrase is then scored using an equation
                that is the product of frequency and inverse mean of rank.
                Rank of a keyphrase is its position in a sample, i.e. the earlier / smaller the better.
                """
                if args.threshold is None:
                    print("Error: --threshold is required")
                    sys.exit(1)  

                line = json.loads(line)['output'][:args.sample_size]
                scores = {}
                for sample in line:
                    sample = sample.strip().lower()
                    sample = ','.join(sample.split('\n')).strip().lstrip('-').strip()
                    keyphrases = []
                    for x in sample.strip().split(','):
                        if len(x.split(':')) == 2:
                            if 'absent' in x.split(':')[-1].strip() or 'present' in x.split(':')[-1].strip():
                                x = x.split(':')[0].strip().lstrip('-').strip()
                            elif 'absent' in x.split(':')[0].strip() or 'present' in x.split(':')[0].strip():
                                x = x.split(':')[-1].strip().lstrip('-').strip()
                        else:
                            x = x.split(':')[-1].strip().lstrip('-').strip()
                        if x != '' and x != "absent" and x != "present" and x != "absent keyphrases" and x != "present keyphrases" and x != "absent keyphrase":
                            keyphrases.append(x)
                            

                    for i, kp in enumerate(keyphrases):
                        if kp in scores:
                            scores[kp].append(i+1)
                        else:
                            scores[kp] = [i+1]

                y = sorted([(k, len(v), np.mean(v), (len(v) / args.sample_size) / np.mean(v)) for k, v in scores.items()], key=lambda x: x[-1], reverse=True)

                print(y)
                z = [x for x in y if x[-1] >= args.threshold]
                if z:
                    print(z)
                    preds.append([item[0] for item in z])
                else:
                    if len(y) == 0:
                        preds.append([''])
                        print(line)
                    else:
                        print(y[0])
                        preds.append([y[0][0]])
                print()        
            else:
                try:
                    line = json.loads(line)['output'].strip()
                    print(line)
                    line = ','.join(line.split('\n'))
                    if args.truncate:
                        preds.append([x.split(':')[-1].strip() for x in line.strip().split(',')if x.split(':')[-1].strip() is not ''][:args.top_k])
                    else:
                        preds.append([x.split(':')[-1].strip() for x in line.strip().split(',')if x.split(':')[-1].strip() is not ''])
                    print(preds[-1])
                    
                except ValueError as e:
                    line = line.split('<end_goal>')[-1]
                    if args.truncate:
                        preds.append([x.strip() for x in line.strip().split(';')][:args.top_k])
                    else:
                        preds.append([x.strip() for x in line.strip().split(';')])


    if args.lower_case:
        print('Note: all the preds and refs have been lower-cased.')
        preds_lower = []
        for kps in preds:
            preds_lower.append([x.lower() for x in kps])
        for i in range(len(data)):
            data[i]['target'] = [x.lower() for x in data[i]['target']]
            
    assert len(data) == len(preds)
    
    model = SentenceTransformer('uclanlp/keyphrase-mpnet-v1')

    detailed_results = []
    for cur_entry, cur_preds in tqdm(zip(data, preds), desc='evaluating'):
        # omit for those <n/a>
        if cur_entry['target'] == ["<n/a>"]:
            continue 

        # step
        cur_refs = cur_entry['target']
        cur_refs_stemmed = [stem_phrase(x) for x in cur_refs]
        cur_preds_stemmed = [stem_phrase(x) for x in cur_preds] 
        
        # pass to sem_matching
        scores_M = sem_matching(model, cur_preds, cur_preds_stemmed, cur_refs, cur_refs_stemmed, truncate_preds_to_O=False)
        scores_O = sem_matching(model, cur_preds, cur_preds_stemmed, cur_refs, cur_refs_stemmed, truncate_preds_to_O=True)
        detailed_results.append({
            'id': cur_entry['id'],
            'refs': cur_refs,
            'preds': cur_preds,
            'scores_M': scores_M,
            'scores_O': scores_O
        })
        # print(cur_preds, cur_preds_stemmed, cur_refs, cur_refs_stemmed, scores)
        
    # report cluster-level success rate & document-level success rate
    with open(args.output_path, 'w') as f:
        for entry in detailed_results:
            print(json.dumps(entry), file=f)


    print("Average # kps/metakp:", np.mean([len(x['preds']) for x in detailed_results]))
    
    print('Cluster-level results @M ({} clusters):'.format(len(detailed_results)))
    for k in detailed_results[0]['scores_M'].keys():
        print('\t{}: {:.04f}'.format(k, np.mean([x['scores_M'][k] for x in detailed_results])), end=' ')
    print()
    
    print('Cluster-level results @O ({} clusters):'.format(len(detailed_results)))
    for k in detailed_results[0]['scores_O'].keys():
        print('\t{}: {:.04f}'.format(k, np.mean([x['scores_O'][k] for x in detailed_results])), end=' ')
    print()

    docids = set([x['id'].split('mkp')[0] for x in detailed_results])
    print('Document-level results @M ({} docs):'.format(len(docids)))
    for k in detailed_results[0]['scores_M'].keys():
        cur_score_results = []
        for docid in docids:
            cur_score_results.append(np.mean([x['scores_M'][k] for x in detailed_results if docid in x['id']]))
        print('\t{}: {:.04f}'.format(k, np.mean(cur_score_results)), end=' ')
    print()
    
    print('Document-level results @O ({} docs):'.format(len(docids)))
    for k in detailed_results[0]['scores_O'].keys():
        cur_score_results = []
        for docid in docids:
            cur_score_results.append(np.mean([x['scores_O'][k] for x in detailed_results if docid in x['id']]))
        print('\t{}: {:.04f}'.format(k, np.mean(cur_score_results)), end=' ')
    print()
    
    # print()
    print('Satisfaction rate @M with different thresholds:')
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    f1_scores = [x['scores_M']['f1'] for x in detailed_results]
    for threshold in thresholds:
        satisfactory = [x >= threshold for x in f1_scores]
        satisfaction_score = sum(satisfactory) / len(satisfactory) if len(satisfactory) > 0 else 0
        print('\t{}: {:.04f}'.format(threshold, satisfaction_score), end=' ')
    print()
        
    print('Satisfaction rate @O with different thresholds:')
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    f1_scores = [x['scores_O']['f1'] for x in detailed_results]
    for threshold in thresholds:
        satisfactory = [x >= threshold for x in f1_scores]
        satisfaction_score = sum(satisfactory) / len(satisfactory) if len(satisfactory) > 0 else 0
        print('\t{}: {:.04f}'.format(threshold, satisfaction_score), end=' ')
    print()
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
