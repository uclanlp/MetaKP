import json
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--prediction_path", type=str)
    parser.add_argument("--output_path", type=str)
    
    return parser.parse_args()


def main(args):

    all_true_labels = []
    all_predictions = []
    meta_goals = {}
    with open(args.data_path, 'r') as file1, open(args.prediction_path, 'r') as file2:
        preds = file2.readlines()
        for i, line1 in enumerate(file1):
            
            ref = json.loads(line1)
            goal = ref['goal']
 
            try:
                pred = json.loads(preds[i])['output'].strip()
                if pred.lower() != "yes":
                    pred = "<n/a>"
            except ValueError as e:
                pred = preds[i].split('<end_goal>')[-1].strip()
                if pred.lower() == "no":
                    pred = "<n/a>"

            
            if ref['keyphrases'] != ["<n/a>"]:
                # positive example
                all_true_labels.append(0)
                if pred != "<n/a>":
                    all_predictions.append(0)
                else:
                    all_predictions.append(1)
            elif ref['keyphrases'] == ["<n/a>"]:
                # negative example
                all_true_labels.append(1)
                if pred == "<n/a>":
                    all_predictions.append(1)
                else:
                    all_predictions.append(0)
            

            if goal not in meta_goals:
                meta_goals[goal] = ([all_true_labels[-1]], [all_predictions[-1]])
            else:
                meta_goals[goal][0].append(all_true_labels[-1])
                meta_goals[goal][1].append(all_predictions[-1])

    """
    Calculate abstain stats
    This code uses the notation for AbstainQA as introduced in the following paper:
    Feng, S., Shi, W., Wang, Y., Ding, W., Balachandran, V., & Tsvetkov, Y. (2024). 
    Don't Hallucinate, Abstain: Identifying LLM Knowledge Gaps via Multi-LLM Collaboration. 
    In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (pp. 14664-14690). 
    Association for Computational Linguistics. 
    URL: https://aclanthology.org/2024.acl-long.786
    """

    all_aa = []
    all_ap = []
    all_ar = []
    all_af1 = []
    for goal in meta_goals:

        A = 0
        B = 0
        C = 0
        D = 0
        for label, pred in zip(meta_goals[goal][0], meta_goals[goal][-1]):
            if label == 0:
                if pred == 0:
                    A += 1
                else:
                    B += 1
            elif label == 1:
                if pred == 0:
                    C += 1
                else:
                    D += 1

        abstain_accuracy = (A + D) / (A + B + C + D)
        precision = D / (B + D) if (B + D) else np.nan
        recall = D / (C + D) if (C + D) else np.nan
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else np.nan

        all_aa.append(abstain_accuracy)
        all_ap.append(precision)
        all_ar.append(recall)
        all_af1.append(f1)


    print("ABSTAIN SCORE avraged by goal")
    print(f'Abstain Accuracy: {np.average([acc for acc in all_aa ]):.4f}')
    print(f'Precision: {np.average([p for p in all_ap if not np.isnan(p)]):.4f}')
    print(f'Recall: {np.average([r for r in all_ar if not np.isnan(r)]):.4f}')
    print(f'F1 Score: {np.average([f1 for f1 in all_af1 if not np.isnan(f1)]):.4f}')

    

    A = 0
    B = 0
    C = 0
    D = 0
    for label, pred in zip(all_true_labels, all_predictions):
        if label == 0:
            if pred == 0:
                A += 1
            else:
                B += 1
        elif label == 1:
            if pred == 0:
                C += 1
            else:
                D += 1

    abstain_accuracy = (A + D) / (A + B + C + D)
    precision = D / (B + D) if (B + D) else np.nan
    recall = D / (C + D) if (C + D) else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else np.nan

    print("\nABSTAIN SCORE in all")
    print(f'Abstain Accuracy: {abstain_accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

if __name__ == '__main__':
    args = parse_args()
    main(args)

