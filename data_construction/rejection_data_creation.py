import json
import sys
import random
from sklearn import metrics
from sentence_transformers import SentenceTransformer, util
from torch.nn.functional import normalize
import numpy as np
from scipy.spatial.distance import cdist
from collections import Counter
import os

# Get the current working directory
current_directory = os.getcwd()
home_directory = os.path.dirname(current_directory)

# maintain a 1:1 pos and neg sample ratio
NUM_NEG_PER_POS = 1  
# impose a constraint to not allow each goal's appearance in negative samples 
# to exceed that in positive samples
REJECTION_STRATEGY = "constraint" 
INPUT_FILE_PATH = home_directory+"/data/final/kpbiomed/valid.humanvalid_processed_release.json"
TRAIN_FILE_PATH = home_directory+"/data/final/kpbiomed/train.humanvalid_processed_release.json"
OUT_FILE_PATH = home_directory+"/data/final_with_neg/kpbiomed/valid.rejection_augmented_____.json"


def load_dataset(file_path):
    dataset = []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            entry = json.loads(line)
            entry['docid_original'] = entry["id"].split("mkp", 1)[0]
            dataset.append(entry)
    return dataset

def generate_embeddings(data, model):
    embeddings = []
    # remove duplication
    metaconcepts_removeduplication = list(set([x['goal'] for x in data]))
    # filter out general concepts, e.g. domain, subject, ...
    exact_keywords = ["topic", "concept", "individual", "location", "event", "organization"]
    keywords = ["subject", "domain", "topic of interest", "concepts/topics", "concept & topic", "concepts/principles", "concept and principle", "knowledge", "study"]   
    filtered_meta = [metaconcept for metaconcept in metaconcepts_removeduplication if metaconcept not in exact_keywords]
    final_meta = [metaconcept for metaconcept in filtered_meta if not any(keyword in metaconcept for keyword in keywords)]
    
    embeddings = model.encode(final_meta, batch_size=32)

    full_embeddings = model.encode(metaconcepts_removeduplication, batch_size=32)
 
    return metaconcepts_removeduplication, full_embeddings, final_meta, embeddings



def generate_new_dataset(data, doc_to_distant_goals, doc_to_goals, sample_method):
    new_data = []
    added_docs = set()
    negative_counts = {mkp: 0 for mkp in mkp_count.keys()}
    negp = NUM_NEG_PER_POS

    for entry in data:
        new_entry = entry.copy()
        new_data.append(new_entry)

        if int(entry["id"].split("mkp", 1)[-1]) == len(doc_to_goals[entry['docid_original']]) and entry['docid_original'] not in added_docs:
            # Find a cluster that is different from the document's original cluster
            distant_goals = doc_to_distant_goals[entry['docid_original']]
        
            if len(distant_goals) == 0:
                print("This doesn't have distant clusters", entry)

            if len(distant_goals) != 0:
                distant_goals_freq_dict = {goal:mkp_count[goal] for goal in distant_goals}
                tot_freq = sum(list(distant_goals_freq_dict.values()))
                distant_goals_prob_dict = {goal:mkp_count[goal] / tot_freq for goal in distant_goals}
                num_selections = int(negp * len(doc_to_goals[entry['docid_original']])) 

                
                if sample_method == "frequency":
                    new_meta_concepts = np.random.choice(list(distant_goals_prob_dict.keys()), p=list(distant_goals_prob_dict.values()), size=num_selections, replace=False)
                elif sample_method == "random":
                    new_meta_concepts = np.random.choice(list(distant_goals_prob_dict.keys()), size=num_selections, replace=False)
                elif sample_method == "constraint":

                    # Filter keys
                    filtered_keys = [mkp for mkp in distant_goals_prob_dict if negative_counts[mkp] < mkp_count[mkp]]
                    if filtered_keys:
                        filtered_freq = [distant_goals_freq_dict[mkp] for mkp in filtered_keys]
                        # Normalize the filtered probabilities
                        filtered_probs = filtered_freq / np.sum(filtered_freq)
                        # Select new meta concepts based on the adjusted probabilities and the number of selections
                        new_meta_concepts = np.random.choice(filtered_keys, p=filtered_probs, size=min(num_selections, len(filtered_keys)), replace=False)
                        
                        if num_selections > len(filtered_keys):
                            remains = [mkp for mkp in distant_goals_prob_dict if mkp not in filtered_keys]
                            remains_freq = [distant_goals_freq_dict[mkp] for mkp in remains]
                            remains_probs = remains_freq / np.sum(remains_freq)
                            new_meta_concepts = np.append(new_meta_concepts, np.random.choice(remains, p=remains_probs, size=num_selections-len(filtered_keys), replace=False))
                           

                    else:
                        new_meta_concepts = np.random.choice(list(distant_goals_prob_dict.keys()), p=list(distant_goals_prob_dict.values()), size=num_selections, replace=False)
                    
                    for mkp in new_meta_concepts:
                        negative_counts[mkp] += 1
                        

                for i, mkp in enumerate(new_meta_concepts):
                    # Create a new entry
                    new_entry = entry.copy()
                    new_entry['id'] = entry['docid_original'] + 'mkp' + str(len(doc_to_goals[entry['docid_original']]) + i+1) 
                    new_entry['goal'] = mkp
                    new_entry['keyphrases'] = ["<n/a>"]
                    new_data.append(new_entry)
                    added_docs.add(entry['docid_original'])

    return new_data

if __name__ == '__main__':
    file_path = INPUT_FILE_PATH
    data = load_dataset(file_path)

    all_mkps = [x['goal'] for x in data]
    mkp_count = Counter(all_mkps)

    model = SentenceTransformer('uclanlp/keyphrase-mpnet-v1')
    metaconcepts, embeddings, potential_mataconcepts, potential_embeddings = generate_embeddings(data, model)
    mkp_to_embed = {mkp: embeddings[i] for i, mkp in enumerate(metaconcepts)}

    # Create a doc to goals mapping
    doc_to_goals = {}
    for i, entry in enumerate(data):
        if entry["docid_original"] not in doc_to_goals:
            doc_to_goals[entry["docid_original"]] = [mkp_to_embed[entry["goal"]]]
        else:
            doc_to_goals[entry["docid_original"]].append(mkp_to_embed[entry["goal"]])

    doc_to_distant_goals = {}
    for doc, embeds in doc_to_goals.items():
        cosine_sim = util.cos_sim(np.array(embeds), np.array(potential_embeddings))
        # calculate the average across rows
        means = cosine_sim.mean(dim=0)
        means_np = means.numpy()
        # find threshold
        threshold = np.percentile(means_np, 50)
        # identify goals in the lowest percentile
        column_indices = np.where(means_np <= threshold)[0]        
        doc_to_distant_goals[doc] = sorted([potential_mataconcepts[i] for i in column_indices], key=lambda x: mkp_count[x], reverse=True)

        
    new_dataset = generate_new_dataset(data, doc_to_distant_goals, doc_to_goals, REJECTION_STRATEGY)

    with open(OUT_FILE_PATH, 'w') as f:
        for entry in new_dataset:
            print(json.dumps(entry), file=f)  
