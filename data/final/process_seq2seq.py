import os
import json
from tqdm import tqdm


# dataset = 'duc2001'   # duc2001 pubmed
# splits = ['test']

dataset = 'kpbiomed'     # kptimes kpbiomed
# splits =  ['train' , 'valid', 'test']

os.makedirs(f'{dataset}/json_seq2seq', exist_ok=True)

for split in splits:
    annotation_file = '{}/{}.humanvalid_processed_release.json'.format(dataset, split)
    target_file = '{}/json_seq2seq/{}.json'.format(dataset, split)
    print(target_file)
    
        
    with open(annotation_file) as annotation_f, open(target_file, 'w') as tgt_f:    
        for line in tqdm(annotation_f.readlines()):
            entry = json.loads(line)
            out_entry = {}
            out_entry['src'] = entry['title'] + ' [sep] ' + entry['document'] 
            out_entry['tgt'] = entry['goal'] + '<end_goal>' + ' ; '.join(entry['keyphrases'])
            print(json.dumps(out_entry), file=tgt_f)
