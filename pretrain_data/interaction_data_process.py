from collections import defaultdict
import gzip
import json
from tqdm import tqdm
import os

SEQ_ROOT = '' # Set your seq data path

pretrain_categories = ['Automotive', 'Cell_Phones_and_Accessories', \
              'Clothing_Shoes_and_Jewelry', 'Electronics', 'Grocery_and_Gourmet_Food', 'Home_and_Kitchen', \
              'Movies_and_TV', 'CDs_and_Vinyl']

pretrain_seq_pathes = [f'{SEQ_ROOT}/{cate}_5.json.gz' for cate in pretrain_categories]

for path in pretrain_seq_pathes:
    if not os.path.exists(path):
        print(path)
        exit(0)

meta_data = json.load(open('meta_data.json'))

def extract_interaction(path, sequences, miss_asin_set):
    user_set, item_set, inter_num = set(), set(), 0
    with gzip.open(path) as f:
        category = path.split('/')[-1]
        for line in tqdm(f):
            line = json.loads(line)
            user = line['reviewerID']
            asin = line['asin']
            time = line['unixReviewTime']
            if asin in meta_data:
                sequences[user+'_'+category].append((time, asin))
                user_set.add(user)
                item_set.add(asin)
                inter_num += 1
            else:
                miss_asin_set.add(asin)

    print(f'Dataset: {category}, User: {len(user_set)}, Items: {len(item_set)}, Interaction numbers: {inter_num}')

    return sequences, miss_asin_set

def post_process(sequences):
    length = 0
    for user, sequence in tqdm(sequences.items()):
        sequences[user] = [ele[1] for ele in sorted(sequence)]
        length += len(sequences[user])

    print(f'Averaged length: {length/len(sequences)}')

    return sequences

training_sequences = defaultdict(list)
dev_sequences = defaultdict(list)
miss_asin_set = set()

# training set
for path in pretrain_seq_pathes[:-1]:
    training_sequences, miss_asin_set = extract_interaction(path, training_sequences, miss_asin_set)

# dev set
dev_sequences, miss_asin_set = extract_interaction(pretrain_seq_pathes[-1], dev_sequences, miss_asin_set)

training_sequences = post_process(training_sequences)
dev_sequences = post_process(dev_sequences)

print(f'Number of Training Sequences:{len(training_sequences)}, Validation Sequences: {len(dev_sequences)}, Missed asins: {len(miss_asin_set)}')

train_seq = list(training_sequences.values())
dev_seq = list(dev_sequences.values())

with open('train.json', 'w') as f:
    json.dump(train_seq, f)

with open('dev.json', 'w') as f:
    json.dump(dev_seq, f)
