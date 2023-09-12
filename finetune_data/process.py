import json
from collections import defaultdict
import gzip
import random
from tqdm import tqdm
import argparse
import os

class LabelField:
    def __init__(self):
        self.label2id = dict()
        self.label_num = 0

    def get_id(self, label):
        
        if label in self.label2id:
            return self.label2id[label]
        
        self.label2id[label] = self.label_num
        self.label_num += 1

        return self.label2id[label]


parser = argparse.ArgumentParser()
    
parser.add_argument('--file_path', default='Industrial_and_Scientific_5.json.gz', help='Processing file path (.gz file).')
parser.add_argument('--meta_file_path', default='meta_Industrial_and_Scientific.json.gz', help='Processing file path (.gz file).')
parser.add_argument('--output_path', default='Scientific', help='Output directory')
args = parser.parse_args()


def extract_meta_data(path):
    meta_data = dict()
    with gzip.open(path) as f:
        for line in tqdm(f):
            line = json.loads(line)
            attr_dict = dict()
            asin = line['asin']
            category = ' '.join(line['category'])
            brand = line['brand']
            title = line['title']

            attr_dict['title'] = title
            attr_dict['brand'] = brand
            attr_dict['category'] = category
            meta_data[asin] = attr_dict

    return meta_data


meta_dict = extract_meta_data(args.meta_file_path)       


output_path = args.output_path
if not os.path.exists(output_path):
    os.mkdir(output_path)

input_file = args.file_path
train_file = os.path.join(output_path, 'train.json')
dev_file = os.path.join(output_path, 'val.json')
test_file = os.path.join(output_path, 'test.json')
umap_file = os.path.join(output_path, 'umap.json')
smap_file = os.path.join(output_path, 'smap.json')
meta_file = os.path.join(output_path, 'meta_data.json')

user_field = LabelField()
s_field = LabelField()
sequences = defaultdict(list)
raw_sequences = defaultdict(list)

gin = gzip.open(input_file, 'rb')

for line in tqdm(gin):

    line = json.loads(line)

    user_id = line['reviewerID']
    item_id = line['asin']
    time = line['unixReviewTime']

    if item_id in meta_dict:
    
        raw_sequences[user_id].append((item_id, time))

for k, v in raw_sequences.items():
    if len(v)>3:
        rand = random.randint(0, 4)
        if rand == 0:
            sequences[user_field.get_id(k)] = [(s_field.get_id(ele[0]), ele[1]) for ele in v]

train_dict = dict()
dev_dict = dict()
test_dict = dict()

intersections = 0

for k, v in tqdm(sequences.items()):
    sequences[k] = sorted(v, key=lambda x: x[1])
    sequences[k] = [ele[0] for ele in sequences[k]]

    length = len(sequences[k])
    intersections += length
    if length<3:
        train_dict[k] = sequences[k]
    else:
        train_dict[k] = sequences[k][:length-2]
        dev_dict[k] = [sequences[k][length-2]]
        test_dict[k] = [sequences[k][length-1]]

print(f'Users: {len(user_field.label2id)}, Items: {len(s_field.label2id)}, Intersects: {intersections}')

f_u = open(umap_file, 'w', encoding='utf8')
json.dump(user_field.label2id, f_u)
f_u.close()

f_s = open(smap_file, 'w', encoding='utf8')
json.dump(s_field.label2id, f_s)
f_s.close()

train_f = open(train_file, 'w', encoding='utf8')
json.dump(train_dict, train_f)
train_f.close()

dev_f = open(dev_file, 'w', encoding='utf8')
json.dump(dev_dict, dev_f)
dev_f.close()

test_f = open(test_file, 'w', encoding='utf8')
json.dump(test_dict, test_f)
test_f.close()

meta_f = open(meta_file, 'w', encoding='utf8')
json.dump(meta_dict, meta_f)
meta_f.close()

