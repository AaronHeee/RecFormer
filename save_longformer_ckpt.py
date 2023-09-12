import torch
import os
from transformers import LongformerForMaskedLM
from recformer import RecformerConfig, RecformerForPretraining

longformer = LongformerForMaskedLM.from_pretrained('allenai/longformer-base-4096')

config = RecformerConfig.from_pretrained('allenai/longformer-base-4096')
config.max_attr_num = 3
config.max_attr_length = 32
config.max_item_embeddings = 51
config.attention_window = [64] * 12
model = RecformerForPretraining(config)

longformer_state_dict = longformer.state_dict()
recformer_state_dict = model.state_dict()
for name, param in longformer_state_dict.items():
    if name not in recformer_state_dict:
        print('missing name', name)
        continue
    else:
        try:
            if not recformer_state_dict[name].size()==param.size():
                print(name)
                print(recformer_state_dict[name].size())
                print(param.size())
            recformer_state_dict[name].copy_(param)
        except:
            print('wrong size', name)

for name, param in longformer_state_dict.items():
    if not torch.all(param == recformer_state_dict[name]):
        print(name)

if not os.path.exists('longformer_ckpt'):
    os.mkdir('longformer_ckpt')
torch.save(recformer_state_dict, 'longformer_ckpt/longformer-base-4096.bin')
