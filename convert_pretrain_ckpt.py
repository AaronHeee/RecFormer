import torch
from collections import OrderedDict
from recformer import RecformerModel, RecformerConfig, RecformerForSeqRec

PRETRAINED_CKPT_PATH = 'pretrain_ckpt/pytorch_model.bin'
LONGFORMER_CKPT_PATH = 'longformer_ckpt/longformer-base-4096.bin'
LONGFORMER_TYPE = 'allenai/longformer-base-4096'
RECFORMER_OUTPUT_PATH = 'pretrain_ckpt/recformer_pretrain_ckpt.bin'
RECFORMERSEQREC_OUTPUT_PATH = 'pretrain_ckpt/seqrec_pretrain_ckpt.bin'

input_file = PRETRAINED_CKPT_PATH
state_dict = torch.load(input_file)

longformer_file = LONGFORMER_CKPT_PATH
longformer_state_dict = torch.load(longformer_file)

state_dict['_forward_module.model.longformer.embeddings.word_embeddings.weight'] = longformer_state_dict['longformer.embeddings.word_embeddings.weight']

output_file = RECFORMER_OUTPUT_PATH
new_state_dict = OrderedDict()

for key, value in state_dict.items():

    if key.startswith('_forward_module.model.longformer.'):
        new_key = key[len('_forward_module.model.longformer.'):]
        new_state_dict[new_key] = value

config = RecformerConfig.from_pretrained(LONGFORMER_TYPE)
config.max_attr_num = 3
config.max_attr_length = 32
config.max_item_embeddings = 51
config.attention_window = [64] * 12
model = RecformerModel(config)
model.load_state_dict(new_state_dict)

print('Convert successfully.')
torch.save(new_state_dict, output_file)



output_file = RECFORMERSEQREC_OUTPUT_PATH
new_state_dict = OrderedDict()

for key, value in state_dict.items():

    if key.startswith('_forward_module.model.'):
        new_key = key[len('_forward_module.model.'):]
        new_state_dict[new_key] = value

config = RecformerConfig.from_pretrained(LONGFORMER_TYPE)
config.max_attr_num = 3
config.max_attr_length = 32
config.max_item_embeddings = 51
config.attention_window = [64] * 12
model = RecformerForSeqRec(config)

model.load_state_dict(new_state_dict, strict=False)

print('Convert successfully.')
torch.save(new_state_dict, output_file)
