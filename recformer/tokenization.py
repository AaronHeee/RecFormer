import torch
from transformers import LongformerTokenizer

class RecformerTokenizer(LongformerTokenizer):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config=None):
        cls.config = config
        return super().from_pretrained(pretrained_model_name_or_path)
        
    def __call__(self, items, pad_to_max=False, return_tensor=False):
        '''
        items: item sequence or a batch of item sequence, item sequence is a list of dict

        return:
        input_ids: token ids
        item_position_ids: the position of items
        token_type_ids: id for key or value
        attention_mask: local attention masks
        global_attention_mask: global attention masks for Longformer
        '''

        if len(items)>0 and isinstance(items[0], list): # batched items
            inputs = self.batch_encode(items, pad_to_max=pad_to_max)

        else:
            inputs = self.encode(items)

        if return_tensor:

            for k, v in inputs.items():
                inputs[k] = torch.LongTensor(v)

        return inputs

    def item_tokenize(self, text):
        return self.convert_tokens_to_ids(self.tokenize(text))

    def encode_item(self, item):

        input_ids = []
        token_type_ids = []
        item = list(item.items())[:self.config.max_attr_num]  # truncate attribute number

        for attribute in item:

            attr_name, attr_value = attribute

            name_tokens = self.item_tokenize(attr_name)
            value_tokens = self.item_tokenize(attr_value)

            attr_tokens = name_tokens + value_tokens
            attr_tokens = attr_tokens[:self.config.max_attr_length]

            input_ids += attr_tokens
            
            attr_type_ids = [1] * len(name_tokens)
            attr_type_ids += [2] * len(value_tokens)
            attr_type_ids = attr_type_ids[:self.config.max_attr_length]
            token_type_ids += attr_type_ids

        return input_ids, token_type_ids


    def encode(self, items, encode_item=True):
        '''
        Encode a sequence of items.
        the order of items:  [past...present]
        return: [present...past]
        '''
        items = items[::-1]  # reverse items order
        items = items[:self.config.max_item_embeddings - 1] # truncate the number of items, -1 for <s>

        input_ids = [self.bos_token_id]
        item_position_ids = [0]
        token_type_ids = [0]

        for item_idx, item in enumerate(items):

            if encode_item:
            
                item_input_ids, item_token_type_ids = self.encode_item(item)

            else:

                item_input_ids, item_token_type_ids = item


            input_ids += item_input_ids
            token_type_ids += item_token_type_ids

            item_position_ids += [item_idx+1] * len(item_input_ids) # item_idx + 1 make idx starts from 1 (0 for <s>)

        input_ids = input_ids[:self.config.max_token_num]
        item_position_ids = item_position_ids[:self.config.max_token_num]
        token_type_ids = token_type_ids[:self.config.max_token_num]

        attention_mask = [1] * len(input_ids)
        global_attention_mask = [0] * len(input_ids)
        global_attention_mask[0] = 1

        return {
            "input_ids": input_ids,
            "item_position_ids": item_position_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "global_attention_mask": global_attention_mask
        }

    def padding(self, item_batch, pad_to_max):

        if pad_to_max:
            max_length = self.config.max_token_num
        else:
            max_length = max([len(items["input_ids"]) for items in item_batch])
        

        batch_input_ids = []
        batch_item_position_ids = []
        batch_token_type_ids = []
        batch_attention_mask = []
        batch_global_attention_mask = []


        for items in item_batch:

            input_ids = items["input_ids"]
            item_position_ids = items["item_position_ids"]
            token_type_ids = items["token_type_ids"]
            attention_mask = items["attention_mask"]
            global_attention_mask = items["global_attention_mask"]

            length_to_pad = max_length - len(input_ids)

            input_ids += [self.pad_token_id] * length_to_pad
            item_position_ids += [self.config.max_item_embeddings - 1] * length_to_pad
            token_type_ids += [3] * length_to_pad
            attention_mask += [0] * length_to_pad
            global_attention_mask += [0] * length_to_pad

            batch_input_ids.append(input_ids)
            batch_item_position_ids.append(item_position_ids)
            batch_token_type_ids.append(token_type_ids)
            batch_attention_mask.append(attention_mask)
            batch_global_attention_mask.append(global_attention_mask)

        return {
            "input_ids": batch_input_ids,
            "item_position_ids": batch_item_position_ids,
            "token_type_ids": batch_token_type_ids,
            "attention_mask": batch_attention_mask,
            "global_attention_mask": batch_global_attention_mask
        }


    def batch_encode(self, item_batch, encode_item=True, pad_to_max=False):

        item_batch = [self.encode(items, encode_item) for items in item_batch]

        return self.padding(item_batch, pad_to_max)
        


if __name__ == "__main__":

    from models import RecformerConfig


    config = RecformerConfig.from_pretrained("allenai/longformer-base-4096")
    tokenizer = RecformerTokenizer.from_pretrained("allenai/longformer-base-4096", config=config)

    items1 = [{'pt': 'PUZZLES',
            'material': 'Cardboard++Cartón',
            'item_dimensions': '27 x 20 x 0.1 inches',
            'number_of_pieces': '1000',
            'brand': 'Galison++',
            'number_of_items': '1',
            'model_number': '9780735366763',
            'size': '1000++',
            'theme': 'Christmas++',
            'color': 'Dresden'},
            {'pt': 'DECORATIVE_SIGNAGE',
            'item_shape': 'Square++Cuadrado',
            'brand': 'Generic++',
            'color': 'Square-5++Cuadrado-5',
            'mounting_type': 'Wall Mount++',
            'material': 'Wood++Madera'}]
    items2 = [{'pt': 'WALL_ART',
            'number_of_items': '1',
            'mounting_type': 'Wall Mount++',
            'item_shape': 'Rectangular++',
            'brand': "Teacher's Discovery++",
            'color': '_++'},
            {'pt': 'CALENDAR',
            'theme': 'Funny, Love, Wedding++',
            'format': 'wall_calendar',
            'model_year': '2022',
            'brand': 'CALVENDO++',
            'size': 'Square++cuadrado',
            'material': 'Paper, Wool++'},
            {'pt': 'BLANK_BOOK',
            'number_of_items': '1',
            'color': 'Hanging Flowers++Flores colgantes',
            'brand': 'Graphique++',
            'ruling_type': 'Ruled++',
            'binding': 'office_product',
            'paper_size': '6.25 x 8.25 inches++',
            'style': 'Hanging Flowers'}]

    inputs = tokenizer(items1)
    print(inputs)
    print(tokenizer.convert_ids_to_tokens(inputs['input_ids']))
    print(len(inputs['input_ids']))

    inputs = tokenizer([items1, items2])
    print(inputs)
    

        
