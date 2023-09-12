from typing import Optional, Union, List, Dict, Tuple
from dataclasses import dataclass
from recformer import RecformerTokenizer
import torch
import unicodedata
import random


# Data collator
@dataclass
class PretrainDataCollatorWithPadding:

    tokenizer: RecformerTokenizer
    tokenized_items: Dict
    mlm_probability: float

    def __call__(self, batch_item_ids: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        '''
        features: A batch of list of item ids
        1. sample training pairs
        2. convert item ids to item features
        3. mask tokens for mlm

        input_ids: (batch_size, seq_len)
        item_position_ids: (batch_size, seq_len)
        token_type_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        global_attention_mask: (batch_size, seq_len)
        '''
        
        batch_item_seq_a, batch_item_seq_b = self.sample_pairs(batch_item_ids)
        batch_feature_a = self.extract_features(batch_item_seq_a)
        batch_feature_b = self.extract_features(batch_item_seq_b)

        batch_encode_features_a = self.encode_features(batch_feature_a)
        batch_encode_features_b = self.encode_features(batch_feature_b)
        batch_a = self.tokenizer.padding(batch_encode_features_a, pad_to_max=False)
        batch_b = self.tokenizer.padding(batch_encode_features_b, pad_to_max=False)

        batch_a["mlm_input_ids"], batch_a["mlm_labels"] = self.mask_mlm(batch_encode_features_a)
        batch_b["mlm_input_ids"], batch_b["mlm_labels"] = self.mask_mlm(batch_encode_features_b)

        batch = dict()

        for k, v in batch_a.items():
            batch[k+'_a'] = torch.LongTensor(v)
        
        for k, v in batch_b.items():
            batch[k+'_b'] = torch.LongTensor(v)

        return batch

    def sample_pairs(self, batch_item_ids):

        batch_item_seq_a = []
        batch_item_seq_b = []

        for item_ids in batch_item_ids:

            item_ids = item_ids['items']
            item_seq_len = len(item_ids)
            start = (item_seq_len-1) // 2
            target_pos = random.randint(start, item_seq_len-1)
            batch_item_seq_a.append(item_ids[:target_pos])
            batch_item_seq_b.append([item_ids[target_pos]])

        return batch_item_seq_a, batch_item_seq_b


    def extract_features(self, batch_item_seq):

        features = []

        for item_seq in batch_item_seq:
            feature_seq = []
            for item in item_seq:
                input_ids, token_type_ids = self.tokenized_items[item]
                feature_seq.append([input_ids, token_type_ids])
            features.append(feature_seq)

        return features

    def encode_features(self, batch_feature):
        
        features = []
        for feature in batch_feature:
            features.append(self.tokenizer.encode(feature, encode_item=False))

        return features

    def mask_mlm(self, flat_features):

        input_ids = [e["input_ids"] for e in flat_features]

        batch_input = self._collate_batch(input_ids)

        mask_labels = []
        for e in flat_features:
            ref_tokens = []
            for id in e["input_ids"]:
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)

            mask_labels.append(self._whole_word_mask(ref_tokens))

        batch_mask = self._collate_batch(mask_labels)
        inputs, labels = self.mask_tokens(batch_input, batch_mask)

        return inputs, labels

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):

        cand_indexes = []

        for (i, token) in enumerate(input_tokens):

            if token == self.tokenizer.bos_token or token == self.tokenizer.eos_token:
                continue

            if self._is_subword(token) and len(cand_indexes) > 0:
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        assert len(covered_indexes) == len(masked_lms)
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

    def _is_subword(self, token: str):
        if (
            not self.tokenizer.convert_tokens_to_string(token).startswith(" ")
            and not self._is_punctuation(token[0])
        ):
            return True
        
        return False

    @staticmethod
    def _is_punctuation(char: str):
        # obtained from:
        # https://github.com/huggingface/transformers/blob/5f25a5f367497278bf19c9994569db43f96d5278/transformers/tokenization_bert.py#L489
        cp = ord(char)
        if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False


    def mask_tokens(self, inputs: torch.Tensor, mask_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


    def _collate_batch(self, examples, pad_to_multiple_of: Optional[int] = None):
        """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
        # Tensorize if necessary.
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]

        # Check if padding is necessary.
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
            return torch.stack(examples, dim=0)

        # If yes, check if we have a `pad_token`.
        if self.tokenizer._pad_token is None:
            raise ValueError(
                "You are attempting to pad samples but the tokenizer you are using"
                f" ({self.tokenizer.__class__.__name__}) does not have a pad token."
            )

        # Creating the full tensor and filling it with our data.
        max_length = max(x.size(0) for x in examples)
        if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
        result = examples[0].new_full([len(examples), max_length], self.tokenizer.pad_token_id)
        for i, example in enumerate(examples):
            if self.tokenizer.padding_side == "right":
                result[i, : example.shape[0]] = example
            else:
                result[i, -example.shape[0] :] = example
        return result


@dataclass
class FinetuneDataCollatorWithPadding:

    tokenizer: RecformerTokenizer
    tokenized_items: Dict

    def __call__(self, batch_item_ids: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        '''
        features: A batch of list of item ids
        1. sample training pairs
        2. convert item ids to item features
        3. mask tokens for mlm

        input_ids: (batch_size, seq_len)
        item_position_ids: (batch_size, seq_len)
        token_type_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        global_attention_mask: (batch_size, seq_len)
        '''
        
        batch_item_seq, labels = self.sample_train_data(batch_item_ids)
        batch_feature = self.extract_features(batch_item_seq)
        batch_encode_features = self.encode_features(batch_feature)
        batch = self.tokenizer.padding(batch_encode_features, pad_to_max=False)
        batch["labels"] = labels

        for k, v in batch.items():
            batch[k] = torch.LongTensor(v)
        
        return batch

    def sample_train_data(self, batch_item_ids):

        batch_item_seq = []
        labels = []

        for item_ids in batch_item_ids:

            item_ids = item_ids['items']
            item_seq_len = len(item_ids)
            start = min(item_seq_len, 0)
            target_pos = random.randint(start, item_seq_len-1)
            batch_item_seq.append(item_ids[:target_pos])
            labels.append(item_ids[target_pos])

        return batch_item_seq, labels


    def extract_features(self, batch_item_seq):

        features = []

        for item_seq in batch_item_seq:
            feature_seq = []
            for item in item_seq:
                input_ids, token_type_ids = self.tokenized_items[item]
                feature_seq.append([input_ids, token_type_ids])
            features.append(feature_seq)

        return features

    def encode_features(self, batch_feature):
        
        features = []
        for feature in batch_feature:
            features.append(self.tokenizer.encode(feature, encode_item=False))

        return features


@dataclass
class EvalDataCollatorWithPadding:

    tokenizer: RecformerTokenizer
    tokenized_items: Dict

    def __call__(self, batch_data: List[Dict[str, Union[int, List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        '''
        features: A batch of list of item ids
        1. sample training pairs
        2. convert item ids to item features
        3. mask tokens for mlm

        input_ids: (batch_size, seq_len)
        item_position_ids: (batch_size, seq_len)
        token_type_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        global_attention_mask: (batch_size, seq_len)
        '''
        
        batch_item_seq, labels = self.prepare_eval_data(batch_data)
        batch_feature = self.extract_features(batch_item_seq)
        batch_encode_features = self.encode_features(batch_feature)
        batch = self.tokenizer.padding(batch_encode_features, pad_to_max=False)

        for k, v in batch.items():
            batch[k] = torch.LongTensor(v)

        labels = torch.LongTensor(labels)
        
        return batch, labels

    def prepare_eval_data(self, batch_data):

        batch_item_seq = []
        labels = []

        for data_line in batch_data:

            item_ids = data_line['items']
            label = data_line['label']
            
            batch_item_seq.append(item_ids)
            labels.append(label)

        return batch_item_seq, labels


    def extract_features(self, batch_item_seq):

        features = []

        for item_seq in batch_item_seq:
            feature_seq = []
            for item in item_seq:
                input_ids, token_type_ids = self.tokenized_items[item]
                feature_seq.append([input_ids, token_type_ids])
            features.append(feature_seq)

        return features

    def encode_features(self, batch_feature):
        
        features = []
        for feature in batch_feature:
            features.append(self.tokenizer.encode(feature, encode_item=False))

        return features