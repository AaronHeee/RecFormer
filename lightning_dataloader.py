from typing import List
from torch.utils.data import Dataset
from collator import PretrainDataCollatorWithPadding


class ClickDataset(Dataset):
    def __init__(self, dataset: List, collator: PretrainDataCollatorWithPadding):
        super().__init__()

        self.dataset = dataset
        self.collator = collator

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        
        return self.dataset[index]

    def collate_fn(self, data):

        return self.collator([{'items': line} for line in data])
