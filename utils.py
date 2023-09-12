import json
import torch
import torch.nn as nn

MAX_VAL = 1e4

def read_json(path, as_int=False):
    with open(path, 'r') as f:
        raw = json.load(f)
        if as_int:
            data = dict((int(key), value) for (key, value) in raw.items())
        else:
            data = dict((key, value) for (key, value) in raw.items())
        del raw
        return data



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)

class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


class Ranker(nn.Module):
    def __init__(self, metrics_ks):
        super().__init__()
        self.ks = metrics_ks
        self.ce = nn.CrossEntropyLoss()
        
    def forward(self, scores, labels):
        labels = labels.squeeze()
        
        try:
            loss = self.ce(scores, labels).item()
        except:
            print(scores.size())
            print(labels.size())
            loss = 0.0
        
        predicts = scores[torch.arange(scores.size(0)), labels].unsqueeze(-1) # gather perdicted values
        
        valid_length = (scores > -MAX_VAL).sum(-1).float()
        rank = (predicts < scores).sum(-1).float()
        res = []
        for k in self.ks:
            indicator = (rank < k).float()
            res.append(
                ((1 / torch.log2(rank+2)) * indicator).mean().item() # ndcg@k
            ) 
            res.append(
                indicator.mean().item() # hr@k
            )
        res.append((1 / (rank+1)).mean().item()) # MRR
        res.append((1 - (rank/valid_length)).mean().item()) # AUC

        return res + [loss]