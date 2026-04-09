"""
@User: sandruskyi
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

class CrossEntropyLoss():
    def __init__(self):
        self.crit = nn.CrossEntropyLoss()

    def __call__(self, logits, targets):
        loss =  self.crit(logits, targets)
        return loss