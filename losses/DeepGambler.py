"""
    Code from https://github.com/LayneH/SAT-selective-cls/blob/main/loss.py
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

class DeepGamblerLoss():

    def __call__(self, outputs, targets, reward):
        outputs = F.softmax(outputs, dim=1)
        outputs, reservation = outputs[:,:-1], outputs[:,-1]
        # gain = torch.gather(outputs, dim=1, index=targets.unsqueeze(1)).squeeze()
        gain = outputs[torch.arange(targets.shape[0]), targets]
        doubling_rate = (gain.add(reservation.div(reward))).log()
        return -doubling_rate.mean()
