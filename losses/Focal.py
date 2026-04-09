"""
    User: sandruskyi
    Version of the code from https://github.com/torrvision/focal_calibration/blob/main/Losses/focal_loss.py
    Reference:
        [1]  T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, Focal loss for dense object detection.
                arXiv preprint arXiv:1708.02002, 2017.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F



class FocalLoss():
    def __init__(self, gamma=0 , alpha=1, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
    def __call__(self, logits, targets):
        """
        Description:
            loss = - alpha(1 - pred_gt)^(gamma)*log(pred_gt)
        Args:
            logits: Tensor with the logits (before apply the softmax) outputs of the NN
            targets: Tensor of labels, they are the position of the final class, example: [0 0 1 0 1]
        Returns:
            When an object of this class invokes this method it is executed as a function. The loss function is calculated.
        """
        targets = targets.view(-1,1)
        logpt = F.log_softmax(logits, dim=1)
        logpt = logpt.gather(1,targets) # we take the prediction related with the ground truth class

        logpt = logpt.view(-1)

        pt = logpt.exp()

        loss = -self.alpha * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()