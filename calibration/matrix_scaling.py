"""
Code from: https://github.com/saurabhgarg1996/calibration/tree/master
"""

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F


def _histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1), np.arange(npt), np.sort(x))


def cross_entropy_loss(logits, labels):
    loss = -np.mean(logits[np.arange(len(labels)), labels])
    return loss


def add_softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def ece_loss(probs, labels, num_bins=10, equal_mass=False):
    predictions = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    accuracies = np.equal(predictions, labels)

    if not equal_mass:
        bins = np.linspace(0, 1, num_bins + 1)
    else:
        bins = _histedges_equalN(confidences, num_bins)

    ece = 0.0

    for i in range(num_bins):
        in_bin = np.greater_equal(confidences, bins[i]) & np.less(
            confidences, bins[i + 1]
        )
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(in_bin * accuracies)
            avg_confidence_in_bin = np.mean(in_bin * confidences)
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


# Adapted from https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py

class MatrixScaling:
    def __init__(
        self, num_label, bias=False, weights=None, device=None, print_verbose=False, args_run=None
    ):
        self.args_run = args_run
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cpu")
        if self.args_run.loss == 'Socrates':
            num_label+=1
        self.temperature_1 = nn.Parameter(torch.ones(num_label).to(self.device) * 1.0)
        self.temperature_2 = nn.Parameter(
            torch.zeros((num_label, num_label)).to(self.device)
        )

        self.bias = (
            nn.Parameter(torch.ones(num_label).to(self.device) * 0.0) if bias else None
        )
        self.biasFlag = bias
        self.print_verbose = print_verbose

        if weights is not None:
            self.weights = weights.to(device)
        else:
            self.weights = None

    def forward(self, input):
        return self.temperature_scale(input)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits

        if self.biasFlag:
            bias = self.bias.unsqueeze(0).expand(logits.size(0), -1)
            return (
                logits
                @ (torch.diag(torch.exp(self.temperature_1)) + self.temperature_2)
                + bias
            )
            # return logits @ torch.exp(self.temperature) + bias
        else:
            return logits @ (
                torch.diag(torch.exp(self.temperature_1)) + self.temperature_2
            )

    def fit(self, logits, labels, eps=1e-12):

        torch_labels = labels.long().to(self.device)
        torch_logits = logits.float().to(self.device)

        if self.weights is not None:
            nll_criterion = nn.CrossEntropyLoss(weight=self.weights)
        else:
            nll_criterion = nn.CrossEntropyLoss()

        # Next: optimize the temperature w.r.t. NLL
        if not self.biasFlag:
            optimizer = optim.LBFGS([self.temperature_1, self.temperature_2])

        else:
            optimizer = optim.LBFGS([self.temperature_1, self.temperature_2, self.bias])

        def eval():
            optimizer.zero_grad()
            # l2 = self.temperature_2.square().sum()
            loss = nll_criterion(
                self.temperature_scale(torch_logits), torch_labels
            )  # + l2*0.0001
            loss.backward()
            return loss

        loss = -10.0
        new_loss = -1.0

        while np.abs(loss - new_loss) > 1e-4:
            loss = new_loss
            if self.print_verbose:
                print(f"Loss : {loss}")

            optimizer.step(eval)

            with torch.no_grad():
                new_loss = (
                    nll_criterion(self.temperature_scale(torch_logits), torch_labels)
                    .cpu()
                    .numpy()
                )

        torch_logits = self.temperature_scale(torch_logits)
        rescaled_probs = F.softmax(torch_logits, dim=-1).detach().cpu().numpy()

    def calibrate(self, logits, eps=1e-12):
        # probs = np.clip(probs, eps, 1 - eps)
        # logits = np.log(probs)

        torch_logits = logits.float().to(self.device)
        torch_logits = self.temperature_scale(torch_logits).detach().cpu().numpy()
        # rescaled_probs = F.softmax(self.temperature_scale(torch_logits), dim=-1).detach().cpu().numpy()

        return torch_logits
