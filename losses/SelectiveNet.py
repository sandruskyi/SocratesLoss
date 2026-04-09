"""
@User: sandruskyi
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

class SelectiveNetLoss():
    """
    Currently this class is not working correctly
    """
    def __init__(self, num_classes = 10, lamda=32):
        self.num_classes = num_classes
        self.lamda = lamda

    def __call__(self, predic_and_selec_outputs, targets, coverage = 0.8):
        outputs_predic = predic_and_selec_outputs[:, :-1]
        outputs_select = predic_and_selec_outputs[:, -1]

        outputs_select = torch.where(outputs_select >= 0.5, torch.tensor(1), torch.tensor(0))
        selected_indices = (outputs_select>= 0.5).nonzero().squeeze(dim=1)

        outputs_predic_selected = outputs_predic[selected_indices]
        targets_selected = targets[selected_indices]

        if outputs_predic_selected.shape[0] == 0:
            loss = 50
            return loss

        empirical_coverage = sum(outputs_select)/outputs_select.shape[0]
        empirical_risk = (sum(F.cross_entropy(outputs_predic_selected, targets_selected, reduction='none'))/outputs_select.shape[0])/empirical_coverage

        coverage = coverage/100
        #penalty_function = self.lamda * torch.maximum(torch.tensor(0.0).to(predic_and_selec_outputs.device), torch.tensor(coverage - empirical_coverage).to(predic_and_selec_outputs.device))
        penalty_function = self.lamda * torch.maximum(torch.tensor(0.0).to(predic_and_selec_outputs.device), torch.tensor(coverage - empirical_coverage).to(predic_and_selec_outputs.device)).pow(2)
        loss = empirical_risk + penalty_function

        if str(loss) == 'nan':
            print("loss, empirical_risk, penalty_function", loss, empirical_risk, penalty_function)

        return loss