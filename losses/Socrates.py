#####################################################################################
#           Socrates Loss: Unifying Confidence Calibration and Classification by Leveraging the Unknown
#           by Sandra Gómez-Gálvez, Tobias Olenyi, Gillian Dobbie, Katerina Taskova
#           Transactions on Machine Learning Research 2026
#           https://openreview.net/forum?id=DONqw1KhHq
#           Github user: sandruskyi
# License: CC BY 4.0.
####################################################################################
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

class SocratesLoss():
    def __init__(self, num_examples=50000, num_classes=10, mom=0.9, gamma: float = 0 , alpha: float = 1, dynamic: bool = True, loss_name: str = "Socrates", version : int = 1, version_SAT_original : bool = True, version_FOCALinGT : bool = True, version_FOCALinSAT : bool = True, version_changingWithIdk : bool = True, old: int = 0):
        """
        Description:
            To replicate Socrates values:
                --old 0 --version 1 --pretrain 0 --dynamic --version_SAT_original --version_FOCALinGT --version_FOCALinSAT --version_changingWithIdk
                Rest of hyperparameter tuning check Appendix F - Model Reproducibility: --arch --sat-momentum --gamma-focal-loss --alpha-focal-loss
            Socrates Loss:
                mitigates the stability-performance trade-off in ad-hoc confidence calibration through explicit uncertainty
                modeling via an unknown class. Predictions from this class are incorporated into the loss function and a
                dynamic uncertainty penalty, penalizing the model for failing to recognize its own uncertainty.
                The loss additionally emphasizes hard-to-classify instances and dynamically guides training using previous predictions.
            loss = target(1 - pred_gt)^(gamma)*log(pred_gt)+alpha_dynamic*(1-target)*(1 - pred_gt)^(gamma)*log(pred_idk)
            1º) Target is calculated with previous prediction after some epochs. For the first epochs target = gt, after some epochs, target = (mom*pred_pred + (1-mom)*pred_gt).
                If version_SAT_original = True, it calculates the prob as in the SAT version taking only into account the number of classes without the idk class
            2º) The target is taking into account completely for the gt part, for the idk part is taking into account (1-target): If the last prediction was not too good, gives more attention to the idk part
            3º) The gt part is smoothed with the focal part. If version_FOCALinSAT = True, also the idk part is smoothed.
            4º) If version_changingWithIdk = True, the idk part take more or less attention depending on if the model knows that it doesn´t know. How?
                    in version = 1 : ## PAPER VERSION!!
                        with alpha dynamic = ((pred_biggest not included in pred_gt) - pred_idk)
                    in version = 2 :
                        with alpha dynamic = ((pred_biggest not included in pred_gt and pred_idk) - pred_idk)
                    in version = 3:
                        with alpha dynamic = if pred_idk <=(pred_biggest not included in pred_gt and pred_idk) => alpha = 0.25, else alpha = 0
                If false, alpha_dynamic = 1.

        Args:
            num_examples: int. Number of instances
            num_classes: int. Number of classes.
            mom: float. Momentum factor. It controls the weight of last predictions. Bigger -> less weight. Default: 0.9.

            gamma: float. Gamma of the focal part. It is the modularity factor of the focal loss, it adjusts the rate at
                which the loss decreases as the pred. prob. increases
            alpha: float. Alpha of the focal part. It is the weighting factor, it gives more importance to the minority class
            dynamic: bool. If false, alpha_dynamic = 0.5, if true it calculates the alpha weighting factor in a dynamic way,
                alpha_dynamic =  ((pred_mayor without pred_gt) - pred_idk). It gives 0 weight when the alg errs but knows
                that it doesn't know (more prob in the idk class), and gives more weight when the alg doesn't know that
                it doesn't know (the prob in the idk class is not the highest apart of the gt class).
            loss_name: str. loss name
            version: int. version for the alpha_dynamic or for ablation study
            version_SAT_original : bool. If version_SAT_original = True, it calculates the prob as in the original version taking only into account the number of classes without the idk class
            version_FOCALinGT : bool. The gt component is smoothed with the focal part.
            version_FOCALinSAT : bool. The unknown component is smoothed with the focal part.
            version_changingWithIdk : bool. If version_changingWithIdk = True, the idk part take more or less attention depending on if the model knows that it doesn´t know.
            old : int. To explore different loss possibilities. Leave it at 0.
        """

        self.loss_name = loss_name

        # SELF ADAPTATIVE TRAINING
        self.prob_history = torch.zeros(num_examples, num_classes)
        self.updated = torch.zeros(num_examples, dtype=torch.int)
        self.mom = mom
        self.num_classes = num_classes
        self.version_SAT_original = version_SAT_original
        # FOCAL PART
        self.gamma_focal = gamma
        self.alpha_focal = alpha
        # IDK PART
        self.dynamic = dynamic
        self.alpha_dynamic = 0.25 # SOCRATES PAPER: DYNAMIC UNCERTAINTY PENALTY - BETA HYPERPARAMETER
        self.version = version
        self.version_FOCALinGT = version_FOCALinGT
        self.version_FOCALinSAT = version_FOCALinSAT
        self.version_changingWithIdk = version_changingWithIdk

        # old versions:
        self.old = old


    def _update_prob(self, prob, index, y):
        onehot = torch.zeros_like(prob)
        onehot[torch.arange(y.shape[0]), y] = 1

        self.prob_history = self.prob_history.to(index.device)
        prob_history = self.prob_history[index].clone()

        # if not inited, use onehot label to initialize runnning vector
        self.updated = self.updated.to(index.device)
        cond = (self.updated[index] == 1).to(index.device).unsqueeze(-1).expand_as(prob)
        prob_mom = torch.where(cond, prob_history, onehot) # IF the condition is true uses the elements of prob_history, if it is false of onehot

        # momentum update
        prob_mom = self.mom * prob_mom + (1 - self.mom) * prob

        self.updated[index] = 1
        self.prob_history[index] = prob_mom.to(self.prob_history.device)

        return prob_mom

    def __call__(self, logits: Tensor, y: Tensor, index: int, debug: bool = False, extract_values_paper: bool = False):
        if debug:
            print("logits.shape", logits.shape, "logits", logits)
            print("y.shape", y.shape, "y", y)

        if self.version_SAT_original:
            prob = F.softmax(logits.detach()[:, :self.num_classes], dim=1) # Original SAT
        else:
            prob = F.softmax(logits.detach()[:, :], dim=1)

        # SAT
        if debug:
            print("Before update: prob.shape", prob.shape, "prob", prob)

        prob = self._update_prob(prob, index, y)

        if debug:
            print("After update: prob.shape", prob.shape, "prob", prob)

        soft_label = torch.zeros_like(logits)
        soft_label[torch.arange(y.shape[0]), y] = prob[torch.arange(y.shape[0]), y]
        soft_label[:, -1] = 1 - prob[torch.arange(y.shape[0]), y]


        if debug: print("After adaptive target: soft_label.shape", soft_label.shape, "soft_label", soft_label)

        # Adding focal part for the gt position:
        probabilities = F.softmax(logits[:, :], dim=1)
        if self.version_FOCALinGT:
            soft_label[torch.arange(y.shape[0]), y] = soft_label[torch.arange(y.shape[0]), y] * self.alpha_focal * (
                    1 - probabilities[torch.arange(y.shape[0]), y]) ** self.gamma_focal
        if self.version_FOCALinSAT:
            soft_label[torch.arange(y.shape[0]), -1] = soft_label[torch.arange(y.shape[0]), -1] * self.alpha_focal * (
                1 - probabilities[torch.arange(y.shape[0]), y]) ** self.gamma_focal


        if debug: print("After Focal: soft_label.shape", soft_label.shape, "soft_label", soft_label)

        # IDK PART:
        if self.version_changingWithIdk:
            if self.dynamic:
                if debug: print("Calculating alpha dynamic:")
                # Create a mask to remove positions specified by y
                mask = torch.ones_like(probabilities)  # , dtype=torch.bool)
                mask[range(len(y)), y] = 0  # We remove the pred for the gt position
                if self.version == 2 or self.version == 3:
                    mask[range(len(y)), -1] = 0  # We remove the pred for the idk position

                if debug: print("mask.shape", mask.shape, "mask", mask)

                # Apply the mask to probabilities
                filtered_probabilities = probabilities * mask
                if debug: print("filtered_probabilities.shape", filtered_probabilities.shape, "filtered_probabilities",
                                filtered_probabilities)

                # Take the max values of each instance of filtered_probabilities:
                max_values, inds = torch.max(filtered_probabilities, dim=1)
                if debug:
                    print("max_values.shape", max_values.shape, "max_values", max_values)
                    print("inds.shape", inds.shape, "inds", inds)

                if self.version == 3:
                    # If predidk <= predbiggest not included in predgt or predidk then 0.25, else 0
                    self.alpha_dynamic = torch.where(
                        probabilities[torch.arange(y.shape[0]), -1] <= max_values[torch.arange(y.shape[0])], 0.25, 0.0)
                else:  # v1 or v2
                    # We finally calculate the alpha dynamic for the idk class:
                    self.alpha_dynamic = (
                                max_values[torch.arange(y.shape[0])] - probabilities[torch.arange(y.shape[0]), -1])
                if debug: print("self.alpha_dynamic.shape", self.alpha_dynamic.shape, "self.alpha_dynamic",
                                self.alpha_dynamic)

            # Self.alpha_dynamic will be the initializated value if the last if it's not runned
            soft_label[torch.arange(y.shape[0]), -1] = soft_label[torch.arange(y.shape[0]), -1] * self.alpha_dynamic


            if debug: print("soft_label.shape", soft_label.shape, "soft_label", soft_label)


        loss = torch.sum(-F.log_softmax(logits, dim=1) * soft_label, dim=1)
        if extract_values_paper:
            if self.version_changingWithIdk:
                values_alpha_dynamic, counts_alpha_dynamic = torch.unique(self.alpha_dynamic, return_counts = True)
                mode_alpha_dynamic = values_alpha_dynamic[torch.argmax(counts_alpha_dynamic)]
            else:
                values_alpha_dynamic, counts_alpha_dynamic, mode_alpha_dynamic = 0, 0, 0

            values_adaptive_target, counts_adaptive_target = torch.unique(prob[torch.arange(y.shape[0]), y], return_counts=True)
            mode_adaptive_target = values_adaptive_target[torch.argmax(counts_adaptive_target)]
            return (torch.mean(loss),
                    torch.mean(self.alpha_dynamic), torch.std(self.alpha_dynamic), mode_alpha_dynamic,
                    torch.mean(prob[torch.arange(y.shape[0]), y]), torch.std(prob[torch.arange(y.shape[0]), y]), mode_adaptive_target)   # loss, beta_penalty, adaptive_target [MEAN, STF, MODE]
        else:
            return torch.mean(loss)