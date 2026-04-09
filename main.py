#####################################################################################
# Code is based on:
#           Huang, L., Zhang, C., & Zhang, H. (2020). Self-Adaptive Training: beyond
#           Empirical Risk Minimization. In H. Larochelle, M. Ranzato, R. Hadsell,
#           M.F. Balcan, & H. Lin (Eds.), Advances in Neural Information Processing
#           Systems (Vol. 33, pp. 19365-19376).
# from https://github.com/LayneH/SAT-selective-cls
#
# Code augmented for the paper
#           Socrates Loss: Unifying Confidence Calibration and Classification by Leveraging the Unknown
#           by Sandra Gómez-Gálvez, Tobias Olenyi, Gillian Dobbie, Katerina Taskova
#           Transactions on Machine Learning Research 2026
#           https://openreview.net/forum?id=DONqw1KhHq
#           Github user: sandruskyi
# License: CC BY 4.0.
####################################################################################

from __future__ import print_function
import argparse
import os
import shutil
import random
import tempfile

import pandas as pd
import numpy as np
import copy
import time
from tqdm import tqdm
import random
import math
from matplotlib import pyplot as plt
from PIL import Image



import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.nn as nn
import ray
from ray import tune
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
from pathlib import Path
from functools import partial



import models as models
from transformers import ViTForImageClassification # Transfer learning for CIFAR-100 VIT
from training_functions.preparing_dataset import prepare_dataset
from losses import DeepGambler, SelfAdaptiveTraining, CrossEntropy, Focal, FocalAdaptiveGamma, DECE, BrierScore, Socrates
from utils import Logger, mkdir_p, savefig, closefig, Bar, AverageMeter, accuracy
from utils.useful_functions import collate_model_info, plot_history
from utils.metric_monitor import MetricMonitor
from utils.risk_coverage_curve import risk_coverage_curve_plotly
from utils.bisection import bisection_method, bisection_method_LeoFeng

from utils.statistic_meter import StatisticsMeter
from sklearn.model_selection import StratifiedKFold, GridSearchCV

#import cProfile

#CSC loss
import moco.CSC
from sklearn.metrics.pairwise import cosine_similarity

# Calibration
from calibration.plot_reliability_diagram import calculate_confidence_accuracy_multiclass
from calibration.temperature_scaling import ModelWithTemperature
from calibration.fulldirichlet import FullDirichletCalibrator
from calibration.matrix_scaling import MatrixScaling
from calibration.vector_scaling import VectorScaling

model_names = ("vgg16", "vgg16_bn", "lcrn_normalization", "lcrn_normalization_new", "cifar10vgg_pytorch", "resnet34", "resnet50", "resnet110", "vit")


# If tensorboard code is enable: After running, to check tensorboard results go to: http://localhost:6006/

###############
## ARGUMENTS ##
###############
parser = argparse.ArgumentParser(description='Framework Calibrated Selective Classification')
## Data
parser.add_argument('-d', '--dataset', default='cifar10', type=str, choices=['cifar10', 'svhn', 'imagenet',
                                                                             'food101', 'cacophony', 'cacophony2',
                                                                             'cifar100', 'cifar10C', 'cifar100C',
                                                                             'inat', 'birds', 'celebA', 'celebAadam'])
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
## Training
parser.add_argument('-t', '--train', action='store_true',
                    help='train the model. When evaluate is true, training is ignored and trained models are loaded.')
parser.add_argument('--not-augmentation', action='store_false',
                    help=' if using it: do not use data augmentation to train the model.')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--early-stopping', action='store_true',
                    help='Early stopping. Default: False')
    # Optimizer
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate. 0.1 if complete training, 5e-4 (0.0005) if transfer learning')
parser.add_argument('--schedule', type=int, nargs='+', default=[25,50,75,100,125,150,175,200,225,250,275],
                        help='Multiply learning rate by gamma at the scheduled epochs (default: 25,50,75,100,125,150,175,'
                             '200,225,250,275). Use default if complete training, use [20, 35, 45] if transfer learning')
parser.add_argument('--gamma', type=float, default=0.5, help='LR is multiplied by gamma on schedule '
                                                             '(default: 0.5)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
    # Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16_bn',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' Architecture (default: vgg16_bn). Working options: vgg16_bn, resnet34, resnet110, vit.'
                        ' Please edit the code to train with other architectures')
    # VIT
parser.add_argument('--vit-patch-size', default=4, type=int, metavar='N',
                    help='ViT Patch Size (default: 4)')
parser.add_argument('--vit-max-len', default=100, type=int, metavar='N',
                    help='ViT Max Length (default: 100) All sequences must be less than 1000 including class token')
parser.add_argument('--vit-embed-dim', default=512, type=int, metavar='N',
                    help='ViT Embed Dim (default: 512)')
parser.add_argument('--vit-layers', default=12, type=int, metavar='N',
                    help='ViT Layers (default: 12)')
parser.add_argument('--vit-resnet-features-channels', default=64, type=int, metavar='N',
                    help='ViT Resnet Features Channels (default: 64)')
parser.add_argument('--vit-heads', default=16, type=int, metavar='N',
                    help='ViT Heads (default: 16)')
    # VIT TRANSFER LEARNING
parser.add_argument('--transfer-learning', action='store_true',
                    help='True if transfer learning with vision transformers for CIFAR-100')
    # Risk-Coverage
parser.add_argument('--coverage', type=float, nargs='+',default=[100.,99.,98.,97.,95.,90.,85.,80.,75.,70.,
                                                                 60.,50.,40.,30.,20.,10.],
                    help='the expected coverages used to evaluated the accuracies after abstention')
    # Loss
parser.add_argument('--loss', default='Socrates', type=str,
                    help='loss function: gambler, ce, sat, focal, focalAdaptive, dece, brierScore, Socrates, csc')

parser.add_argument('--pretrain', type=int, default=0,
                    help='Number of pretraining epochs for the loss function (two-step training). ')
parser.add_argument('--sat-momentum', default=0.9, type=float, help='Momentum factor for Socrates or momentum for sat')
parser.add_argument('--first-epochs-ce', action='store_true', help='If First epochs are training with CE. Default: False')
## DeepGambler Loss
parser.add_argument('-o', '--rewards', dest='rewards', type=float, nargs='+', default=[2.2],
                    metavar='o', help='For Deepgamblers. The reward o for a correct prediction; Abstention has a '
                                      'reward of 1. Provided parameters would be stored as a list for multiple runs.')
## Focal loss & Socrates loss:
parser.add_argument('--gamma-focal-loss', type=float, default=0,
                    help='Modularity factor for the Socrates Loss or Value of the gamma (modularity factor) for the focal loss.'
                         ' Default = 0')
parser.add_argument('--alpha-focal-loss', type=float, default=1,
                    help='Value of the alpha (weighting factor) for the focal loss.'
                         ' Default = 1')
## Socrates loss:
parser.add_argument('--dynamic', action='store_true',
                    help='Value of the dynamic (bool). True if modularity factor alpha for the idk part is dynamic. True for Socrates'
                         ' Default = False')
parser.add_argument('--version', type=int, default=1,
                    help='Value of the version (int). Version of Socrates: 1. Possible values: '
                         '{1: Dynamic Uncertainty penalty = ((pred_biggest without pred_gt) - pred_idk),'
                         '2: Dynamic Uncertainty penalty = ((pred_biggest without pred_gt and pred_idk) - pred_idk),'
                         '3: Dynamic Uncertainty penalty = if pred_idk <=(pred_biggest without pred_gt & pred_idk),'
                         'then alpha = 0.25, else alpha = 0, }'
                         ' Default = 1')
parser.add_argument('--old', type=int, default=0,
                    help=' To explore different Socrates loss possibilities. Socrates: 0.'
                         ' Default = 0')
parser.add_argument('--version_SAT_original', action='store_true',
                    help='Socrates loss: True. (bool).'
                         'If version_SAT_original = True, it calculates the prob'
                         'as in the original version taking only into account the'
                         'number of classes without the idk class'
                         ' Default = False')
parser.add_argument('--version_FOCALinGT', action='store_true',
                    help='Socrates loss True. (bool).'
                         'The gt part is smoothed with the focal part.'
                         'If version_FOCALinGT = True, also the idk part is smoothed.'
                         ' Default = False')
parser.add_argument('--version_FOCALinSAT', action='store_true',
                    help='Socrates loss True. (bool).'
                         'The gt part is smoothed with the focal part.'
                         'If version_FOCALinSAT = True, also the idk part is smoothed.'
                         ' Default = False')
parser.add_argument('--version_changingWithIdk', action='store_true',
                    help='Socrates loss True. (bool).'
                         'If version_changingWithIdk = True, the idk part take more or less attention'
                         'depending on if the model knows that it doesn´t know.'
                         ' Default = False')
parser.add_argument('--debug', action='store_true',
                    help='Debug for Socrates loss.'
                         ' Default = False')
parser.add_argument('--extract-values-paper', action='store_true',
                    help='Debug for Socrates loss.'
                         ' Default = False')
# CSC loss
#moco
parser.add_argument(
    "--moco-k",
    default=300,
    type=int,
    help="queue size; number of negative keys (default: 65536)",
)
parser.add_argument(
    "--moco-m",
    default=0.999,
    type=float,
    help="moco momentum of updating key encoder (default: 0.999)",
)
parser.add_argument(
    "--moco-t", default=0.07, type=float, help="softmax temperature (default: 0.07)"
)
parser.add_argument(
    "--moco-dim", default=512, type=int, help="Moco dim (default: 512)"
)

# Temperature Scaling
parser.add_argument('--temperatureScaling', action='store_true',
                    help='Temperature Scaling. Default: False')
# Dirichlet post-hoc calibration method
parser.add_argument('--dirichlet', action='store_true',
                    help='Dirichlet post-hoc. Default: False')
# Matrix Scaling post-hoc calibration method
parser.add_argument('--matrixScaling', action='store_true',
                    help='Matrix Scaling. Default: False')
# Vector Scaling post-hoc calibration method
parser.add_argument('--vectorScaling', action='store_true',
                    help='Vector Scaling. Default: False')
## New Approach: Idk class -> Predator
parser.add_argument('--idkforpredator', action='store_true',
                    help='Idk class as Predator')
## Save
parser.add_argument('-s', '--save', default='save', type=str, metavar='PATH',
                    help='path to save checkpoint (default: save)')
## Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate trained models on validation set, following the paths defined by "save", "arch" and "rewards"')
parser.add_argument('--hypersearchSplit', action='store_true',
                    help='For big datasets. It split the training dataset in 80% train and 20% validation for the hyperparameter search.')
parser.add_argument('--ray_tune', action='store_true',
                    help='Uses Ray Tune for the hyperparameter search.')
parser.add_argument('--eval-path', default="", type=str,
                    help='path of the .pth (Default search the .pth, default: "" )')
parser.add_argument('--cluster', dest='cluster', action='store_true',
                    help='If we are using the cluster to train')
parser.add_argument('--clusterLessPower', dest='clusterLessPower', action='store_true',
                    help='If we are using the cluster to train')
## Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
num_classes=10 # this is modified later in main() when defining the specific datasets

## Use CUDA
if not args.clusterLessPower:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

print("Using CUDA?", use_cuda)

## CLUSTER
if args.cluster:
    mp.set_sharing_strategy('file_system')
    if args.clusterLessPower:
        primary_device = list(map(int, args.gpu_id.split(',')))[0]
        device = torch.device(f'cuda:{primary_device}')
    else:
        device = torch.device(f'cuda:{int(args.gpu_id)}' if use_cuda else 'cpu')

def train(trainloader, model, criterion, optimizer, epoch, use_cuda, reward, args, writer, num_classes,
          loss_idx_value=0, model_k=None, archive=None):
    # switch to train mode
    model.train()

    # CSC loss
    if args.loss == 'csc':
        model_k.train()
        if args.arch == 'vgg16_bn':
            feature_extractor = model.features
        if epoch == args.pretrain:
            for param_q, param_k in zip(model.parameters(), model_k.parameters()):
                param_k.data = param_q.data
        if epoch > args.pretrain:
            for param_q, param_k in zip(model.parameters(), model_k.parameters()):
                param_k.data = param_k.data * args.moco_m + param_q.data * (1.0 - args.moco_m)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses2 = AverageMeter()  # csc
    NLLlosses = AverageMeter()
    NLLlosses_correct = AverageMeter()
    NLLlosses_incorrect = AverageMeter()
    avgIdkConfidences = AverageMeter()
    l2norms = AverageMeter()
    beta_penalties_mode = StatisticsMeter()
    beta_penalties_mean = StatisticsMeter()
    beta_penalties_std = StatisticsMeter()
    adaptive_targets_mode = StatisticsMeter()
    adaptive_targets_mean = StatisticsMeter()
    adaptive_targets_std = StatisticsMeter()
    top1 = AverageMeter()  # Accuracy top 1
    top5 = AverageMeter()  # Accuracy top 5
    moco_top1 = AverageMeter()
    train_dataset_calculate_calibration_with_idk = {'y_gt': [], 'y_train_class_predicted': [],
                                                    'y_train_confidences_predicted_argmax': []}

    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    current_reward = reward
    for batch_idx, batch_data in tqdm(enumerate(trainloader), total=len(trainloader)):

        inputs, targets, indices = batch_data
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            if args.clusterLessPower:
                primary_device = list(map(int, args.gpu_id.split(',')))[0]
                inputs, targets, indices = inputs.to(f'cuda:{primary_device}'), targets.to(
                    f'cuda:{primary_device}'), indices.to(f'cuda:{primary_device}')
            else:
                inputs, targets, indices = inputs.cuda(), targets.cuda(), indices.cuda()

        targets = targets.to(torch.int64)  # For pytorch version > 0.4.0 this is not neccesary

        # Tensorboard
        if args.dataset != "cacophony" and batch_idx == 0:
            writer.add_image("Training - Example image input at batch 0 of each epoch ", inputs[0], global_step=epoch)
            writer.add_scalar("Training - Example image label at batch 0 of each epoch ", targets[0], global_step=epoch)

        # We calculate the outputs:
        if args.arch == "vit":
            if args.transfer_learning:
                outputs_logits_tl = model(inputs)
                outputs_logits = outputs_logits_tl.logits
            else:
                outputs_logits, _ = model(inputs)
        else:
            outputs_logits = model(inputs)

        if args.loss == 'csc':
            with torch.no_grad():
                if args.arch == "vit":
                    if args.transfer_learning:
                        outputs_logits_tl = model_k(inputs)
                        outputs_k = outputs_logits_tl.logits
                    else:
                        outputs_k, _ = model_k(inputs)
                else:
                    outputs_k = model_k(inputs)
            if args.arch == 'vgg16_bn':
                outputs_feature = feature_extractor(inputs)
                outputs_feature = outputs_feature.view(outputs_feature.size(0), -1)
                outputs_projection = nn.Sequential(*list(model.classifier.children())[:3])(outputs_feature)
                outputs_logits = nn.Sequential(*list(model.classifier.children())[3:])(outputs_projection)

        global full_k1
        global full_k2

        beta_penalty_mean, beta_penalty_std, beta_penalty_mode, adaptive_target_mean, adaptive_target_std, adaptive_target_mode = torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0)
        if epoch >= args.pretrain and (
                args.loss == 'csc' or args.loss == 'csc_entropy' or args.loss == 'csc_sat_entropy') and full_k1 and full_k2:
            if args.arch != 'vgg16_bn':
                temp_full_k1, temp_full_k2, moco_error, loss2 = archive(
                    torch.flatten(hidden_features, 1),
                    torch.flatten(hidden_features_k, 1),
                    targets, outputs_logits, outputs_k, epoch + 1,
                    args.pretrain, full_k1 and full_k2)
            else:
                temp_full_k1, temp_full_k2, moco_error, loss2 = archive(outputs_projection,
                                                                        hidden_features_k,
                                                                        targets, outputs_logits, outputs_k, epoch + 1,
                                                                        args.pretrain, full_k1 and full_k2)
            full_k1 = full_k1 or temp_full_k1
            full_k2 = full_k2 or temp_full_k2

            if args.loss == 'csc':
                if full_k1 and full_k2:
                    losses2.update(loss2.item(), inputs.size(0))
                    moco_top1.update(moco_error, inputs.size(0))
                    loss = criterion(outputs_logits, targets) + loss2 * current_reward
                else:
                    loss = F.cross_entropy(outputs_logits, targets)
            elif args.loss == 'csc_entropy':
                if full_k1 and full_k2:
                    softmax = nn.Softmax(-1)
                    losses2.update(loss2.item(), inputs.size(0))
                    moco_top1.update(moco_error, inputs.size(0))
                    loss = criterion(outputs_logits, targets) + loss2 * current_reward + (
                            args.entropy * (-softmax(outputs_logits) * outputs_logits).sum(-1)).mean()
                else:
                    loss = F.cross_entropy(outputs_logits, targets)
        elif epoch >= args.pretrain and (
                args.loss != 'csc' and args.loss != 'csc_entropy' and args.loss != 'csc_sat_entropy'):
            if args.loss == 'gambler':
                loss = criterion(outputs_logits, targets, reward)
            elif args.loss == 'sat':
                loss = criterion(outputs_logits, targets, indices)
            elif args.loss == 'dece':
                loss_ce = F.cross_entropy(outputs_logits, targets)
                loss_dece = criterion(outputs_logits, targets)
                loss = loss_ce + 0.5 * loss_dece
            elif args.loss == 'focal' or args.loss == 'focalAdaptive':
                loss = criterion(outputs_logits, targets)
            elif args.loss == 'Socrates':
                if args.extract_values_paper:
                    loss, beta_penalty_mean, beta_penalty_std, beta_penalty_mode, adaptive_target_mean, adaptive_target_std, adaptive_target_mode = criterion(outputs_logits, targets, indices, debug=args.debug, extract_values_paper=args.extract_values_paper)
                else:
                    loss = criterion(outputs_logits, targets, indices, debug=args.debug)
            else:
                loss = criterion(outputs_logits, targets)
        else:  # ce, mc, brier, FLSD, Focal and Socrates (es=0) must be always pretrain = 0
            if args.loss == 'csc' or 'csc_entropy' or 'csc_sat_entropy':
                moco_error = 1 / num_classes
                if args.arch != 'vgg16_bn' and epoch >= args.pretrain:
                    temp_full_k1, temp_full_k2, moco_error = archive(
                        torch.flatten(hidden_features, 1),
                        torch.flatten(hidden_features_k, 1), targets,
                        outputs_logits, outputs_k, epoch + 1, args.pretrain,
                        full_k1 and full_k2)
                else:
                    if epoch >= args.pretrain:
                        temp_full_k1, temp_full_k2, moco_error = archive(outputs_projection,
                                                                         hidden_features_k,
                                                                         targets,
                                                                         outputs_logits, outputs_k, epoch + 1,
                                                                         args.pretrain,
                                                                         full_k1 and full_k2)
                    else:
                        temp_full_k1, temp_full_k2 = False, False  # csc Initialization
                full_k1 = full_k1 or temp_full_k1
                full_k2 = full_k2 or temp_full_k2
                loss = F.cross_entropy(outputs_logits, targets)
                moco_top1.update(moco_error, inputs.size(0))
            elif args.loss == 'ce':
                loss = F.cross_entropy(outputs_logits, targets)
            elif args.loss == 'focal' or args.loss == 'focalAdaptive' or args.loss == 'brierScore':   # Put focal with pretrain 0
                loss = criterion(outputs_logits, targets)
            elif args.loss == 'dece':
                loss_ce = F.cross_entropy(outputs_logits, targets)
                loss_dece = criterion(outputs_logits, targets)
                loss = loss_ce + 0.5 * loss_dece
            elif args.loss == 'Socrates' and not args.first_epochs_ce:
                adaptive_target_mean = torch.mean(targets)
                adaptive_target_std = torch.std(targets)
                adaptive_target_mode = torch.mode(targets)
                criterion2 = Focal.FocalLoss(gamma=args.gamma_focal_loss, alpha=args.alpha_focal_loss,
                                             size_average=True)
                loss = criterion2(outputs_logits[:, :-1], targets)
            else:  # SAT and others
                adaptive_target_mean = torch.mean(targets)
                adaptive_target_std = torch.std(targets)
                adaptive_target_mode = torch.mode(targets)
                loss = F.cross_entropy(outputs_logits[:, :-1], targets)  # SAT: As in the initial epochs we take into account only the gt (that is 1) as t, it is the same than only to do the cross_entropy, because for the idk class prediction (1-t)=(1-1)=0

        # We calculate the rest of values for future graph evaluation
        with torch.no_grad():
            NLL_loss = F.cross_entropy(outputs_logits, targets)
            confidences_outputs_with_idk = F.softmax(outputs_logits, dim=1).cpu().detach().numpy()
            class_predicted_with_idk = np.argmax(confidences_outputs_with_idk,
                                                 axis=1)  # get indices of the maximum values with idk class. In this case the indices are the same than the class
            predicted_class_confidences_with_idk = confidences_outputs_with_idk[np.arange(
                len(class_predicted_with_idk)), class_predicted_with_idk]  # List with the confidences of the argmax softmax
            avg_idk_class_confidences = confidences_outputs_with_idk[
                np.arange(len(class_predicted_with_idk)), -1].mean()

            train_dataset_calculate_calibration_with_idk["y_gt"].extend(targets)
            train_dataset_calculate_calibration_with_idk["y_train_class_predicted"].extend(class_predicted_with_idk)
            train_dataset_calculate_calibration_with_idk["y_train_confidences_predicted_argmax"].extend(
                predicted_class_confidences_with_idk)

            # We calculate NLL for corrects and incorrects:
            correctly_classified = class_predicted_with_idk == targets.cpu().detach().numpy()
            logits_correct = outputs_logits[correctly_classified]
            targets_correct = targets[correctly_classified]
            NLL_loss_correct = F.cross_entropy(logits_correct, targets_correct)

            logits_incorrect = outputs_logits[~correctly_classified]
            targets_incorrect = targets[~correctly_classified]
            NLL_loss_incorrect = F.cross_entropy(logits_incorrect, targets_incorrect)

            outputs_pred = F.softmax(outputs_logits, dim=1)
            # measure accuracy and record loss
            if args.dataset != 'catsdogs' and args.dataset != 'cacophony' and args.dataset != "celebA":
                prec1_with_idk, prec5_with_idk = accuracy(outputs_pred.data, targets.data, topk=(1, 5))
                top5.update(prec5_with_idk.item(), inputs.size(0))
            else:
                prec1_with_idk = accuracy(outputs_pred.data, targets.data, topk=(1,))[0]
            losses.update(loss.item(), inputs.size(0))
            NLLlosses.update(NLL_loss.item(), inputs.size(0))
            NLLlosses_correct.update(NLL_loss_correct.item(), inputs.size(0))
            NLLlosses_incorrect.update(NLL_loss_incorrect.item(), inputs.size(0))
            top1.update(prec1_with_idk.item(), inputs.size(0))
            avgIdkConfidences.update(avg_idk_class_confidences.item(), inputs.size(0))
            beta_penalties_mean.update(beta_penalty_mean.item(), 1)
            beta_penalties_std.update(beta_penalty_std.item(), 1)
            beta_penalties_mode.update(beta_penalty_mode.item(), 1)
            adaptive_targets_mean.update(adaptive_target_mean.item(), 1)
            adaptive_targets_std.update(adaptive_target_std.item(), 1)
            adaptive_targets_mode.update(adaptive_target_mode.item(), 1)

            writer.add_scalar("Loss per batch ", loss.item(), loss_idx_value)  # Tensorboard
            loss_idx_value += 1  # Tensorboard

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # We calculate the l2 norm of the weights in the final layer for future graphs
            if not args.clusterLessPower and (
                    args.loss == 'csc' or args.loss == 'csc_entropy' or args.loss == 'csc_sat_entropy'):
                model_part = model
            else:
                if isinstance(model, torch.nn.DataParallel):
                    model_part = model.module
                else:
                    model_part = model
            if args.arch == 'vgg16_bn':
                last_layer_of_weights = model_part.classifier[-1].weight
            elif args.arch == 'resnet50' or args.arch == 'resnet34' or args.arch == "resnet110":
                last_layer_of_weights = model_part.fc.weight
            elif args.arch == 'lcrn_normalization':
                last_layer_of_weights = model_part.linear.weight
            elif args.arch == 'vit':
                if args.transfer_learning:
                    last_layer_of_weights = model_part.classifier.weight
                else:
                    last_layer_of_weights = model_part.classification_head.fc2.weight
            else:
                raise ValueError(
                    f"You are using an arch different to vgg16_bn or resnet50 or resnet34 or resnet110 or vit: {args.arch}. In this case create a new if your your last_layer_weights.")
            l2norm = torch.norm(last_layer_of_weights, p=2)
            l2norms.update(l2norm.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} |  Loss2: {loss2:.4f} | NLLloss: {NLLloss:.4f} | NLLlossCorrect: {NLLlossCorrect:.4f} | NLLlossIncorrect: {NLLlossIncorrect:.4f} | l2norm: {l2norm:.4f} | top1: {top1: .4f} | top5: {top5: .4f} | idkConfidencesAvg: {idkConfidencesAvg: .4f} | idkConfidencesStd: {idkConfidencesStd: .4f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            loss2=losses2.avg,
            NLLloss=NLLlosses.avg,
            NLLlossCorrect=NLLlosses_correct.avg,
            NLLlossIncorrect=NLLlosses_incorrect.avg,
            l2norm=l2norms.avg,
            top1=top1.avg,
            top5=top5.avg,
            idkConfidencesAvg=avgIdkConfidences.avg,
            idkConfidencesStd=avgIdkConfidences.std
        )
        bar.next()
    bar.finish()
    # Values for the Reliability Diagram

    accuracy_list_with_idk, confidence_list_with_idk, counter_list_with_idk, gap_list_with_idk, MCE_with_idk, ECE_with_idk, RMSCE_with_idk, min_list_with_idk, counter_zero_list_with_idk = calculate_confidence_accuracy_multiclass(
        train_dataset_calculate_calibration_with_idk["y_gt"],
        train_dataset_calculate_calibration_with_idk["y_train_class_predicted"],
        train_dataset_calculate_calibration_with_idk["y_train_confidences_predicted_argmax"], 10)

    return (
    losses.avg, losses2.avg, NLLlosses.avg, NLLlosses_correct.avg, NLLlosses_incorrect.avg, l2norms.avg, top1.avg,
    avgIdkConfidences.avg, avgIdkConfidences.std, MCE_with_idk, ECE_with_idk, beta_penalties_mean.mean(), beta_penalties_std.std(), beta_penalties_mode.mode(),
    adaptive_targets_mean.mean(), adaptive_targets_std.std(), adaptive_targets_mode.mode(), accuracy_list_with_idk,
    confidence_list_with_idk, counter_list_with_idk, gap_list_with_idk, RMSCE_with_idk, min_list_with_idk,
    counter_zero_list_with_idk, moco_top1.avg)


def test(testloader, model, criterion, epoch, use_cuda, reward, args, writer=None, loss_idx_value=0):
    global best_acc

    # Switch to eval mode
    model.eval()

    # switch to test mode
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    NLLlosses = AverageMeter()
    NLLlosses_correct = AverageMeter()
    NLLlosses_incorrect = AverageMeter()
    avgIdkConfidences = AverageMeter()
    l2norms = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    test_dataset_calculate_calibration_with_idk = {'y_gt': [], 'y_test_class_predicted': [],
                                                   'y_test_confidences_predicted_argmax': []}

    end = time.time()
    bar = Bar('Processing', max=len(testloader))

    # Compute output
    abstention_results = []
    sr_results = []

    for batch_idx, batch_data in enumerate(testloader):
        inputs, targets, indices = batch_data
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            if args.clusterLessPower:
                primary_device = list(map(int, args.gpu_id.split(',')))[0]
                inputs, targets = inputs.to(f'cuda:{primary_device}'), targets.to(f'cuda:{primary_device}')
            else:
                inputs, targets = inputs.cuda(), targets.cuda()

        # inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets) # For pytorch version > 0.4.0 this is not neccesary
        targets = targets.to(torch.int64)

        # Tensorboard
        if args.dataset != "cacophony" and batch_idx == 0:
            writer.add_image("Dev - Example image input at batch 0 of each epoch ", inputs[0], global_step=epoch)
            writer.add_scalar("Dev - Example image label at batch 0 of each epoch ", targets[0], global_step=epoch)

        # compute output
        with torch.no_grad():
            if args.arch == "vit":
                if args.transfer_learning:
                    outputs_logits_tl = model(inputs)
                    outputs_logits = outputs_logits_tl.logits
                else:
                    outputs_logits, _ = model(inputs)
            else:
                outputs_logits = model(inputs)

            # Calculate loss
            if epoch >= args.pretrain:
                if args.loss == 'sat':
                    if args.clusterLessPower:
                        loss = F.cross_entropy(outputs_logits[:, :-1].to(f'cuda:{primary_device}'), targets)
                    else:
                        loss = F.cross_entropy(outputs_logits[:, :-1].cuda(), targets)
                    """
                        Explanation why only take the classes without idk class: 
                            We cannot use the criterion of SAT to calculate on test because 
                            we do not have last predictions. For that reason, t is going to be 1 and
                            the idk class is not going to take into account never. 
                            So, first we only calculate the cross_entropy without the idk class
                    """
                elif args.loss == 'gambler':
                    loss = criterion(outputs_logits, targets, reward)
                elif args.loss == 'Socrates' and not args.first_epochs_ce:
                    criterion2 = Focal.FocalLoss(gamma=args.gamma_focal_loss, alpha=args.alpha_focal_loss,
                                                 size_average=True)
                    loss = criterion2(outputs_logits[:, :-1], targets)
                    """
                        Explanation why only take the classes without idk class and why focal loss: 
                            We cannot use the Socrates loss to calculate on test because 
                            we do not have last predictions. For that reason, t is going to be 1 and
                            the idk class is not going to take into account never. 
                            So, first we only calculate the Focal Loss without the idk class
                    """
                elif args.loss == 'Socrates':
                    loss = F.cross_entropy(outputs_logits[:, :-1], targets)
                    """
                        Same explanation than above.
                    """
                elif args.loss == 'dece':
                    loss_ce = F.cross_entropy(outputs_logits, targets)
                    loss_dece = criterion(outputs_logits, targets)
                    loss = loss_ce + 0.5 * loss_dece
                else:  # Ce, focal, brierScore ...
                    loss = criterion(outputs_logits, targets)

                # analyze the accuracy at different abstention level
                # Only after pretrain because in pretrain
                outputs_abstention = F.softmax(outputs_logits, dim=1)
                values, predictions = outputs_abstention.data.max(1)  # Return 2 elems, first the max values along classes axis, second a tensor with the indices (the class index) of the max values
                if args.loss == 'ce' or args.loss == 'focal' or args.loss == 'focalAdaptive' or args.loss == 'dece' or args.loss == 'csc' or args.loss == 'csc_entropy' or args.loss == 'brierScore':
                    outputs, reservation = outputs_abstention, (outputs_abstention * torch.log(outputs_abstention)).sum(
                        -1)  # Reservation is neg. entropy here.
                    pred_logits = F.softmax(outputs_logits, -1)  # For the sr_results
                else:
                    outputs, reservation = outputs_abstention[:, :-1], outputs_abstention[:, -1]
                    pred_logits = F.softmax(outputs_logits[:, :-1], -1)  # For the sr_results
                abstention_results.extend(
                    zip(list(reservation.cpu().numpy()), list(predictions.eq(targets.data).cpu().numpy())))
                sr_results.extend(
                    zip(list(pred_logits.max(-1)[0].cpu().numpy()), list(predictions.eq(targets.data).cpu().numpy())))
            else:
                if args.loss == 'ce' or args.loss == 'csc' or args.loss == 'csc_entropy':
                    loss = F.cross_entropy(outputs_logits, targets)
                elif args.loss == 'focal' or args.loss == 'focalAdaptive' or args.loss == 'brierScore':   # Put focal with pretrain 0
                    loss = criterion(outputs_logits, targets)
                elif args.loss == 'dece':
                    loss_ce = F.cross_entropy(outputs_logits, targets)
                    loss_dece = criterion(outputs_logits, targets)
                    loss = loss_ce + 0.5 * loss_dece
                elif args.loss == 'Socrates' and not args.first_epochs_ce:
                    criterion2 = Focal.FocalLoss(gamma=args.gamma_focal_loss, alpha=args.alpha_focal_loss,
                                                 size_average=True)
                    loss = criterion2(outputs_logits[:, :-1], targets)
                else:
                    if args.clusterLessPower:
                        loss = F.cross_entropy(outputs_logits[:, :-1].to(f'cuda:{primary_device}'), targets)
                    else:
                        loss = F.cross_entropy(outputs_logits[:, :-1].cuda(), targets)

            NLL_loss = F.cross_entropy(outputs_logits, targets)
            confidences_outputs_with_idk = F.softmax(outputs_logits, dim=1).cpu().detach().numpy()

            class_predicted_with_idk = np.argmax(confidences_outputs_with_idk,
                                                 axis=1)  # get indices of the maximum values without idk class. In this case the indices are the same than the class
            predicted_class_confidences_with_idk = confidences_outputs_with_idk[np.arange(
                len(class_predicted_with_idk)), class_predicted_with_idk]  # List with the confidences of the argmax softmax
            avg_idk_class_confidences = confidences_outputs_with_idk[
                np.arange(len(class_predicted_with_idk)), -1].mean()

            test_dataset_calculate_calibration_with_idk["y_gt"].extend(targets)
            test_dataset_calculate_calibration_with_idk["y_test_class_predicted"].extend(class_predicted_with_idk)
            test_dataset_calculate_calibration_with_idk["y_test_confidences_predicted_argmax"].extend(
                predicted_class_confidences_with_idk)

            # We calculate NLL for corrects and incorrects:
            correctly_classified = class_predicted_with_idk == targets.cpu().detach().numpy()
            logits_correct = outputs_logits[correctly_classified]
            targets_correct = targets[correctly_classified]
            NLL_loss_correct = F.cross_entropy(logits_correct, targets_correct)

            logits_incorrect = outputs_logits[~correctly_classified]
            targets_incorrect = targets[~correctly_classified]
            NLL_loss_incorrect = F.cross_entropy(logits_incorrect, targets_incorrect)

            # We calculate the l2 norm of the weights in the final layer for future graphs
            if not args.clusterLessPower and (
                    args.loss == 'csc' or args.loss == 'csc_entropy' or args.loss == 'csc_sat_entropy'):
                model_part = model
            else:
                model_part = model.module
            if args.arch == 'vgg16_bn':
                last_layer_of_weights = model_part.classifier[-1].weight
            elif args.arch == 'resnet50' or args.arch == 'resnet34' or args.arch == "resnet110":
                last_layer_of_weights = model_part.fc.weight
            elif args.arch == 'lcrn_normalization':
                last_layer_of_weights = model_part.linear.weight
            elif args.arch == 'vit':
                if args.transfer_learning:
                    last_layer_of_weights = model_part.classifier.weight
                else:
                    last_layer_of_weights = model_part.classification_head.fc2.weight
            else:
                raise ValueError(
                    f"You are using an arch different to vgg16_bn or resnet50 or resnet34 or resnet110 or vit: {args.arch}. In this case create a new if your your last_layer_weights.")

            l2norm = torch.norm(last_layer_of_weights, p=2)

            # measure accuracy and record loss
            # To methods with the idk class, the accuracy and losses are calculated without the idk class
            # But, it is notable that the output were calculated with the softmax taking into account the idk class
            outputs_pred = F.softmax(outputs_logits, dim=1)
            if args.dataset != 'catsdogs' and args.dataset != 'cacophony' and args.dataset != "celebA":
                prec1_with_idk, prec5_with_idk = accuracy(outputs_pred.data, targets.data, topk=(1, 5))
                top5.update(prec5_with_idk.item(), inputs.size(0))
            else:
                prec1_with_idk = accuracy(outputs_pred.data, targets.data, topk=(1,))[0]
            losses.update(loss.item(), inputs.size(0))
            NLLlosses.update(NLL_loss.item(), inputs.size(0))
            NLLlosses_correct.update(NLL_loss_correct.item(), inputs.size(0))
            NLLlosses_incorrect.update(NLL_loss_incorrect.item(), inputs.size(0))
            top1.update(prec1_with_idk.item(), inputs.size(0))
            avgIdkConfidences.update(avg_idk_class_confidences.item(), inputs.size(0))
            l2norms.update(l2norm.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | NLLloss: {NLLloss:.4f} | NLLlossCorrect: {NLLlossCorrect:.4f} | NLLlossIncorrect: {NLLlossIncorrect:.4f} |  l2norm: {l2norm:.4f} | top1: {top1: .4f} | top5: {top5: .4f}| idkConfidencesAvg: {idkConfidencesAvg: .4f} | idkConfidencesStd: {idkConfidencesStd: .4f}'.format(
            batch=batch_idx + 1,
            size=len(testloader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            NLLloss=NLLlosses.avg,
            NLLlossCorrect=NLLlosses_correct.avg,
            NLLlossIncorrect=NLLlosses_incorrect.avg,
            l2norm=l2norms.avg,
            top1=top1.avg,
            top5=top5.avg,
            idkConfidencesAvg=avgIdkConfidences.avg,
            idkConfidencesStd=avgIdkConfidences.std
        )
        bar.next()
    bar.finish()

    # Reliability Diagram
    accuracy_list_with_idk, confidence_list_with_idk, counter_list_with_idk, gap_list_with_idk, MCE_with_idk, ECE_with_idk, RMSCE_with_idk, min_list_with_idk, counter_zero_list_with_idk = calculate_confidence_accuracy_multiclass(
        test_dataset_calculate_calibration_with_idk["y_gt"],
        test_dataset_calculate_calibration_with_idk["y_test_class_predicted"],
        test_dataset_calculate_calibration_with_idk["y_test_confidences_predicted_argmax"], 10)

    return (losses.avg, NLLlosses.avg, NLLlosses_correct.avg, NLLlosses_incorrect.avg, l2norms.avg, top1.avg,
            avgIdkConfidences.avg, avgIdkConfidences.std, MCE_with_idk, ECE_with_idk, accuracy_list_with_idk,
            confidence_list_with_idk, counter_list_with_idk, gap_list_with_idk, RMSCE_with_idk, min_list_with_idk,
            counter_zero_list_with_idk)


# this function is used to organize all data and write into one file
def save_data(results_valid, results, reward_list, save_path):
    for reward in reward_list:
        save = open(os.path.join(save_path, 'coverage_vs_err.csv'), 'w')
        save.write(
            '0,100val.,100test,99v,99t,98v,98t,97v,97t,95v,95t,90v,90t,85v,85t,80v,80t,75v,75t,70v,70t,60v,60t,50v,50t,40v,40t,30v,30t,20v,20t,10v,10t\n')
        save.write('o{:.2f},'.format(reward))
        for idx, _ in enumerate(results_valid):
            save.write('{:.3f},'.format((1 - results_valid[idx][1]) * 100))
            if results != -1:
                save.write('{:.3f},'.format((1 - results[idx][1]) * 100))
        save.write('\n')

        save.write('thresholds\n')
        for idx, _ in enumerate(results_valid):
            save.write('{:.3f},'.format((1 - results_valid[idx][2]) * 100))
            if results != -1:
                save.write('{:.3f},'.format((1 - results[idx][2]) * 100))

        save.write('\n')

        save.write("AbstentionLogitTest,Coverage,Error,Threshold\n")
        for idx, _ in enumerate(results_valid):
            save.write('{:.2f},\t\t{:.3f},\t\t{:.3f}\n'.format(results_valid[idx][0] * 100.,
                                                               (1 - results_valid[idx][1]) * 100,
                                                               results_valid[idx][2]))

        save.write('\n')
        save.close()


def save_data_all_results(type, abortion_results, abortion_results_LeoFeng, sr_results, expected_coverage, reward_list,
                          save_path):
    for reward in reward_list:
        save = open(os.path.join(save_path, 'coverage_vs_err.csv'), 'w')
        save.write("Results with test set\n")
        save.write("abortion_results\n")
        save.write("AbstentionLogitTest,Coverage,Error,Threshold\n")
        for idx, _ in enumerate(abortion_results):
            save.write(
                '{:.0f},\t{:.2f},\t\t{:.3f},\t\t{:.3f}\n'.format(expected_coverage[idx],
                                                                 abortion_results[idx][0] * 100.,
                                                                 (1 - abortion_results[idx][1]) * 100,
                                                                 abortion_results[idx][2]))
        save.write("abortion_results_LeoFeng\n")
        save.write("AbstentionLogitTest,Coverage,Error,Threshold\n")
        for idx, _ in enumerate(abortion_results_LeoFeng):
            save.write('{:.0f},\t{:.2f},\t\t{:.3f},\t\t{:.3f}\n'.format(expected_coverage[idx],
                                                                        abortion_results_LeoFeng[idx][0] * 100.,
                                                                        (1 - abortion_results_LeoFeng[idx][1]) * 100,
                                                                        abortion_results_LeoFeng[idx][2]))

        # Softmax Response Results
        save.write("Softmax Response Results\n")
        save.write("AbstentionLogitTest,Coverage,Error,Threshold\n")
        for idx, _ in enumerate(sr_results):
            save.write(
                '{:.0f},\t{:.2f},\t\t{:.3f},\t\t{:.3f}\n'.format(expected_coverage[idx], sr_results[idx][0] * 100.,
                                                                 (1 - sr_results[idx][1]) * 100, sr_results[idx][2]))

        save.write('\n')
        save.close()


def evaluate(model, use_cuda, args, expected_coverage, reward_list, base_path, save_path, trainloader=None,
             valloader=None, testloader=None, train_batch=None, val_batch=None, test_batch=None, labels=None,
             input_size=(45, 3, 24, 24), num_classes=2, types_collate = None):

    with (torch.no_grad()):
        if args.dirichlet:
            reg = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
            # Full Dirichlet
            calibrator = FullDirichletCalibrator(reg_lambda=reg, reg_mu=None, args_run = args)
            print("Doing GridSearchCV for Dirichlet...")
            gscv = GridSearchCV(calibrator, param_grid={'reg_lambda': reg, 'reg_mu': [None]}, verbose=10, n_jobs=-1,
                                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0), scoring='neg_log_loss')
            print("Done GridSearchCV for Dirichlet. Continuing")

            if isinstance(model, torch.nn.DataParallel):
                model = model.module
            device_0 = torch.device("cuda:0")
            model = model.to(device_0)
            output_d = []
            labels_d = []

            for inputs_d, lbls_d, _ in valloader:
                inputs_d = inputs_d.to(device)
                output_d.append(model(inputs_d).cpu())
                labels_d.append(lbls_d.cpu())
            output_val = torch.cat(output_d)  # shape [N, num_classes]
            labels_val = torch.cat(labels_d)  # shape [N]
            print("Doing Fit for Dirichlet...")
            if args.loss == 'Socrates':
                output_val = output_val[:, :-1]  # Removing the idk class
            gscv.fit(output_val.cpu().numpy(), labels_val.cpu().numpy())
            best_model = gscv.best_estimator_
            print("Done Fit for Dirichlet. Continuing")
        if args.temperatureScaling:
            if isinstance(model, torch.nn.DataParallel):
                model = model.module
            device_0 = torch.device("cuda:0")
            model = model.to(device_0)
            model = ModelWithTemperature(model, args)
            model = model.set_temperature(valloader)
        if args.vectorScaling or args.matrixScaling:
            if isinstance(model, torch.nn.DataParallel):
                model = model.module
            device_0 = torch.device("cuda:0")
            model = model.to(device_0)
            output_d = []
            labels_d = []
            for inputs_d, lbls_d, _ in valloader:
                inputs_d = inputs_d.to(device)
                if args.arch == "vit" and not args.temperatureScaling:
                    if args.transfer_learning:
                            outputs_logits_tl = model(inputs_d)
                            output_tl = outputs_logits_tl.logits
                            output_d.append(output_tl.cpu())
                    else:
                        output_d_notl, _ = model(inputs_d)
                        output_d.append(output_d_notl.cpu())
                else:
                    output_d.append(model(inputs_d).cpu())
                labels_d.append(lbls_d.cpu())
            output_val = torch.cat(output_d)  # shape [N, num_classes]
            labels_val = torch.cat(labels_d)  # shape [N]
            print("Doing Fit for Vector/Matrix Scaling...")
            if args.vectorScaling:
                calibrator_scaling = VectorScaling(num_classes, True, args_run=args)
            if args.matrixScaling:
                calibrator_scaling = MatrixScaling(num_classes, True, args_run=args)
            calibrator_scaling.fit(output_val, labels_val)
            print("Done Fit for Vector/Matrix Scaling. Continuing")


    # Switch to eval mode
    model.eval()
    if types_collate == None:
        types_collate = {"test": [test_batch, testloader], "val": [val_batch, valloader]}
    folder_path = save_path + "/evaluation/"  ## CHANGE YOURS!
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    for k, v in types_collate.items():
        if args.dirichlet:
            collate_model_info(model, None, folder_path, v[0], v[1], labels,
                               args, input_size=input_size, ds_type=k, use_cuda=use_cuda, num_classes=num_classes,
                               calibrator=best_model)
        elif args.vectorScaling or args.matrixScaling:
            collate_model_info(model, None, folder_path, v[0], v[1], labels,
                               args, input_size=input_size, ds_type=k, use_cuda=use_cuda, num_classes=num_classes, calibrator=calibrator_scaling)
        else:
            collate_model_info(model, None, folder_path, v[0], v[1], labels,
                               args, input_size=input_size, ds_type=k, use_cuda=use_cuda, num_classes=num_classes)

    # Selective Classification:
    print("SELECTIVE CLASSIFICATION")
    abortion_results = [[], []]
    sr_results = [[], []]

    feature = []

    # New code CSC-CL
    sim_abortion = [[], []]
    nce_abortion = [[], []]
    sim_dot_product_abortion = [[], []]
    prediction_list = []
    confidences_list = []
    y_list = []
    abortion_results_valid = [[], []]
    sr_results_valid = [[], []]



    with torch.no_grad():
        for batch_idx, batch_data in enumerate(testloader):
            inputs, targets = batch_data[:2]  # inputs, targets, indices = batch_data

            targets = targets.to(torch.int64)
            if use_cuda:
                if args.clusterLessPower:
                    primary_device = list(map(int, args.gpu_id.split(',')))[0]
                    inputs, targets = inputs.to(f'cuda:{primary_device}'), targets.to(f'cuda:{primary_device}')
                else:
                    inputs, targets = inputs.cuda(), targets.cuda()

            y_list.append(targets.cpu().numpy())

            if args.arch == "vit" and not args.temperatureScaling:
                if args.transfer_learning:
                    outputs_logits_tl = model(inputs)
                    output = outputs_logits_tl.logits
                else:
                    output, _ = model(inputs)
            else:
                output = model(inputs)

            if args.dirichlet:
                if args.loss == 'Socrates':
                    output = output[:, :-1]  # Removing the idk class
                output = gscv.predict_proba(output)
                output = torch.from_numpy(output).to(inputs.device)
            elif args.vectorScaling or args.matrixScaling:
                output = calibrator_scaling.calibrate(output)
                output = torch.from_numpy(output).to(inputs.device)

            output_logits = output.detach().clone()

            if args.loss == 'csc' or args.loss == 'csc_entropy':
                if args.arch != 'vgg16_bn':
                    feature.extend(list(torch.flatten(hidden_features, 1)))
                else:
                    feature.extend(list(hidden_features))

            if args.loss == 'ce' or args.loss == 'focal' or args.loss == 'focalAdaptive' or args.loss == 'csc' or args.loss == 'dece' or args.loss == 'csc_entropy' or args.loss == 'brierScore':
                pred_logits = F.softmax(output, -1)  # to calculate the SR
            else:
                pred_logits = F.softmax(output[:, :-1], -1)  # to calculate the SR
            output = F.softmax(output, dim=1)  # To calculate the other methods (SAT, deepgamblers...)
            confidences = output.detach().clone()



            print("################################### OUTPUT SOFTMAX SELECTIVE CLASSIFICATION", output)
            if args.loss == 'csc' or args.loss == 'csc_entropy':
                top_logits, top_indices = torch.topk(output, k=2, dim=-1)

                first_max_indices = top_indices[:, 0]
                second_max_indices = top_indices[:, 1]

                first_max_logits = torch.gather(output, 1, first_max_indices.unsqueeze(1))
                second_max_logits = torch.gather(output, 1, second_max_indices.unsqueeze(1))

                first_max_logits_nosoftmax = torch.gather(output_logits, 1, first_max_indices.unsqueeze(1))
                second_max_logits_nosoftmax = torch.gather(output_logits, 1, second_max_indices.unsqueeze(1))
                difference = first_max_logits - second_max_logits
                difference_nosoftmax = first_max_logits_nosoftmax - second_max_logits_nosoftmax
                output, reservation, reservation_nosoftmax = output, -difference.squeeze(), -difference_nosoftmax.squeeze()
            elif args.loss == 'ce' or args.loss == 'focal' or args.loss == 'focalAdaptive' or args.loss == 'dece' or args.loss == 'brierScore':
                reservation = 1 - output.data.max(1)[0]
            else:
                output, reservation = output[:, :-1], (output[:, -1])
            values, predictions = output.data.max(
                1)  # Get the prediction values and the prediction positions (i.e., the class)
            prediction_list.extend(list(predictions))
            confidences_list.extend(list(confidences))
            abortion_results[0].extend(list(reservation.cpu()))
            abortion_results[1].extend(list(predictions.eq(targets.data).cpu()))
            sr_results[0].extend(list(-pred_logits.max(-1)[0].cpu()))
            sr_results[1].extend(list(predictions.eq(targets.data).cpu()))
        y_list = np.concatenate(y_list)


    #################################################################
    ######## CODE FROM CCL-SC TO EVALUATE SELECTIVE CLASSIFICATION:
    #################################################################
    print("STARTING EVALUATION CCL-SC")
    # valid & test
    correct_list = sr_results[1]
    num_valid = 1000

    if args.loss == 'csc' or args.loss == 'csc_entropy':
        def get_feature_avg(correct_list, prediction_list, features):
            data = [(feature, prediction, correct) for feature, prediction, correct in
                    zip(features, prediction_list, correct_list)]

            print("NUM CLASSES", num_classes)
            average_features = [np.zeros_like(features[0].cpu().numpy()) for _ in range(num_classes)]
            class_counts = [0] * num_classes
            error_class_counts = [0] * num_classes

            for feature, prediction, correct in data:
                if correct:
                    average_features[prediction.item()] += feature.cpu().numpy()
                    class_counts[prediction.item()] += 1
                else:
                    error_class_counts[prediction.item()] += 1
            for i in range(num_classes):
                if class_counts[i] > 0:
                    average_features[i] /= class_counts[i]

            return average_features, class_counts

        # average_features_train, class_counts_train = get_feature_avg(correct_list_train, prediction_list_train, feature_train)
        average_features, class_counts = get_feature_avg(correct_list[:num_valid], prediction_list[:num_valid],
                                                         feature[:num_valid])

        def cal_sim(average_features, features, prediction_list, correct_list):
            sim_abortion = []
            sim_abortion_class = [[[], []] for _ in range(num_classes)]
            sim_class = [0] * num_classes
            nce_abortion = []
            average_features_tensor = torch.tensor(average_features)
            sim_dot_product = []
            class_count = [0] * num_classes
            for i, prediction in enumerate(prediction_list):
                a = features[i].cpu().numpy().reshape(1, -1)
                b = average_features[prediction.item()].reshape(1, -1)
                cosine_similarity_matrix = cosine_similarity(a, b)

                dot_product = a[0, 0] * b[0, 0]
                sim_abortion.append(-cosine_similarity_matrix[0, 0])
                sim_dot_product.append(dot_product)
                sim_abortion_class[prediction.item()][0].append(-cosine_similarity_matrix[0, 0])
                sim_abortion_class[prediction.item()][1].append(correct_list[i])
                class_count[prediction.item()] += 1
                sim_class[prediction.item()] += sim_abortion[-1]

                # info_nce loss
                a = torch.tensor(a)
                labels = torch.tensor(prediction.item()).unsqueeze(0)
                logits = torch.mm(a, average_features_tensor.t())

                loss = F.cross_entropy(logits, labels)
                nce_abortion.append(loss.item())

            return torch.Tensor(sim_abortion), sim_abortion_class, torch.Tensor(nce_abortion), torch.Tensor(sim_dot_product)

        sim_abortion[0], sim_abortion_class, nce_abortion[0], sim_dot_product_abortion[0] = cal_sim(average_features,
                                                                                                    feature[
                                                                                                    num_valid:],
                                                                                                    prediction_list[
                                                                                                    num_valid:],
                                                                                                    correct_list[
                                                                                                    num_valid:])

    sim_abortion[1] = sr_results[1][num_valid:]
    nce_abortion[1] = sr_results[1][num_valid:]
    sim_dot_product_abortion[1] = sr_results[1][num_valid:]

    abortion_results_valid[0] = abortion_results[0][:num_valid]
    abortion_results_valid[1] = abortion_results[1][:num_valid]
    sr_results_valid[0] = sr_results_valid[0][:num_valid]
    sr_results_valid[1] = sr_results_valid[1][:num_valid]

    abortion_results_csc = abortion_results.copy()
    abortion_results_csc[0] = abortion_results_csc[0][num_valid:]
    abortion_results_csc[1] = abortion_results_csc[1][num_valid:]

    sr_results_csc = sr_results.copy()
    sr_results_csc[0] = sr_results_csc[0][num_valid:]
    sr_results_csc[1] = sr_results_csc[1][num_valid:]

    abortion_scores, abortion_correct = torch.stack(abortion_results_csc[0]).cpu(), torch.stack(abortion_results_csc[1]).cpu()
    sr_scores, sr_correct = torch.stack(sr_results_csc[0]).cpu(), torch.stack(sr_results_csc[1]).cpu()

    # Abstention Logit Results
    abortion_results_csc = []
    bisection_method_LeoFeng(abortion_scores, abortion_correct, abortion_results_csc, expected_coverage)

    ##   Reliability Diagram with idk class according to splitted of the data
    print("Doing reliability diagram with splitted data")
    prediction_list_valid = prediction_list[num_valid:]
    if isinstance(prediction_list_valid, torch.Tensor):
        prediction_list_valid.cpu().numpy()
    else:
        # If it's a list, move each tensor in the list to CPU and convert to NumPy
        prediction_list_valid = [
            tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor
            for tensor in prediction_list_valid
        ]
        prediction_list_valid = np.array(prediction_list_valid)  # Combine into a NumPy array

    confidences_tensor = torch.stack(confidences_list[num_valid:])
    higher_predicted_confidences_with_idk = confidences_tensor[
        torch.arange(len(prediction_list_valid)), torch.tensor(prediction_list_valid, device=confidences_tensor.device)].cpu().numpy()  # List with the confidences of the argmax softmax
    accuracy_list_1_idk, confidence_list_1_idk, counter_list_1_idk, gap_list_1_idk, MCE_1_idk, ECE_1_idk, RMSCE_1_idk, min_list_1_idk, counter_zero_list_1_idk = calculate_confidence_accuracy_multiclass(
        y_list[num_valid:], prediction_list_valid, higher_predicted_confidences_with_idk, 10)
    df_ECE_MCE_RMSCE = pd.DataFrame(
        {"ECE": [f"{ECE_1_idk:.6}"], "MCE": [f"{MCE_1_idk:.6f}"], "RMSCE": [f"{RMSCE_1_idk:.6f}"]})
    df_ECE_MCE_RMSCE.to_csv(f"{folder_path}seed_{args.manualSeed}_ECE_MCE_RMSCE_TEST_SELECTIVE_CLASSIFICATION.csv", index=False)
    print("Continue evaluating selective classification")
    # Continue evaluating csc
    with open(f"{folder_path}seed_{args.manualSeed}_selective_risk_ccl_sc.txt", 'w') as file:
        file.write("\nAbstention\tLogit\tTest\tCoverage\tError")
        print("\nAbstention\tLogit\tTest\tCoverage\tError")
        for idx, _ in enumerate(abortion_results_csc):
            print('{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], abortion_results_csc[idx][0] * 100.,
                                                      (1 - abortion_results_csc[idx][1]) * 100))
            file.write('\n{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], abortion_results_csc[idx][0] * 100.,
                                                             (1 - abortion_results_csc[idx][1]) * 100))
        # Softmax Response Results
        sr_results_csc = []
        bisection_method_LeoFeng(sr_scores, sr_correct, sr_results_csc, expected_coverage)
        file.write("\n\nSoftmax\tResponse\tTest\tCoverage\tError")
        print("\Softmax\tResponse\tTest\tCoverage\tError")
        for idx, _ in enumerate(sr_results_csc):
            # print('{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], abortion_results_csc[idx][0]*100., (1 - abortion_results_csc[idx][1])*100))
            print('{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], sr_results_csc[idx][0] * 100.,
                                                      (1 - sr_results_csc[idx][1]) * 100))
            file.write('\n{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], sr_results_csc[idx][0] * 100.,
                                                             (1 - sr_results_csc[idx][1]) * 100))

        if args.loss == 'csc' or args.loss == 'csc_entropy':
            sim_scores, sim_correct = sim_abortion[0].cpu(), torch.stack(sim_abortion[1]).cpu()
            sim_results = []
            bisection_method_LeoFeng(sim_scores, sim_correct, sim_results, expected_coverage)
            file.write("\nsim\tLogit\tTest\tCoverage\tError")
            print("\nsim\tLogit\tTest\tCoverage\tError")
            for idx, _ in enumerate(sim_results):
                print('{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], sim_results[idx][0] * 100.,
                                                          (1 - sim_results[idx][1]) * 100))
                file.write('\n{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], sim_results[idx][0] * 100.,
                                                                 (1 - sim_results[idx][1]) * 100))
            print(sim_scores.size(), sim_correct.size())
        if args.loss == 'csc' or args.loss == 'csc_entropy':
            product_scores, product_correct = sim_dot_product_abortion[0].cpu(), torch.stack(
                sim_dot_product_abortion[1]).cpu()
            product_results = []
            print(product_scores)
            bisection_method_LeoFeng(-product_scores, product_correct, product_results, expected_coverage)
            file.write("\nproduct_results\tLogit\tTest\tCoverage\tError")
            print("\nproduct_results\tLogit\tTest\tCoverage\tError")
            for idx, _ in enumerate(product_results):
                print('{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], product_results[idx][0] * 100.,
                                                          (1 - product_results[idx][1]) * 100))
                file.write('\n{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], product_results[idx][0] * 100.,
                                                                 (1 - product_results[idx][1]) * 100))

        if args.loss == 'csc' or args.loss == 'csc_entropy':
            nce_scores, nce_correct = nce_abortion[0].cpu(), torch.stack(nce_abortion[1]).cpu()
            nce_results = []
            bisection_method_LeoFeng(nce_scores, nce_correct, nce_results, expected_coverage)
            file.write("\nnce\tLogit\tTest\tCoverage\tError")
            print("\nnce\tLogit\tTest\tCoverage\tError")
            for idx, _ in enumerate(nce_results):
                print('{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], nce_results[idx][0] * 100.,
                                                          (1 - nce_results[idx][1]) * 100))
                file.write('\n{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], nce_results[idx][0] * 100.,
                                                                 (1 - nce_results[idx][1]) * 100))

        if args.loss == 'csc' or args.loss == 'csc_entropy':
            sim_scores, sim_correct = sim_abortion[0].cpu(), torch.stack(sim_abortion[1]).cpu()
            sim_scores2 = (sr_scores + sim_scores)

            sim_results = []
            bisection_method_LeoFeng(sim_scores2, sim_correct, sim_results, expected_coverage)
            file.write("\nsimxsr\tLogit\tTest\tCoverage\tError")
            print("\nsimxsr\tLogit\tTest\tCoverage\tError")
            for idx, _ in enumerate(sim_results):
                print('{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], sim_results[idx][0] * 100.,
                                                          (1 - sim_results[idx][1]) * 100))
                file.write('\n{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], sim_results[idx][0] * 100.,
                                                                 (1 - sim_results[idx][1]) * 100))
        if args.loss == 'csc' or args.loss == 'csc_entropy':
            sim_scores, sim_correct = sim_abortion[0].cpu(), torch.stack(sim_abortion[1]).cpu()
            sim_scores_2 = torch.max(sim_scores, sr_scores)
            print(sim_scores_2[:10], sim_scores[:10], sr_scores[:10])
            sim_results = []
            bisection_method_LeoFeng(sim_scores_2, sim_correct, sim_results, expected_coverage)
            file.write("\nsimsr_max\tLogit\tTest\tCoverage\tError")
            print("\nsimsr_max\tLogit\tTest\tCoverage\tError")
            for idx, _ in enumerate(sim_results):
                print('{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], sim_results[idx][0] * 100.,
                                                          (1 - sim_results[idx][1]) * 100))
                file.write('\n{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], sim_results[idx][0] * 100.,
                                                                 (1 - sim_results[idx][1]) * 100))

        if (args.loss == 'csc' or args.loss == 'csc_entropy') and args.dataset != "svhn":
            sim_results = []
            sim_scores = []
            sim_correct = []
            data_save = []
            for i in range(0, num_classes):
                sim_results_temp = []
                sim_scores_class, sim_correct_class = torch.Tensor(sim_abortion_class[i][0]), torch.stack(
                    sim_abortion_class[i][1]).cpu()
                sim_scores.extend(sim_scores_class)
                sim_correct.extend(sim_correct_class)
            print(torch.Tensor(sim_scores).size(), torch.Tensor(sim_correct).size())

            bisection_method_LeoFeng(torch.Tensor(sim_scores), torch.Tensor(sim_correct), sim_results,
                                     expected_coverage)
            file.write("\nsim\tLogit\tTest\tCoverage\tError")
            print("\nsim\tLogit\tTest\tCoverage\tError")
            for idx, _ in enumerate(sim_results):
                print('{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], sim_results[idx][0] * 100.,
                                                          (1 - sim_results[idx][1]) * 100))
                file.write('\n{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], sim_results[idx][0] * 100.,
                                                                 (1 - sim_results[idx][1]) * 100))
        if (args.loss == 'csc' or args.loss == 'csc_entropy') and args.dataset != "svhn":
            sim_results = [(0, 0)] * len(expected_coverage)

            for i in range(0, num_classes):
                sim_results_temp = []
                sim_scores_class, sim_correct_class = torch.Tensor(sim_abortion_class[i][0]), torch.stack(
                    sim_abortion_class[i][1]).cpu()
                bisection_method_LeoFeng(sim_scores_class, sim_correct_class, sim_results_temp, expected_coverage)
                # print(sim_results_temp)
                for j, (a, b, c) in enumerate(sim_results_temp):
                    sim_results[j] = (sim_results[j][0] + a, sim_results[j][1] + b)
            # print(sim_results)
            file.write("\nsim\tLogit\tTest\tCoverage\tError")
            print("\nsim\tLogit\tTest\tCoverage\tError")
            for idx, _ in enumerate(sim_results):
                print('{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx],
                                                          sim_results[idx][0] / num_classes * 100.,
                                                          (1 - sim_results[idx][1] / num_classes) * 100))
                file.write('\n{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx],
                                                                 sim_results[idx][0] / num_classes * 100.,
                                                                 (1 - sim_results[idx][1] / num_classes) * 100))

    ####### Until here
    ################################################
    ################################################

    ################################################
    ################################################
    ####### Our code:

    abortion_scores, abortion_correct = torch.stack(abortion_results[0]), torch.stack(abortion_results[1])
    sr_scores, sr_correct = torch.stack(sr_results[0]), torch.stack(sr_results[1])

    # Abstention Logit Results

    risk_coverage_results = {"selective_risk": [], "selective_risk_LeoFeng": [], "coverage": np.array([])}
    abortion_results = []
    bisection_method(abortion_scores, abortion_correct, abortion_results, expected_coverage)
    # Risk-coverage curves
    selective_risk = [(1 - r[1]) * 100 for r in abortion_results]
    risk_coverage_results["selective_risk"] = selective_risk
    risk_coverage_curve_plotly(selective_risk, np.array(expected_coverage), "test (without idk class)", folder_path,
                               loss=args.loss)

    risk_coverage_results["coverage"] = np.array(expected_coverage)

    abortion_results_LeoFeng = []
    bisection_method_LeoFeng(abortion_scores, abortion_correct, abortion_results_LeoFeng, expected_coverage)
    # Risk-coverage curves
    selective_risk_LeoFeng = [(1 - r[1]) * 100 for r in abortion_results_LeoFeng]
    risk_coverage_results["selective_risk_LeoFeng"] = selective_risk_LeoFeng
    risk_coverage_curve_plotly(selective_risk_LeoFeng, np.array(expected_coverage), "test (without idk class) - LF",
                               folder_path, loss=args.loss)

    # Saving risk-coverage in csv
    df_risk_coverage = pd.DataFrame(risk_coverage_results)
    df_risk_coverage.to_csv(f"{folder_path}seed_{args.manualSeed}_risk_coverage.csv", index=False)

    print("\nAbstention\tLogit\tTest\tCoverage\tError\tThreshold")
    for idx, _ in enumerate(abortion_results):
        print('{:.0f},\t{:.2f},\t\t{:.3f},\t\t{:.3f}'.format(expected_coverage[idx], abortion_results[idx][0] * 100.,
                                                             (1 - abortion_results[idx][1]) * 100,
                                                             abortion_results[idx][2]))

    print("\nAbstention\tLogit\tTest\tCoverage\tError\tThreshold \tLeoFeng")
    for idx, _ in enumerate(abortion_results_LeoFeng):
        print('{:.0f},\t{:.2f},\t\t{:.3f},\t\t{:.3f}'.format(expected_coverage[idx],
                                                             abortion_results_LeoFeng[idx][0] * 100.,
                                                             (1 - abortion_results_LeoFeng[idx][1]) * 100,
                                                             abortion_results_LeoFeng[idx][2]))

    # Softmax Response Results
    sr_results = []
    bisection_method_LeoFeng(sr_scores, sr_correct, sr_results, expected_coverage)

    print("\Softmax\tResponse\tTest\tCoverage\tError\tThreshold")
    for idx, _ in enumerate(sr_results):
        print('{:.0f},\t{:.2f},\t\t{:.3f},\t\t{:.3f}'.format(expected_coverage[idx], sr_results[idx][0] * 100.,
                                                             (1 - sr_results[idx][1]) * 100, sr_results[idx][2]))
    save_data_all_results("Test", abortion_results, abortion_results_LeoFeng, sr_results, expected_coverage,
                          reward_list, folder_path)

    print("Evaluation Finished")

    # Copying .log
    print(f"Copying .log in {folder_path}")
    log_file_path = os.getcwd() + "/log/" + save_path.split("/")[-1].split("\\")[0] + ".log"
    shutil.copy(log_file_path, folder_path)
    print("END")
    # Empty new .log for future runs
    with open(log_file_path, 'w') as log_file:  # Restart the file
        pass

    return


def _get_num_covered_and_confident_error_idxs(desired_coverages, preds, confidences, y_true):
    """Returns the number of covered samples and a list of confident error indices for each coverage"""
    sorted_confidences = list(sorted(confidences, reverse=True))

    confident_error_idxs = []
    num_covered = []
    for coverage in desired_coverages:
        threshold = sorted_confidences[int(coverage * len(preds)) - 1]
        confident_mask = confidences >= threshold
        confident_error_mask = (y_true != preds) * confident_mask
        confident_error_idx = confident_error_mask.nonzero()[0]

        confident_error_idxs.append(confident_error_idx)
        num_covered.append(np.sum(confident_mask))

    return num_covered, confident_error_idxs


def eval_converage(logits, confidences, labels, coverages=[100, 95, 90, 85, 80, 75, 70]):
    preds = np.argmax(logits, axis=1)
    correct = np.equal(preds, labels).astype(np.float32)

    desired_coverages = np.linspace(0.01, 1.00, 100)
    num_covered, confident_error_idxs = _get_num_covered_and_confident_error_idxs(
        desired_coverages, preds, confidences, labels)

    # Add accuracy at desired coverages to table and results
    num_errors_at, num_covered_at, acc_at = {}, {}, {}
    for cov in coverages:
        num_errors_at[cov] = len(confident_error_idxs[cov - 1])
        num_covered_at[cov] = num_covered[cov - 1]
        acc_at[cov] = 1.0 - (float(num_errors_at[cov]) / num_covered_at[cov])

    assert len(logits) == num_covered[-1]
    assert len(logits) == (np.sum(correct, axis=0) + len(confident_error_idxs[-1]))

    return acc_at


def eval_clean(model, device, test_loader, args):
    """
    evaluate model by white-box attack
    """
    model.eval()
    logits, label = [], []

    for batch in test_loader:
        data, target = batch[:2]
        targets = targets.to(torch.int64)
        data, target = data.to(device), target.to(device)
        X, y = data, target
        with torch.no_grad():
            if args.arch == "vit":
                if args.transfer_learning:
                    outputs_logits_tl = model(X)
                    out = outputs_logits_tl.logits
                else:
                    out, _ = model(X)
            else:
                out = model(X)
        logits.append(out.cpu().detach().numpy())
        label.append(y.cpu().detach().numpy())

    logits = np.concatenate(logits)
    label = np.concatenate(label)

    def shuffle_list(lst, seed=10):
        random.seed(seed)
        random.shuffle(lst)

    idx = list(range(label.shape[0]))
    shuffle_list(idx)
    idx = np.asarray(idx, dtype=np.int)
    logits = logits[idx]
    label = label[idx]

    if args.loss != 'ce' or args.loss != 'focal' or args.loss!='focalAdaptive' or args.loss!='dece' or args.loss != 'brierScore':
        logits = F.softmax(torch.from_numpy(logits), dim=1).numpy()
        confidences = 1 - logits[:, -1]
        logits = logits[:, :10]
    else:
        logits = F.softmax(torch.from_numpy(logits), dim=1).numpy()
        confidences = np.max(logits, axis=1)

    acc_at = eval_converage(logits[:2000], confidences[:2000], label[:2000])

    print("\tVal")
    for cov in acc_at:
        print("ERR@{}\t{:.2f}".format(cov, (1 - acc_at[cov]) * 100))

    acc_at = eval_converage(logits[2000:], confidences[2000:], label[2000:])

    print("\tTest")
    for cov in acc_at:
        print("ERR@{}\t{:.2f}".format(cov, (1 - acc_at[cov]) * 100))


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

def checkpoint(model, path, epoch, history):
    filename = f"{path}/STOPPED_{epoch}.pth"
    old_filename = f"{path}/STOPPED_{epoch-1}.pth"
    torch.save(model.state_dict(), filename)
    if os.path.exists(old_filename):
        os.remove(old_filename)
    filename = f"{path}/STOPPED_{epoch}_history.csv"
    old_filename = f"{path}/STOPPED_{epoch - 1}_history.csv"
    df_history = pd.DataFrame(history)
    df_history.to_csv(filename, index=False)
    if os.path.exists(old_filename):
        os.remove(old_filename)
    
def resume(model, filename):
    model.load_state_dict(torch.load(filename))
    if args.clusterLessPower:
        primary_device = list(map(int, args.gpu_id.split(',')))[0]
        model = model.to(f'cuda:{primary_device}')
    else:
        model = model.to('cuda')
    return model


## CSC loss

hidden_features = None
hidden_features_k = None
full = False

## CSC loss
def hook_fn(module, input, output):
    global hidden_features
    hidden_features = output

def hook_fn_k(module, input, output):
    global hidden_features_k
    hidden_features_k = output


def main(reward, reward_list, expected_coverage, base_path, save_path, resume_path, args, seed, config=None):
    print(args)

    # Initializing SummaryWriter for Tensorboard. It outputs will be written to ./runs/
    writer = SummaryWriter()

    # Starting logger for save results as txt
    title = args.dataset + '-' + args.arch + ' o={:.2f}'.format(reward)
    log_file_path_text = os.path.join(save_path, 'eval.txt' if args.evaluate else 'log.txt')
    logger = Logger(log_file_path_text, title=title)
    logger.set_names(
        ['Epoch', 'Learning Rate', 'Train Loss', 'Train Loss2', 'Val Loss', 'Train Acc.', 'Val Acc.',
         'Train Err.', 'Val Err.',
         'Train NLLloss', 'Val NLLloss',
         'Train NLLlossCorrect', 'Val NLLlossCorrect',
         'Train NLLlossIncorrect', 'Val NLLlossIncorrect',
         'Train L2Norm', 'Val L2Norm',
         'Train avgIdkConfidences', 'Val avgIdkConfidences',
         'Train stdIdkConfidences', 'Val stdIdkConfidences',
         'Train MCE', 'Train ECE', 'Val MCE', 'Val ECE',
         'MOCO Err.'
         ])


    #############
    ## DATASET ##
    #############
    print('==> Preparing dataset %s' % args.dataset)
    global num_classes

    trainloader, valloader, testloader, trainset, valset, testset, num_classes, train_batch, val_batch, test_batch, labels, input_size, input_shape = prepare_dataset(args.dataset, num_classes, args, 42, valid_size=0.1)
    if args.dataset == 'cacophony2':
        args.dataset = 'cacophony'
        print("args.dataset name changed from cacophony2 to "+args.dataset)
    # print("-- Example of trainloader ")
    # for batch_idx,  batch_data in tqdm(enumerate(trainloader), total=len(trainloader)):
    #     inputs, targets, indices = batch_data
    #     print("inputs.shape train one batch (size ", args.train_batch, "):", inputs.shape)
    #     print("targets.shape train one batch (size ", args.train_batch, "):", targets.shape)
    #     print("inputs example ", inputs)
    #     print("targets example ", targets)
    #     print()
    #     break

    # End of Dataset

    #############
    ##  MODEL  ##
    #############
    print("==> creating model '{}'".format(args.arch))

    # Model instantiation
    if args.arch == "lcrn_normalization" or args.arch == "lcrn_normalization_new":
        if args.arch == "lcrn_normalization":
            model = models.__dict__[args.arch](dataset_name=args.dataset,
                                               classes=num_classes if args.loss == 'ce' or args.loss == 'focal' or args.loss == 'focalAdaptive' or args.loss == 'dece' or args.loss == 'brierScore' or args.loss == 'csc' else num_classes+1,
                                               input_shape=input_shape)
        else:
            model = models.__dict__[args.arch](dataset_name=args.dataset,
                                               classes=num_classes if args.loss == 'ce' or args.loss == 'focal' or args.loss == 'focalAdaptive' or args.loss == 'dece' or args.loss == 'brierScore' or args.loss == 'csc' else num_classes + 1,
                                               input_shape=input_shape, dropout_1=0.45, dropout_2=0.3, l1_reg=0, l2_reg=0)
        model_name = "vgg16_cacophony"
    elif args.arch == "vit":
        if args.transfer_learning: # https://huggingface.co/google/vit-base-patch16-224 https://huggingface.co/docs/transformers/model_doc/vit
            # Things to reinitiate in the transformer for transfer learning: the classification head as we do not use
            # the same number of classes (zero_head = True) to randomize the weights, the number of classes,
            # Then, using from_pretrained and num_labels detects automatically the change in the number of classes and
            # reinitiate automatically the classification head
            # Now, we can frozen different parts of the ViT:
            # 1) Feature extraction / linear probing: frozen all the backbone, only trains the classification head
            # 2) Partial fine-tuning / gradual unfreezing: froze the first layers, train the last ones and the classification head
            # 3) Full fine-tuning / end-to-end fine-tuning / fine-tuning: trains all the ViT, without frozen anything
            # We are going to use 3) fine-tuning with the classification head reinitiated as we have a different number of classes
            model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', #-in21k')  # ViT-B_16 Base path 16
                                                              num_labels = num_classes if args.loss == 'ce' or args.loss == 'focal' or args.loss == 'focalAdaptive' or args.loss == 'dece' or args.loss == 'brierScore' or args.loss == 'csc' else num_classes + 1,
                                                              ignore_mismatched_sizes=True
                                                              )
            resnet_features = None
            model_name = "vit_tl"
        else:
            model = models.__dict__[args.arch](patch_size=args.vit_patch_size, max_len=args.vit_max_len, embed_dim=args.vit_embed_dim,
                                               classes=num_classes if args.loss == 'ce' or args.loss == 'focal' or args.loss == 'focalAdaptive' or args.loss == 'dece' or args.loss == 'brierScore' or args.loss == 'csc' else num_classes + 1,
                                               layers=args.vit_layers, channels=3, heads=args.vit_heads)
            resnet_features = None
            model_name = model.name
    else:
        if args.dataset == "cacophony":
            model = models.__dict__[args.arch](dataset_name=args.dataset,
                                               classes=num_classes if args.loss == 'ce' or args.loss == 'focal' or args.loss == 'focalAdaptive' or args.loss == 'dece' or args.loss == 'brierScore' or args.loss == 'csc' else num_classes + 1)
        else:
            model = models.__dict__[args.arch](dataset_name=args.dataset,
                                               num_classes=num_classes if args.loss == 'ce' or args.loss == 'focal' or args.loss == 'focalAdaptive' or args.loss == 'dece' or args.loss == 'brierScore' or args.loss == 'csc' else num_classes+1,
                                               input_size=input_size)
        model_name = model.name

    # Cuda
    if use_cuda:
        if args.loss == 'csc' and not args.clusterLessPower:
            model = model.cuda()
        else:
            if args.clusterLessPower:
                model = torch.nn.DataParallel(model, device_ids= list(map(int, args.gpu_id.split(','))))
                model.to(device)
            else:
                model = torch.nn.DataParallel(model.cuda())

    cudnn.benchmark = True # Optimize the cudnn library (gpu-accelerated library)  for the specific hardware configuration
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))


    ##########
    ## LOSS ##
    ##########
    # Loss/criterion selection
    if args.pretrain: # We only use this loss if we don´t want to use any of our loss
        criterion = CrossEntropy.CrossEntropyLoss() # nn.CrossEntropyLoss()
    if args.loss == 'ce' or args.loss == 'csc':
        criterion = CrossEntropy.CrossEntropyLoss() # nn.CrossEntropyLoss()
    elif args.loss == 'brierScore':
        criterion = BrierScore.BrierScore()
    elif args.loss == 'gambler':
        criterion = DeepGambler.DeepGamblerLoss()
    elif args.loss == 'sat':
        criterion = SelfAdaptiveTraining.SelfAdaptiveTrainingLoss(num_examples=len(trainset), num_classes=num_classes,
                                                                  mom=args.sat_momentum)
    elif args.loss == 'dece':
        criterion = DECE.DECE(15, 100, 0.01)
    elif args.loss == 'focal':
        criterion = Focal.FocalLoss(gamma=args.gamma_focal_loss, alpha= args.alpha_focal_loss, size_average=True)
    elif args.loss == 'focalAdaptive':
        criterion = FocalAdaptiveGamma.FocalLossAdaptive(gamma=args.gamma_focal_loss, alpha=args.alpha_focal_loss, size_average=True)
    elif args.loss == 'Socrates':
        criterion = Socrates.SocratesLoss(num_examples=len(trainset), num_classes=num_classes,
                                                                     mom=args.sat_momentum,
                                                                     gamma=args.gamma_focal_loss,
                                                                     alpha=args.alpha_focal_loss,
                                                                     dynamic=args.dynamic,
                                                                     loss_name=args.loss,
                                                                     version=args.version,
                                                                     version_SAT_original=args.version_SAT_original,
                                                                     version_FOCALinGT=args.version_FOCALinGT,
                                                                     version_FOCALinSAT=args.version_FOCALinSAT,
                                                                     version_changingWithIdk=args.version_changingWithIdk,
                                                                     old=args.old
                                          )

    # CSC Loss
    # encoder_k
    model_k = None
    if args.loss == 'csc':
        model_k = copy.deepcopy(model)

        if args.arch == 'vgg16_bn' or args.arch == "vit":
            if args.arch == "vit":
                if args.transfer_learning:
                    print(model.classifier.register_forward_hook(hook_fn))
                    print(model_k.classifier.register_forward_hook(hook_fn_k))
                else:
                    if args.clusterLessPower:
                        print(model.module.classification_head.fc2.register_forward_hook(hook_fn))
                        print(model_k.module.classification_head.fc2.register_forward_hook(hook_fn_k))
                    else:
                        print(model.classification_head.fc2.register_forward_hook(hook_fn))
                        print(model_k.classification_head.fc2.register_forward_hook(hook_fn_k))
            elif args.clusterLessPower:
                print(list(model.module.classifier._modules.items())[2][1].register_forward_hook(hook_fn))
                print(list(model_k.module.classifier._modules.items())[2][1].register_forward_hook(hook_fn_k))
            else:
                print(list(model.classifier._modules.items())[2][1].register_forward_hook(hook_fn))
                print(list(model_k.classifier._modules.items())[2][1].register_forward_hook(hook_fn_k))

            if args.dataset == 'food101':
                args.moco_dim = 4096
            elif args.dataset == 'cifar100':
                if args.arch == "resnet110":
                    args.moco_dim = 2048
                elif args.arch == "vit":
                    args.moco_dim = 100
                else:
                    args.moco_dim = 512
            else:
                args.moco_dim = 512

        elif args.arch == 'resnet34' or args.arch == 'resnet18' or args.arch == "resnet110":
            if args.clusterLessPower:
                print(model.module.avgpool.register_forward_hook(hook_fn))
                print(model_k.module.avgpool.register_forward_hook(hook_fn_k))
            else:
                print(model.avgpool.register_forward_hook(hook_fn))
                print(model_k.avgpool.register_forward_hook(hook_fn_k))
            if args.dataset == 'food101':
                args.moco_dim = 25088
            elif args.arch == "resnet110":
                args.moco_dim = 2048
            else:
                args.moco_dim = 512


    # If evaluation (args.evaluate == True), the training part will not be executed
    ################
    ## EVALUATION ##
    ################
    if args.evaluate:
        print('\nEvaluation only')
        if args.dataset == 'cifar10C':
            save_path = save_path.replace('cifar10C', 'cifar10')
        elif args.dataset == 'cifar100C':
            save_path = save_path.replace('cifar100C', 'cifar100')

        save_path_aux = save_path
        if args.temperatureScaling:
            save_path_aux = save_path.replace('_temp_scaling', '')
        elif args.dirichlet:
            save_path_aux = save_path.replace('_dirichlet', '')
        elif args.vectorScaling:
            save_path_aux = save_path.replace('_vect_scaling', '')
        elif args.matrixScaling:
            save_path_aux = save_path.replace('_mat_scaling', '')

        if args.dataset == "celebAadam":
            args.dataset = "celebA"
        print(save_path_aux, resume_path)

        if resume_path=="":
            resume_path = os.path.join(save_path_aux, '{:d}.pth'.format(args.epochs))
        if args.temperatureScaling:
            resume_path = resume_path.replace('_temp_scaling', '')
        elif args.dirichlet:
            resume_path = resume_path.replace('_dirichlet', '')
        elif args.vectorScaling:
            save_path_aux = save_path.replace('_vect_scaling', '')
        elif args.matrixScaling:
            save_path_aux = save_path.replace('_mat_scaling', '')

        print(save_path_aux, resume_path)
        # Search the .pth file
        if not os.path.isfile(resume_path):
            files_withpth = []
            for root, dirs, files in list(os.walk(save_path_aux)):
                for f in files:
                    if str(args.epochs) in f and f.endswith('.pth'):
                        complete_path = os.path.join(root, f)
                        print(complete_path)
                        files_withpth.append(complete_path)
            resume_path = files_withpth[0]
        print("resume_path", resume_path)
        assert os.path.isfile(resume_path), 'no model exists at "{}"'.format(resume_path)
        epoch_saved = resume_path.split("_")[-1].split(".")[0]
        print("epoch_saved", epoch_saved)

        # Search if
        not_last_epoch = epoch_saved != str(args.epochs)
        if args.transfer_learning:
            # As we are using hugging face from the begining, we only need the state (weights) of the model
            loaded_model = torch.load(resume_path, map_location=device)
            state_dict_model = loaded_model.state_dict()
            model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224',
                                                              # -in21k')  # ViT-B_16 Base path 16
                                                              num_labels = num_classes if args.loss == 'ce' or args.loss == 'focal' or args.loss == 'focalAdaptive' or args.loss == 'dece' or args.loss == 'brierScore' or args.loss == 'csc' else num_classes + 1,
                                                              ignore_mismatched_sizes=True
                                                              )
            new_state_dict = {}
            for k, v in state_dict_model.items():
                if k.startswith("module."):
                    new_k = k[len("module."):]
                else:
                    new_k = k
                new_state_dict[new_k] = v
            model.load_state_dict(new_state_dict)
            model.to(device)
        elif not_last_epoch:
            model.load_state_dict(torch.load(resume_path))
        else:
            model = torch.load(resume_path, map_location=device)

        print("MODEL EVALUATED: ", model)
        if args.dataset == 'cifar10C':
            save_path = save_path.replace('cifar10', 'cifar10C')
        elif args.dataset == 'cifar100C':
            save_path = save_path.replace('cifar100', 'cifar100C')

        if args.loss == 'csc':
            if args.clusterLessPower:
                if args.arch == 'vgg16_bn':
                    print(list(model.module.classifier._modules.items())[2][1].register_forward_hook(hook_fn))
                    print(list(model.module.classifier._modules.items()))
                elif args.arch == "vit":
                    if args.transfer_learning:
                        print(model.classifier.register_forward_hook(hook_fn))
                        print(model_k.classifier.register_forward_hook(hook_fn_k))
                    else:
                        print(model.module.classification_head.fc2.register_forward_hook(hook_fn))
                        print(model_k.module.classification_head.fc2.register_forward_hook(hook_fn_k))
                elif args.arch == 'resnet34' or args.arch == 'resnet18' or args.arch == "resnet110":
                    print(model.module.avgpool)
                    print(model.module.avgpool.register_forward_hook(hook_fn))
            else:
                if args.arch == 'vgg16_bn':
                    print(list(model.classifier._modules.items())[2][1].register_forward_hook(hook_fn))
                    print(list(model.classifier._modules.items()))
                elif args.arch == "vit":
                    if args.transfer_learning:
                        print(model.classifier.register_forward_hook(hook_fn))
                        print(model_k.classifier.register_forward_hook(hook_fn_k))
                    else:
                        print(model.classification_head.fc2.register_forward_hook(hook_fn))
                        print(model_k.classification_head.fc2.register_forward_hook(hook_fn_k))
                elif args.arch == 'resnet34' or args.arch == 'resnet18' or args.arch == "resnet110":
                    print(model.avgpool)
                    print(model.avgpool.register_forward_hook(hook_fn))

        evaluate(model, use_cuda, args, expected_coverage, reward_list, base_path, save_path,
                 trainloader, valloader, testloader, train_batch, val_batch, test_batch, labels, input_size, num_classes)

        return
    # Evaluation part finished

    ##############
    ## TRAINING ##
    ##############

    # Optimizer
    if args.arch == "lcrn_normalization_new":
        optimizer = optim.Adam(model.parameters(), lr=2.6e-4, weight_decay=0)
        reduce_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.25, threshold=0.0001, min_lr=1e-8,
                                                         threshold_mode="abs")
    elif args.dataset == "celebAadam":
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        args.dataset = "celebA"
    else:
        if args.transfer_learning:
            # To try to not change too much the pretrained weights, we want to change little by little the backbone, but
            # as the classification head is completely reinitialized (as it has different number of classes) we want a
            # bigger search for the classification head
            if args.loss == 'csc':
                optimizer = optim.SGD([
                    {'params': model.vit.parameters(), 'lr': state['lr'] * 0.1},
                    # Backbone, slower change, lower learning rate
                    {'params': model.classifier.parameters(), 'lr': state['lr']},
                    # Classifier, faster change, higher learning rate
                ], momentum=args.momentum, weight_decay=args.weight_decay)
            else:
                optimizer = optim.SGD([
                    {'params': model.module.vit.parameters(), 'lr': state['lr'] * 0.1}, # Backbone, slower change, lower learning rate
                    {'params': model.module.classifier.parameters(), 'lr': state['lr']}, # Classifier, faster change, higher learning rate
                    ], momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(), lr=state['lr'], momentum=args.momentum,
                                  weight_decay=args.weight_decay)

    # Early Stopping
    if args.early_stopping:
        if args.arch == "lcrn_normalization_new":
            early_stopping = MetricMonitor(10, mode="max")
        else:
            early_stopping = MetricMonitor(13, mode="min")

    print("MODEL: ", model)

    # Train model
    print("Starting training...")

    # Initializing history to save the results during training
    history = {"epoch":[], "train_loss": [], "train_loss2": [], "val_loss": [], "train_NLLloss": [], "val_NLLloss": [],"train_NLLlossCorrect": [],
               "val_NLLlossCorrect": [], "train_NLLlossIncorrect": [], "val_NLLlossIncorrect": [], "train_l2norm": [],
               "val_l2norm": [], "train_accuracy": [], "val_accuracy": [], "train_avgIdkConfidences":[], "val_avgIdkConfidences":[],
               "train_stdIdkConfidences":[], "val_stdIdkConfidences":[], "train_ECE": [], "val_ECE": [], "train_MCE": [],
               "val_MCE": [], "train_beta_penalties_mean": [],"train_beta_penalties_std": [],"train_beta_penalties_mode": [],
               "train_adaptive_targets_mean": [],"train_adaptive_targets_std": [],"train_adaptive_targets_mode": [],
               "train_accuracy_list": [], "val_accuracy_list": [], "train_confidence_list": [],
               "val_confidence_list": [], "train_counter_list": [], "val_counter_list": [], "train_gap_list": [],
               "val_gap_list": [], "train_RMSCE": [], "val_RMSCE": [], "train_min_list": [], "val_min_list": [],
               "train_counter_zero_list": [], "val_counter_zero_list": [],
               "train_moco_err":[]}

    epoch_e_s = args.epochs
    
    control_checkpoint = False
    for root, dirs, files in os.walk(save_path):
        for f in files:
            if "STOPPED" in f:
                control_checkpoint = True
                epoch_checkpoint_aux = f.split(".")[0]
                epoch_checkpoint = epoch_checkpoint_aux.split("_")[1]

    #if control_checkpoint and input("Do you want to continue the stopped training? (y/n): ").lower() == 'y':
    if control_checkpoint:
            #epoch_checkpoint = input("Introduce your stopped epoch (number):")
            save_path_check = f"{save_path}/STOPPED_{epoch_checkpoint}.pth"
            print(save_path_check, os.path.exists(save_path_check))
            model = resume(model, save_path_check)
            save_path_check_history = f"{save_path}/STOPPED_{epoch_checkpoint}_history.csv"
            df_history = pd.read_csv(save_path_check_history)
            history = df_history.to_dict(orient="list")
            start_epoch = int(epoch_checkpoint)+1
            loss_idx_value_per_epoch = start_epoch  # Counter for the loss scalar for Tensorboard
    else:
        start_epoch = 0
        loss_idx_value_per_epoch = 0  # Counter for the loss scalar for Tensorboard
    loss_idx_value_per_batch = 0 # Counter for the loss scalar for Tensorboard

    # CSC loss
    archive = None
    if args.loss == 'csc':
        archive = moco.CSC.MoCo(
            args.moco_dim,
            args.moco_k,
            args.moco_m,
            args.moco_t,
            num_class=num_classes,
            args=args
        )

    # Training
    try:
        for epoch in range(start_epoch, args.epochs):
            if args.arch != "lcrn_normalization_new":
                adjust_learning_rate(optimizer, epoch)

            print('%s Epoch: [%d | %d] LR: %f' % (args.loss, epoch + 1, args.epochs, state['lr']))
            history["epoch"].append(epoch + 1)

            (train_loss, train_loss2, train_NLLloss, train_NLLlossCorrect, train_NLLlossIncorrect, train_l2norm, train_acc,
             train_avgIdkConfidences,train_stdIdkConfidences, train_MCE, train_ECE,
             train_beta_penalties_mean, train_beta_penalties_std, train_beta_penalties_mode,
             train_adaptive_targets_mean, train_adaptive_targets_std, train_adaptive_targets_mode,
             train_accuracy_list, train_confidence_list, train_counter_list, train_gap_list, train_RMSCE,
             train_min_list, train_counter_zero_list, moco_top1) = (
                train(trainloader=trainloader, model=model, criterion=criterion, optimizer=optimizer, epoch=epoch,
                      use_cuda=use_cuda, reward=reward, args=args, writer=writer, num_classes=num_classes,
                      loss_idx_value=loss_idx_value_per_batch,
                      model_k=model_k, archive=archive))

            writer.add_scalar("Train loss per epoch ", train_loss, loss_idx_value_per_epoch) # Tensorboard
            loss_idx_value_per_epoch+=1 # Tensorboard

            print("train_loss: ", train_loss, "train_NLLloss: ", train_NLLloss, "train_NLLlossCorrect: ", train_NLLlossCorrect,
                  "train_NLLlossIncorrect: ", train_NLLlossIncorrect, "train_MCE", train_MCE, "train_ECE", train_ECE,
                  "MOCO_TOP1", moco_top1 * 100)
            history["train_loss"].append(train_loss)
            history["train_loss2"].append(train_loss2)
            history["train_NLLloss"].append(train_NLLloss)
            history["train_NLLlossCorrect"].append(train_NLLlossCorrect)
            history["train_NLLlossIncorrect"].append(train_NLLlossIncorrect)
            history["train_l2norm"].append(train_l2norm)
            history["train_accuracy"].append(train_acc)
            history["train_avgIdkConfidences"].append(train_avgIdkConfidences)
            history["train_stdIdkConfidences"].append(train_stdIdkConfidences)
            history["train_MCE"].append(train_MCE)
            history["train_ECE"].append(train_ECE)
            history["train_beta_penalties_mean"].append(train_beta_penalties_mean)
            history["train_beta_penalties_std"].append(train_beta_penalties_std)
            history["train_beta_penalties_mode"].append(train_beta_penalties_mode)
            history["train_adaptive_targets_mean"].append(train_adaptive_targets_mean)
            history["train_adaptive_targets_std"].append(train_adaptive_targets_std)
            history["train_adaptive_targets_mode"].append(train_adaptive_targets_mode)
            history["train_accuracy_list"].append(train_accuracy_list)
            history["train_confidence_list"].append(train_confidence_list)
            history["train_counter_list"].append(train_counter_list)
            history["train_gap_list"].append(train_gap_list)
            history["train_RMSCE"].append(train_RMSCE)
            history["train_min_list"].append(train_min_list)
            history["train_counter_zero_list"].append(train_counter_zero_list)
            history["train_moco_err"].append(100-moco_top1 * 100)

            (val_loss, val_NLLloss, val_NLLlossCorrect, val_NLLlossIncorrect, val_l2norm, val_acc, val_avgIdkConfidences,
             val_stdIdkConfidences, val_MCE, val_ECE, val_accuracy_list, val_confidence_list, val_counter_list, val_gap_list,
             val_RMSCE, val_min_list, val_counter_zero_list) = test(testloader=valloader, model=model, criterion=criterion,
                                                                    epoch=epoch, use_cuda=use_cuda, reward=reward, args=args,
                                                                    writer=writer, loss_idx_value=loss_idx_value_per_batch)

            print("val_loss: ", val_loss, "val_NLLloss: ", val_NLLloss,"val_NLLlossCorrect: ", val_NLLlossCorrect,
                  "val_NLLlossIncorrect: ", val_NLLlossIncorrect, "val_l2norm: ", val_l2norm, "val_MCE", val_MCE, "val_ECE", val_ECE)
            history["val_loss"].append(val_loss)
            history["val_NLLloss"].append(val_NLLloss)
            history["val_NLLlossCorrect"].append(val_NLLlossCorrect)
            history["val_NLLlossIncorrect"].append(val_NLLlossIncorrect)
            history["val_l2norm"].append(val_l2norm)
            history["val_accuracy"].append(val_acc)
            history["val_avgIdkConfidences"].append(val_avgIdkConfidences)
            history["val_stdIdkConfidences"].append(val_stdIdkConfidences)
            history["val_MCE"].append(val_MCE)
            history["val_ECE"].append(val_ECE)
            history["val_accuracy_list"].append(val_accuracy_list)
            history["val_confidence_list"].append(val_confidence_list)
            history["val_counter_list"].append(val_counter_list)
            history["val_gap_list"].append(val_gap_list)
            history["val_RMSCE"].append(val_RMSCE)
            history["val_min_list"].append(val_min_list)
            history["val_counter_zero_list"].append(val_counter_zero_list)

            if args.arch == "lcrn_normalization_new" and reduce_lr is not None:
                reduce_lr.step(val_loss)

            # append logger file
            logger.append([epoch+1, state['lr'], train_loss, train_loss2, val_loss, train_acc, val_acc,
                           100-train_acc, 100-val_acc,
                           train_NLLloss, val_NLLloss,
                           train_NLLlossCorrect, val_NLLlossCorrect,
                           train_NLLlossIncorrect, val_NLLlossIncorrect,
                           train_l2norm, val_l2norm,
                           train_avgIdkConfidences, val_avgIdkConfidences,
                           train_stdIdkConfidences, val_stdIdkConfidences,
                           train_MCE, train_ECE, val_MCE, val_ECE,
                           100 - moco_top1 * 100
                           ])

            checkpoint(model, save_path, epoch, history)

            if args.early_stopping:
                if early_stopping is not None and early_stopping(val_loss, model.state_dict()):
                    print(
                        f"Early stopping triggered at epoch {str(epoch + 1)}, restoring weights from epoch "
                        f"{(epoch - early_stopping.patience )}")
                    epoch_e_s = epoch - early_stopping.patience
                    break

            if (epoch+1) % 25 == 0 and (epoch+1) <= 250: # Save the model every 25 epochs until 250. (Epoch+1 because my loop starts from 0. It is easier if I transpose it to 25)
                torch.save(model, f"{save_path}/seed_{args.manualSeed}_model_{model_name}_{(epoch+1)}.pth") # I only need the pth in eval
            elif args.transfer_learning and (epoch + 1) % 5 == 0:  # Save the model every 5 epochs
                    torch.save(model,
                               f"{save_path}/seed_{args.manualSeed}_model_{model_name}_{(epoch + 1)}.pth")  # I only need the pth in eval

        ######################
        ## SAVING THE MODEL ##
        ######################
        if args.early_stopping:
            model.load_state_dict(early_stopping.get_best_weights())
        if args.early_stopping:
            torch.save(model.state_dict(), f"{save_path}/seed_{args.manualSeed}_model_{model_name}_weights.pt")
            torch.save(model.state_dict(), f"{save_path}/seed_{args.manualSeed}_model_{model_name}_{epoch_e_s}.pth")
        else:
            torch.save(model, f"{save_path}/seed_{args.manualSeed}_model_{model_name}_weights.pt")
            torch.save(model, f"{save_path}/seed_{args.manualSeed}_model_{model_name}_{epoch_e_s}.pth")

        # Saving history in csv
        df_history = pd.DataFrame(history)
        df_history.to_csv(f"{save_path}/seed_{args.manualSeed}_history.csv", index=False)

        # Processing Results
        print("Processing results")

        logger.plot(['Train Loss', 'Val Loss'])
        savefig(os.path.join(save_path, 'logLoss.eps'))
        closefig()
        logger.plot(['Train Err.', 'Val Err.'])
        savefig(os.path.join(save_path, 'logErr.eps'))
        closefig()
        logger.close()

        # Create history plot, reliability diagrams and confusion matrices
        collate_model_info(model, history, save_path + "/", val_batch, valloader, labels, args, input_size=input_shape,
                           ds_type="val", use_cuda=use_cuda, num_classes=num_classes)

        print("Training Finished")

        # Copying .log
        print(f"Copying .log in {save_path}")
        log_file_path = os.getcwd()+"/log/"+save_path.split("/")[-1].split("\\")[0]+".log"
        shutil.copy(log_file_path, save_path)
    except KeyboardInterrupt:
        print("Stopping training...")


def main_ray_tune(config):  # RAY TUNE
    # print(args)

    # Ray Tune
    # if args.ray_tune:
    reward = config["reward"]
    reward_list = config["reward"]
    expected_coverage = config["expected_coverage"]
    base_path = config["base_path"]
    save_path = config["save_path"]
    resume_path = config["resume_path"]
    args_dict = config["args"]
    args = argparse.Namespace(**args_dict)
    seed = config["seed"]
    if args.loss == "focal" or args.loss == 'focalAdaptive':
        args.gamma_focal_loss = config["gamma-focal-loss"]
    elif args.loss == "csc":
        args.moco_m = config["moco-m"]
        args.moco_t = config["moco-t"]
    elif args.loss == "sat":
        args.pretrain = config["pretrain"]
        args.sat_momentum = config["sat-momentum"]
    elif args.loss == "Socrates":
        args.sat_momentum = config["sat-momentum"]
        args.gamma_focal_loss = config["gamma-focal-loss"]
        args.version = config["version"]

    #############
    ## DATASET ##
    #############
    print('==> Preparing dataset %s' % args.dataset)
    global num_classes

    trainloader, valloader, testloader, trainset, valset, testset, num_classes, train_batch, val_batch, test_batch, labels, input_size, input_shape = prepare_dataset(
        args.dataset, num_classes, args, 42, valid_size=0.1)
    if args.dataset == 'cacophony2':
        args.dataset = 'cacophony'
        print("args.dataset name changed from cacophony2 to " + args.dataset)

    #############
    ##  MODEL  ##
    #############
    print("==> creating model '{}'".format(args.arch))

    # Model instantiation
    if args.arch == "lcrn_normalization" or args.arch == "lcrn_normalization_new":
        if args.arch == "lcrn_normalization":
            model = models.__dict__[args.arch](dataset_name=args.dataset,
                                               classes=num_classes if args.loss == 'ce' or args.loss == 'focal' or args.loss == 'focalAdaptive' or args.loss == 'dece' or args.loss == 'brierScore' or args.loss == 'csc' else num_classes + 1,
                                               input_shape=input_shape)
        else:
            model = models.__dict__[args.arch](dataset_name=args.dataset,
                                               classes=num_classes if args.loss == 'ce' or args.loss == 'focal' or args.loss == 'focalAdaptive' or args.loss == 'dece' or args.loss == 'brierScore' or args.loss == 'csc' else num_classes + 1,
                                               input_shape=input_shape, dropout_1=0.45, dropout_2=0.3, l1_reg=0,
                                               l2_reg=0)
        model_name = "vgg16_cacophony"
    elif args.arch == "vit":
        model = models.__dict__[args.arch](patch_size=args.vit_patch_size, max_len=args.vit_max_len,
                                           embed_dim=args.vit_embed_dim,
                                           classes=num_classes if args.loss == 'ce' or args.loss == 'focal' or args.loss == 'focalAdaptive' or args.loss == 'dece' or args.loss == 'brierScore' or args.loss == 'csc' else num_classes + 1,
                                           layers=args.vit_layers, channels=3, heads=args.vit_heads)
        resnet_features = None
        model_name = model.name
    else:
        if args.dataset == "cacophony":
            model = models.__dict__[args.arch](dataset_name=args.dataset,
                                               classes=num_classes if args.loss == 'ce' or args.loss == 'focal' or args.loss == 'focalAdaptive' or args.loss == 'dece' or args.loss == 'brierScore' or args.loss == 'csc' else num_classes + 1)
        else:
            model = models.__dict__[args.arch](dataset_name=args.dataset,
                                               num_classes=num_classes if args.loss == 'ce' or args.loss == 'focal' or args.loss == 'focalAdaptive' or args.loss == 'dece' or args.loss == 'brierScore' or args.loss == 'csc' else num_classes + 1,
                                               input_size=input_size)
        model_name = model.name

    # Cuda
    if use_cuda:
        if args.loss == 'csc' and not args.clusterLessPower:
            model = model.cuda()
        else:
            if args.clusterLessPower:
                model = torch.nn.DataParallel(model, device_ids=list(map(int, args.gpu_id.split(','))))
                model.to(device)
            else:
                model = torch.nn.DataParallel(model.cuda())
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    ##########
    ## LOSS ##
    ##########
    # Loss/criterion selection
    if args.pretrain:  # We only use this loss if we don´t want to use any of our loss
        criterion = CrossEntropy.CrossEntropyLoss()  # nn.CrossEntropyLoss()
    if args.loss == 'ce' or args.loss == 'csc':
        criterion = CrossEntropy.CrossEntropyLoss()  # nn.CrossEntropyLoss()
    elif args.loss == 'brierScore':
        criterion = BrierScore.BrierScore()
    elif args.loss == 'gambler':
        criterion = DeepGambler.DeepGamblerLoss()
    elif args.loss == 'sat':
        criterion = SelfAdaptiveTraining.SelfAdaptiveTrainingLoss(num_examples=len(trainset), num_classes=num_classes,
                                                                  mom=args.sat_momentum)
    elif args.loss == 'dece':
        criterion = DECE.DECE(15, 100, 0.01)
    elif args.loss == 'focal':
        criterion = Focal.FocalLoss(gamma=args.gamma_focal_loss, alpha=args.alpha_focal_loss, size_average=True)
    elif args.loss == 'focalAdaptive':
        criterion = FocalAdaptiveGamma.FocalLossAdaptive(gamma=args.gamma_focal_loss, alpha=args.alpha_focal_loss, size_average=True)
    elif args.loss == 'Socrates':
        criterion = Socrates.SocratesLoss(num_examples=len(trainset), num_classes=num_classes,
                                          mom=args.sat_momentum,
                                          gamma=args.gamma_focal_loss,
                                          alpha=args.alpha_focal_loss,
                                          dynamic=args.dynamic,
                                          loss_name=args.loss,
                                          version=args.version,
                                          version_SAT_original=args.version_SAT_original,
                                          version_FOCALinGT=args.version_FOCALinGT,
                                          version_FOCALinSAT=args.version_FOCALinSAT,
                                          version_changingWithIdk=args.version_changingWithIdk,
                                          old=args.old
                                          )

    # CSC Loss
    # encoder_k
    model_k = None
    if args.loss == 'csc':
        model_k = copy.deepcopy(model)

        if args.arch == 'vgg16_bn' or args.arch == "vit":
            if args.arch == "vit":
                if args.transfer_learning:
                    print(model.classifier.register_forward_hook(hook_fn))
                    print(model_k.classifier.register_forward_hook(hook_fn_k))
                else:
                    if args.clusterLessPower:
                        print(model.module.classification_head.fc2.register_forward_hook(hook_fn))
                        print(model_k.module.classification_head.fc2.register_forward_hook(hook_fn_k))
                    else:
                        print(model.classification_head.fc2.register_forward_hook(hook_fn))
                        print(model_k.classification_head.fc2.register_forward_hook(hook_fn_k))
            elif args.clusterLessPower:
                print(list(model.module.classifier._modules.items())[2][1].register_forward_hook(hook_fn))
                print(list(model_k.module.classifier._modules.items())[2][1].register_forward_hook(hook_fn_k))
            else:
                print(list(model.classifier._modules.items())[2][1].register_forward_hook(hook_fn))
                print(list(model_k.classifier._modules.items())[2][1].register_forward_hook(hook_fn_k))
            if args.dataset == 'food101':
                args.moco_dim = 4096
            elif args.dataset == 'cifar100':
                if args.arch == "resnet110":
                    args.moco_dim = 2048
                elif args.arch == "vit":
                    args.moco_dim = 100
                else:
                    args.moco_dim = 512
            else:
                args.moco_dim = 512

        elif args.arch == 'resnet34' or args.arch == 'resnet18' or args.arch == "resnet110":
            if args.clusterLessPower:
                print(model.module.avgpool.register_forward_hook(hook_fn))
                print(model_k.module.avgpool.register_forward_hook(hook_fn_k))
            else:
                print(model.avgpool.register_forward_hook(hook_fn))
                print(model_k.avgpool.register_forward_hook(hook_fn_k))
            if args.dataset == 'food101':
                args.moco_dim = 25088
            elif args.arch == "resnet110":
                args.moco_dim = 2048
            else:
                args.moco_dim = 512



    ##############
    ## TRAINING ##
    ##############

    # Optimizer
    if args.arch == "lcrn_normalization_new":
        optimizer = optim.Adam(model.parameters(), lr=2.6e-4, weight_decay=0)
        reduce_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.25, threshold=0.0001,
                                                         min_lr=1e-8,
                                                         threshold_mode="abs")
    elif args.dataset == "celebAadam":
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        args.dataset = "celebA"
    else:
        optimizer = optim.SGD(model.parameters(), lr=state['lr'], momentum=args.momentum,
                              weight_decay=args.weight_decay)

    # Early Stopping
    if args.early_stopping:
        if args.arch == "lcrn_normalization_new":
            early_stopping = MetricMonitor(10, mode="max")
        else:
            early_stopping = MetricMonitor(13, mode="min")

    epoch_e_s = args.epochs
    start_epoch = 0
    loss_idx_value_per_batch = 0  # Counter for the loss scalar for Tensorboard

    # CSC loss
    archive = None
    if args.loss == 'csc':
        archive = moco.CSC.MoCo(
            args.moco_dim,
            args.moco_k,
            args.moco_m,
            args.moco_t,
            num_class=num_classes,
            args=args
        )
    writer = SummaryWriter()

    # Training
    try:
        for epoch in range(start_epoch, args.epochs):
            if args.arch != "lcrn_normalization_new":
                adjust_learning_rate(optimizer, epoch)


            (train_loss, train_loss2, train_NLLloss, train_NLLlossCorrect, train_NLLlossIncorrect, train_l2norm,
             train_acc,
             train_avgIdkConfidences, train_stdIdkConfidences, train_MCE, train_ECE,
             train_beta_penalties_mean, train_beta_penalties_std, train_beta_penalties_mode,
             train_adaptive_targets_mean, train_adaptive_targets_std, train_adaptive_targets_mode,
             train_accuracy_list, train_confidence_list, train_counter_list, train_gap_list, train_RMSCE,
             train_min_list, train_counter_zero_list, moco_top1) = (
                train(trainloader=trainloader, model=model, criterion=criterion, optimizer=optimizer, epoch=epoch,
                      use_cuda=use_cuda, reward=reward, args=args, writer=writer, num_classes=num_classes,
                      loss_idx_value=loss_idx_value_per_batch,
                      model_k=model_k, archive=archive))



            (
            val_loss, val_NLLloss, val_NLLlossCorrect, val_NLLlossIncorrect, val_l2norm, val_acc, val_avgIdkConfidences,
            val_stdIdkConfidences, val_MCE, val_ECE, val_accuracy_list, val_confidence_list, val_counter_list,
            val_gap_list,
            val_RMSCE, val_min_list, val_counter_zero_list) = test(testloader=valloader, model=model,
                                                                   criterion=criterion,
                                                                   epoch=epoch, use_cuda=use_cuda, reward=reward,
                                                                   args=args,
                                                                   writer=writer,
                                                                   loss_idx_value=loss_idx_value_per_batch)

            if args.arch == "lcrn_normalization_new" and reduce_lr is not None:
                reduce_lr.step(val_loss)

            #checkpoint(model, save_path, epoch)

            if args.early_stopping:
                if early_stopping is not None and early_stopping(val_loss, model.state_dict()):
                    print(
                        f"Early stopping triggered at epoch {str(epoch + 1)}, restoring weights from epoch "
                        f"{(epoch - early_stopping.patience)}")
                    epoch_e_s = epoch - early_stopping.patience
                    break

            if args.ray_tune:
                checkpoint_data = {
                    "epoch": epoch,
                    "net_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                with tempfile.TemporaryDirectory() as checkpoint_dir:
                    data_path = Path(checkpoint_dir) / "data.pkl"
                    with open(data_path, "wb") as fp:
                        pickle.dump(checkpoint_data, fp)

                    checkpoint = Checkpoint.from_directory(checkpoint_dir)
                    combined_metric = val_acc - val_loss - val_ECE
                    ray.train.report({"val_accuracy": val_acc, "val_ECE": val_ECE, "train_accuracy": train_acc, "train_ECE": train_ECE, "val_loss": val_loss, "train_loss": train_loss,    "val_combined_metric":combined_metric},
                                     checkpoint=checkpoint)
        ######################
        ## SAVING THE MODEL ##
        ######################
        if args.early_stopping:
            model.load_state_dict(early_stopping.get_best_weights())
        if args.early_stopping:
            torch.save(model.state_dict(), f"{save_path}/seed_{args.manualSeed}_model_{model_name}_weights.pt")
            torch.save(model.state_dict(), f"{save_path}/seed_{args.manualSeed}_model_{model_name}_{epoch_e_s}.pth")
        else:
            torch.save(model, f"{save_path}/seed_{args.manualSeed}_model_{model_name}_weights.pt")
            torch.save(model, f"{save_path}/seed_{args.manualSeed}_model_{model_name}_{epoch_e_s}.pth")


    except KeyboardInterrupt:
        print("Stopping training...")


if __name__ == '__main__':
    # Paths for folder structure
    if args.dataset == "celebAadam":
        base_path = os.path.join(args.save, "celebA", args.arch)
    else:
        base_path = os.path.join(args.save, args.dataset, args.arch)
    resume_path = args.eval_path

    # Set the abstention definitions
    expected_coverage = args.coverage
    reward_list = args.rewards  # For deepgamblers

    #LR:
    baseLR = state['lr']

    # Seeds:
    if args.manualSeed is None:
        seeds = [i for i in range(0, 5)]
    else:
        seeds = [args.manualSeed]

    for s in seeds:
        # Seeds
        args.manualSeed = s
        np.random.seed(args.manualSeed)
        random.seed(args.manualSeed)
        torch.manual_seed(args.manualSeed)
        if use_cuda:
            torch.cuda.manual_seed_all(args.manualSeed)

        # Reward list is equal or longer than 1 in Gambler loss
        # Rest of the loss is equal to 1, rewards are not used by these losses
        for i in range(len(reward_list)):
            state['lr'] = baseLR
            reward = reward_list[i]

            save_path = base_path + 'o{:.2f}'.format(reward)+ "_" + str(s)
            # Make path for the current architecture
            if not resume_path and not os.path.isdir(save_path):
                mkdir_p(save_path)

            # Gambler Loss: default the pretraining epochs to 100 to reproduce the results in the paper
            if args.loss == 'gambler' and args.pretrain == 0:
                if args.dataset == 'cifar10' and reward < 6.3:
                    args.pretrain = 100
                elif args.dataset == 'svhn' and reward < 6.0:
                    args.pretrain = 50

            if args.loss == 'csc':
                full_k1 = False
                full_k2 = False
            else:
                full_k1 = True
                full_k2 = True

            ## IF RAY TUNE
            if args.ray_tune:
                ray.init(_temp_dir=f"/data/{SET_USER}/tmp/ray_tmp/") # change this
                config = {
                    "reward": reward,
                    "reward_list": reward_list,
                    "expected_coverage": expected_coverage,
                    "base_path": base_path,
                    "save_path": save_path,
                    "resume_path": resume_path,
                    "args": vars(args),
                    "seed": s,
                }
                if args.loss == "focal" or args.loss == 'focalAdaptive':
                    config["gamma-focal-loss"] = tune.grid_search([1, 2, 3])
                elif args.loss == "csc":
                    config["moco-m"] = tune.grid_search([0.9,0.99,0.999])
                    config["moco-t"] = tune.grid_search([0.1,0.5,1])
                elif args.loss == "sat":
                    #config["pretrain"] = tune.grid_search([0, 50, 100, 150, 200])
                    config["pretrain"] = tune.grid_search([50, 100, 150, 200])
                    config["sat-momentum"] = tune.grid_search([0.9,0.99,0.999])
                elif args.loss == "Socrates":
                    config["gamma-focal-loss"] = tune.grid_search([1, 2, 3])
                    config["sat-momentum"] = tune.grid_search([0.9, 0.99, 0.999])
                    config["version"] = tune.grid_search([1])
                gpus_per_trial = 1
                cpus_per_trial = 30
                scheduler = ASHAScheduler(
                    metric="val_ECE",
                    mode="min",
                    max_t=300, # Max epochs
                    grace_period=150, # This is for the early stopping
                    reduction_factor=2,
                )
                result = tune.run(
                    main_ray_tune,
                    resources_per_trial={"cpu":cpus_per_trial, "gpu": gpus_per_trial},
                    config=config,
                    #num_samples=2, # Only if we don´t use grid search
                    scheduler=scheduler,
                    storage_path=f"/data/{SET_USER}/ray_results", # change this
                    #checkpoint_at_end=True,
                    #resume="AUTO"
                )

                best_trial = result.get_best_trial("val_ECE", "min", "last")
                print(f"Best trial config: {best_trial.config}")
                print(f"Best trial final validation combined metric (Acc-ECE-loss): {best_trial.last_result['val_combined_metric']}")
                print(f"Best trial final validation loss: {best_trial.last_result['val_loss']}")
                print(f"Best trial final validation ECE: {best_trial.last_result['val_ECE']}")
                print(f"Best trial final validation accuracy: {best_trial.last_result['val_accuracy']}")
                print(f"Best trial final training loss: {best_trial.last_result['train_loss']}")
                print(f"Best trial final training ECE: {best_trial.last_result['train_ECE']}")
                print(f"Best trial final training accuracy: {best_trial.last_result['train_accuracy']}")
            else:
                main(reward, reward_list, expected_coverage, base_path, save_path, resume_path, args, s)
            #cProfile.run('main(reward, reward_list, expected_coverage, base_path, save_path, resume_path, args, s)')