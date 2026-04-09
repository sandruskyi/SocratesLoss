"""
Extended by
@User: sandruskyi
"""
import time
from utils import Bar, AverageMeter, accuracy
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import math
import os
import numpy as np
import pandas as pd
from utils.useful_functions import collate_model_info
from matplotlib import pyplot as plt
from PIL import Image
from utils.bisection import bisection_method, bisection_method_LeoFeng
import shutil
from calibration.plot_reliability_diagram import calculate_confidence_accuracy_multiclass
from utils.risk_coverage_curve import risk_coverage_curve_plotly
from losses import DeepGambler, SelfAdaptiveTraining, CrossEntropy, Focal, Socrates
import utils.hidden_features_var as hidden_features_var
import utils.hidden_features_k_var as hidden_features_k_var
from sklearn.metrics.pairwise import cosine_similarity


__all__ = ['train', 'test', 'evaluate', 'eval_clean', 'eval_converage']

def train(trainloader, model, criterion, optimizer, epoch, use_cuda, reward, args, writer, num_classes, full_k1, full_k2, loss_idx_value=0,  model_k=None, archive=None):

    # switch to train mode
    model.train()

    #CSC loss
    if args.loss == 'csc':
        model_k.train()
        if args.arch == 'vgg16_bn':
            feature_extractor = model.features
        if epoch == args.pretrain:
            for param_q, param_k in zip( model.parameters(), model_k.parameters()):
                param_k.data = param_q.data
        if epoch > args.pretrain:
            for param_q, param_k in zip(model.parameters(), model_k.parameters()):
                param_k.data = param_k.data * args.moco_m + param_q.data * (1.0 - args.moco_m)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses2 = AverageMeter() #csc
    NLLlosses = AverageMeter()
    NLLlosses_correct = AverageMeter()
    NLLlosses_incorrect = AverageMeter()
    avgIdkConfidences = AverageMeter()
    l2norms = AverageMeter()
    top1 = AverageMeter() # Accuracy top 1
    top5 = AverageMeter() # Accuracy top 5
    moco_top1 = AverageMeter()
    train_dataset_calculate_calibration_with_idk = {'y_gt':[],'y_train_class_predicted':[],'y_train_confidences_predicted_argmax':[]}

    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    current_reward = reward
    for batch_idx,  batch_data in tqdm(enumerate(trainloader), total=len(trainloader)):

        inputs, targets, indices = batch_data
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            if args.clusterLessPower:
                # device = torch.device(f'cuda' if use_cuda else 'cpu')
                # inputs, targets, indices = inputs.to(device), targets.to(device),  indices.to(device)
                primary_device = list(map(int, args.gpu_id.split(',')))[0]
                inputs, targets, indices = inputs.to(f'cuda:{primary_device}'), targets.to(f'cuda:{primary_device}'),  indices.to(f'cuda:{primary_device}')
            else:
                inputs, targets, indices = inputs.cuda(), targets.cuda(), indices.cuda()

        # inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets) # For pytorch version > 0.4.0 this is not neccesary
        targets = targets.to(torch.int64) # For pytorch version > 0.4.0 this is not neccesary

        # Tensorboard
        if args.dataset!="cacophony" and batch_idx==0:
            writer.add_image("Training - Example image input at batch 0 of each epoch ", inputs[0], global_step=epoch)
            writer.add_scalar("Training - Example image label at batch 0 of each epoch ", targets[0], global_step=epoch)

        # We calculate the outputs:
        outputs_logits = model(inputs)

        if args.loss == 'csc':
            with torch.no_grad():
                outputs_k = model_k(inputs)
            if args.arch == 'vgg16_bn':
                outputs_feature = feature_extractor(inputs)
                outputs_feature = outputs_feature.view(outputs_feature.size(0), -1)
                outputs_projection = nn.Sequential(*list(model.classifier.children())[:3])(outputs_feature)
                # print('new feature: ', outputs_projection, outputs_projection.shape)
                outputs_logits = nn.Sequential(*list(model.classifier.children())[3:])(outputs_projection)

        if epoch>=args.pretrain and (args.loss == 'csc' or args.loss =='csc_entropy' or args.loss =='csc_sat_entropy') and full_k1 and full_k2:
            if args.arch != 'vgg16_bn':
                temp_full_k1, temp_full_k2, moco_error, loss2 = archive(torch.flatten(hidden_features_var.get_hidden_features(), 1),
                                                                        torch.flatten(hidden_features_k_var.get_hidden_features_k(), 1),
                                                                        targets, outputs_logits, outputs_k, epoch + 1,
                                                                        args.pretrain, full_k1 and full_k2)
            else:
                temp_full_k1, temp_full_k2, moco_error, loss2 = archive(outputs_projection, hidden_features_k_var.get_hidden_features_k(),
                                                                        targets, outputs_logits, outputs_k, epoch + 1,
                                                                        args.pretrain, full_k1 and full_k2)
            full_k1 = full_k1 or temp_full_k1
            full_k2 = full_k2 or temp_full_k2

            if args.loss == 'csc':
                if full_k1 and full_k2:
                    losses2.update(loss2.item(), inputs.size(0))
                    moco_top1.update(moco_error, inputs.size(0))
                    # print(loss2)
                    loss = criterion(outputs_logits, targets) + loss2 * current_reward
                else:
                    loss = F.cross_entropy(outputs_logits, targets)
            elif args.loss == 'csc_entropy':
                if full_k1 and full_k2:
                    softmax = nn.Softmax(-1)
                    losses2.update(loss2.item(), inputs.size(0))
                    moco_top1.update(moco_error, inputs.size(0))
                    # print(loss2)
                    loss = criterion(outputs_logits, targets) + loss2 * current_reward + (
                            args.entropy * (-softmax(outputs_logits) * outputs_logits).sum(-1)).mean()
                else:
                    loss = F.cross_entropy(outputs_logits, targets)
        elif epoch>=args.pretrain and (args.loss != 'csc' and args.loss != 'csc_entropy' and args.loss != 'csc_sat_entropy'):
            if args.loss == 'gambler':
                loss = criterion(outputs_logits, targets, reward)
            elif args.loss == 'sat':
                loss = criterion(outputs_logits, targets, indices)
            elif args.loss == 'focal':
                loss = criterion(outputs_logits, targets)
            elif args.loss == 'Socrates':
                loss = criterion(outputs_logits, targets, indices, debug=args.debug)
            else:
                loss = criterion(outputs_logits, targets)
        else: # ce, Focal and Socrates (es=0) must be always pretrain = 0
            if args.loss == 'csc' or 'csc_entropy' or 'csc_sat_entropy':
                moco_error = 1 / num_classes
                if args.arch != 'vgg16_bn' and epoch >= args.pretrain:
                    temp_full_k1, temp_full_k2, moco_error = archive(torch.flatten(hidden_features_var.get_hidden_features(), 1),
                                                                     torch.flatten(hidden_features_k_var.get_hidden_features_k(), 1), targets,
                                                                     outputs_logits, outputs_k, epoch + 1, args.pretrain,
                                                                     full_k1 and full_k2)
                else:
                    if epoch >= args.pretrain:
                        temp_full_k1, temp_full_k2, moco_error = archive(outputs_projection, hidden_features_k_var.get_hidden_features_k(), targets,
                                                                         outputs_logits, outputs_k, epoch + 1, args.pretrain,
                                                                         full_k1 and full_k2)
                    else:
                        temp_full_k1, temp_full_k2 = False, False # Initialization
                full_k1 = full_k1 or temp_full_k1
                full_k2 = full_k2 or temp_full_k2
                loss = F.cross_entropy(outputs_logits, targets)
                moco_top1.update(moco_error, inputs.size(0))
            elif args.loss == 'ce':
                loss = F.cross_entropy(outputs_logits, targets)
            elif  args.loss == 'focal':  # Put focal with pretrain 0
                loss = criterion(outputs_logits, targets)
            elif args.loss == 'Socrates' and not args.first_epochs_ce:
                criterion2 = Focal.FocalLoss(gamma=args.gamma_focal_loss, alpha=args.alpha_focal_loss, size_average=True)
                loss = criterion2(outputs_logits[:, :-1], targets)
            else: # SAT and others
                loss = F.cross_entropy(outputs_logits[:, :-1], targets) # SAT: As in the initial epochs we take into account only the gt (that is 1) as t, it is the same than only to do the cross_entropy, because for the idk class prediction (1-t)=(1-1)=0

        # We calculate the rest of values for future graph evaluation
        with torch.no_grad():

            NLL_loss = F.cross_entropy(outputs_logits, targets)

            confidences_outputs_with_idk = F.softmax(outputs_logits, dim=1).cpu().detach().numpy()

            class_predicted_with_idk = np.argmax(confidences_outputs_with_idk, axis=1) # get indices of the maximum values with idk class. In this case the indices are the same than the class
            predicted_class_confidences_with_idk = confidences_outputs_with_idk[np.arange(len(class_predicted_with_idk)), class_predicted_with_idk] # List with the confidences of the argmax softmax
            avg_idk_class_confidences = confidences_outputs_with_idk[np.arange(len(class_predicted_with_idk)), -1].mean()

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
            if args.dataset != 'catsdogs' and args.dataset != 'cacophony' and args.dataset!="celebA":
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

            writer.add_scalar("Loss per batch ", loss.item(), loss_idx_value) # Tensorboard
            loss_idx_value+=1 # Tensorboard

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # We calculate the l2 norm of the weights in the final layer for future graphs
            if not args.clusterLessPower and (args.loss == 'csc' or args.loss =='csc_entropy' or args.loss =='csc_sat_entropy'):
                model_part = model
            else:
                model_part = model.module
            if args.arch == 'vgg16_bn':
                last_layer_of_weights = model_part.classifier[-1].weight
            elif args.arch == 'resnet50' or args.arch == 'resnet34':
                last_layer_of_weights = model_part.fc.weight
            elif args.arch == 'lcrn_normalization':
                last_layer_of_weights = model_part.linear.weight
            else:
                raise ValueError(
                    f"You are using an arch different to vgg16_bn or resnet50 or resnet34 : {args.arch}. In this case create a new if your your last_layer_weights.")
                sys.exit()
            l2norm = torch.norm(last_layer_of_weights, p=2)
            l2norms.update(l2norm.item(), inputs.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} |  Loss2: {loss2:.4f} | NLLloss: {NLLloss:.4f} | NLLlossCorrect: {NLLlossCorrect:.4f} | NLLlossIncorrect: {NLLlossIncorrect:.4f} | l2norm: {l2norm:.4f} | top1: {top1: .4f} | top5: {top5: .4f} | idkConfidencesAvg: {idkConfidencesAvg: .4f} | idkConfidencesStd: {idkConfidencesStd: .4f}'.format(
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
        train_dataset_calculate_calibration_with_idk["y_gt"], train_dataset_calculate_calibration_with_idk["y_train_class_predicted"], train_dataset_calculate_calibration_with_idk["y_train_confidences_predicted_argmax"], 10)

    return (losses.avg, losses2.avg, NLLlosses.avg, NLLlosses_correct.avg, NLLlosses_incorrect.avg, l2norms.avg, top1.avg, avgIdkConfidences.avg, avgIdkConfidences.std, MCE_with_idk, ECE_with_idk, accuracy_list_with_idk, confidence_list_with_idk, counter_list_with_idk, gap_list_with_idk, RMSCE_with_idk, min_list_with_idk, counter_zero_list_with_idk, moco_top1.avg)


def test(testloader, model, criterion, epoch, use_cuda, reward, args, writer=None, loss_idx_value=0):
    global best_acc

    # Switch to eval mode
    model.eval()

    # whether to evaluate uncertainty, or confidence
    #if evaluation:
    #    evaluate(testloader, model, use_cuda, args, expected_coverage, reward_list, base_path)
        # eval_clean(model, 'cuda', testloader)
    #    return

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
    test_dataset_calculate_calibration_with_idk = {'y_gt':[],'y_test_class_predicted':[],'y_test_confidences_predicted_argmax':[]}


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

        #inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets) # For pytorch version > 0.4.0 this is not neccesary
        targets = targets.to(torch.int64)

        # Tensorboard
        if args.dataset!="cacophony" and batch_idx == 0:
            writer.add_image("Dev - Example image input at batch 0 of each epoch ", inputs[0], global_step=epoch)
            writer.add_scalar("Dev - Example image label at batch 0 of each epoch ", targets[0], global_step=epoch)

        # compute output
        with torch.no_grad():
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
                elif args.loss == 'Socrates':
                    criterion2 = Focal.FocalLoss(gamma=args.gamma_focal_loss, alpha=args.alpha_focal_loss,
                                                 size_average=True)
                    loss = criterion2(outputs_logits[:, :-1], targets)
                else: # Ce, focal...
                    loss = criterion(outputs_logits, targets)


                # analyze the accuracy at different abstention level
                # Only after pretrain because in pretrain
                outputs_abstention = F.softmax(outputs_logits, dim=1)
                # Esto lo he cambiado en comparaci'on al original!!!
                values, predictions = outputs_abstention.data.max(1)  # Return 2 elems, first the max values along classes axis, second a tensor with the indices (the class index) of the max values
                if args.loss == 'ce' or args.loss == 'focal' or args.loss == 'csc' or args.loss == 'csc_entropy':
                    outputs, reservation = outputs_abstention, (outputs_abstention * torch.log(outputs_abstention)).sum(-1)  # Reservation is neg. entropy here.
                    pred_logits = F.softmax(outputs_logits, -1) # For the sr_results
                else:
                    outputs, reservation = outputs_abstention[:,:-1], outputs_abstention[:,-1]
                    pred_logits = F.softmax(outputs_logits[:, :-1], -1) # For the sr_results
                abstention_results.extend(zip(list( reservation.cpu().numpy() ),list( predictions.eq(targets.data).cpu().numpy() )))
                sr_results.extend(zip(list(pred_logits.max(-1)[0].cpu().numpy()), list( predictions.eq(targets.data).cpu().numpy() )))
            else:
                if args.loss == 'ce' or args.loss == 'csc' or args.loss == 'csc_entropy':
                    loss = F.cross_entropy(outputs_logits, targets)
                elif args.loss == 'focal':  # Put focal with pretrain 0
                    loss = criterion(outputs_logits, targets)
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

            class_predicted_with_idk = np.argmax(confidences_outputs_with_idk , axis=1)  # get indices of the maximum values without idk class. In this case the indices are the same than the class
            predicted_class_confidences_with_idk = confidences_outputs_with_idk[np.arange(len(class_predicted_with_idk)), class_predicted_with_idk]  # List with the confidences of the argmax softmax
            avg_idk_class_confidences = confidences_outputs_with_idk[np.arange(len(class_predicted_with_idk)), -1].mean()

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
            if not args.clusterLessPower and (args.loss == 'csc' or args.loss =='csc_entropy' or args.loss =='csc_sat_entropy'):
                model_part = model
            else:
                model_part = model.module
            if args.arch == 'vgg16_bn':
                last_layer_of_weights = model_part.classifier[-1].weight
            elif args.arch == 'resnet50' or args.arch == 'resnet34':
                last_layer_of_weights = model_part.fc.weight
            elif args.arch == 'lcrn_normalization':
                last_layer_of_weights = model_part.linear.weight
            else:
                raise ValueError(f"You are using an arch different to vgg16_bn or resnet50 or resnet34: {args.arch}. In this case create a new if your your last_layer_weights.")
                sys.exit()
            l2norm = torch.norm(last_layer_of_weights, p=2)

            # measure accuracy and record loss
            # To methods with the idk class, the accuracy and losses are calculated without the idk class
            # But, it is notable that the output were calculated with the softmax taking into account the idk class
            outputs_pred = F.softmax(outputs_logits, dim=1)
            if args.dataset != 'catsdogs' and args.dataset != 'cacophony' and args.dataset!="celebA":
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
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | NLLloss: {NLLloss:.4f} | NLLlossCorrect: {NLLlossCorrect:.4f} | NLLlossIncorrect: {NLLlossIncorrect:.4f} |  l2norm: {l2norm:.4f} | top1: {top1: .4f} | top5: {top5: .4f}| idkConfidencesAvg: {idkConfidencesAvg: .4f} | idkConfidencesStd: {idkConfidencesStd: .4f}'.format(
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


    # Abstention results. We only take into account the abstention class when epoch >= pretrain,
    # because it is the only case that we train with the idk class
    """ # It doesn't make sense to calculate the abstention results per epoch
    
    if epoch >= args.pretrain:
    	# sort the abstention results according to their reservations, from high to low
        abstention_results.sort(key = lambda x: x[0], reverse=True) # We sort by reservations
        # get the "correct or not" list for the sorted results
        # Map applies a function for each element (in this case if correct True or incorrect False) of abstention result
        sorted_correct = list(map(lambda x: int(x[1]), abstention_results))
        size = len(sorted_correct)
        #print('Abstention Logit: accuracy of coverage ',end='')
        #for coverage in expected_coverage:
        #    print('{:.0f}: {:.3f}, '.format(coverage, sum(sorted_correct[int(size/100*coverage):])),end='')
        #print('')

        # sort the abstention results according to Softmax Response scores, from high to low
        sr_results.sort(key=lambda x: -x[0])
        # get the "correct or not" list for the sorted results
        sorted_correct = list(map(lambda x: int(x[1]), sr_results))
        size = len(sorted_correct)
        #print('Softmax Response: accuracy of coverage ', end='')
        #for coverage in expected_coverage:
        #    covered_correct = sorted_correct[:round(size / 100 * coverage)]
        #    print('{:.0f}: {:.3f}, '.format(coverage, sum(covered_correct) / len(covered_correct) * 100.), end='')
        #print('')
    """

    # Reliability Diagram
    accuracy_list_with_idk, confidence_list_with_idk, counter_list_with_idk, gap_list_with_idk, MCE_with_idk, ECE_with_idk, RMSCE_with_idk, min_list_with_idk, counter_zero_list_with_idk = calculate_confidence_accuracy_multiclass(
        test_dataset_calculate_calibration_with_idk["y_gt"], test_dataset_calculate_calibration_with_idk["y_test_class_predicted"],
        test_dataset_calculate_calibration_with_idk["y_test_confidences_predicted_argmax"], 10)
    return (losses.avg, NLLlosses.avg, NLLlosses_correct.avg, NLLlosses_incorrect.avg, l2norms.avg, top1.avg, avgIdkConfidences.avg, avgIdkConfidences.std, MCE_with_idk, ECE_with_idk, accuracy_list_with_idk, confidence_list_with_idk, counter_list_with_idk, gap_list_with_idk, RMSCE_with_idk, min_list_with_idk, counter_zero_list_with_idk)




# this function is used to organize all data and write into one file
def save_data(results_valid, results, reward_list, save_path):

    for reward in reward_list:
        save = open(os.path.join(save_path, 'coverage_vs_err.csv'), 'w')
        save.write('0,100val.,100test,99v,99t,98v,98t,97v,97t,95v,95t,90v,90t,85v,85t,80v,80t,75v,75t,70v,70t,60v,60t,50v,50t,40v,40t,30v,30t,20v,20t,10v,10t\n')
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
            save.write('{:.2f},\t\t{:.3f},\t\t{:.3f}\n'.format( results_valid[idx][0] * 100.,
                                                      (1 - results_valid[idx][1]) * 100, results_valid[idx][2]))


        save.write('\n')
        save.close()


def save_data_all_results(type, abortion_results, abortion_results_LeoFeng, sr_results, expected_coverage, reward_list, save_path):
    for reward in reward_list:
        save = open(os.path.join(save_path, 'coverage_vs_err.csv'), 'w')
        save.write("Results with test set\n")
        save.write("abortion_results\n")
        save.write("AbstentionLogitTest,Coverage,Error,Threshold\n")
        for idx, _ in enumerate(abortion_results):
            save.write(
                '{:.0f},\t{:.2f},\t\t{:.3f},\t\t{:.3f}\n'.format(expected_coverage[idx], abortion_results[idx][0] * 100.,
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
            save.write('{:.0f},\t{:.2f},\t\t{:.3f},\t\t{:.3f}\n'.format(expected_coverage[idx], sr_results[idx][0] * 100.,
                                                                 (1 - sr_results[idx][1]) * 100, sr_results[idx][2]))



        save.write('\n')
        save.close()
def evaluate(model, use_cuda, args, expected_coverage, reward_list, base_path, save_path, trainloader = None, valloader = None, testloader = None, train_batch = None, val_batch = None, test_batch = None, labels = None , input_size=(45, 3, 24, 24), num_classes=2):
    # Switch to eval mode
    model.eval()

    types_collate = {"test":[test_batch, testloader], "val":[val_batch, valloader]}
    folder_path = save_path + "/evaluation/"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)


    # Selective Classification:
    print("SELECTIVE CLASSIFICATION")
    abortion_results = [[],[]]
    sr_results = [[], []]

    feature = []


    # New code CSC-CL
    sim_abortion = [[], []]
    nce_abortion = [[], []]
    sim_dot_product_abortion = [[], []]
    prediction_list = []
    abortion_results_valid = [[], []]
    sr_results_valid = [[], []]

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(testloader):
            inputs, targets = batch_data[:2] # inputs, targets, indices = batch_data

            targets = targets.to(torch.int64)
            if use_cuda:
                if args.clusterLessPower:
                    primary_device = list(map(int, args.gpu_id.split(',')))[0]
                    inputs, targets = inputs.to(f'cuda:{primary_device}'), targets.to(f'cuda:{primary_device}')
                #if args.clusterLessPower:
                #    inputs = inputs.to(f'cuda:{int(args.gpu_id[0])}')
                #    targets = targets.to(f'cuda:{int(args.gpu_id[0])}')
                #else:
                #    inputs = inputs.to(f'cuda:{int(args.gpu_id)}')
                #    targets = targets.to(f'cuda:{int(args.gpu_id)}')
                else:
                    inputs, targets = inputs.cuda(), targets.cuda()

            #print(inputs.device, targets.device)
            #for m in model.module.parameters():
            #    print("Modulo:", m.device)
            output = model(inputs)
            output_logits = output.detach().clone()

            if args.loss == 'csc' or args.loss == 'csc_entropy':
                if args.arch != 'vgg16_bn':
                    feature.extend(list(torch.flatten(hidden_features_var.get_hidden_features(), 1)))
                else:
                    feature.extend(list(hidden_features_var.get_hidden_features()))

            if args.loss == 'ce' or args.loss == 'focal' or args.loss == 'csc' or args.loss == 'csc_entropy':
                pred_logits = F.softmax(output, -1)  # to calculate the SR
            else:
                pred_logits = F.softmax(output[:, :-1], -1) # to calculate the SR
            output = F.softmax(output,dim=1) # To calculate the other methods (SAT, deepgamblers...)
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
            elif args.loss == 'ce' or args.loss == 'focal':
                reservation = 1 - output.data.max(1)[0]
            else:
                output, reservation = output[:,:-1], (output[:,-1])
            values, predictions = output.data.max(1) # Get the prediction values and the prediction positions (i.e., the class)
            prediction_list.extend(list(predictions))
            abortion_results[0].extend(list( reservation.cpu() ))
            abortion_results[1].extend(list( predictions.eq(targets.data).cpu()))
            sr_results[0].extend(list(-pred_logits.max(-1)[0].cpu()))
            sr_results[1].extend(list(predictions.eq(targets.data).cpu()))

    # As we have our own validation set, it is not necessary to divide the set in two, then comment:
    """
    def shuffle_list(lst, seed=10):
        random.seed(seed)
        random.shuffle(lst)
    shuffle_list(abortion_results[0]); shuffle_list(abortion_results[1])
    """

    #################################################################
    ######## CODE FROM CCL-SC TO EVALUATE SELECTIVE CLASSIFICATION:
    #################################################################
    print("STARTING EVALUATION CCL-SC")
    # valid & test
    correct_list = sr_results[1]
    num_valid = 1000

    def get_feature_avg(correct_list, prediction_list, features):
        data = [(feature, prediction, correct) for feature, prediction, correct in
                zip(features, prediction_list, correct_list)]

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

    abortion_results[0] = abortion_results[0][num_valid:]
    abortion_results[1] = abortion_results[1][num_valid:]
    sr_results[0] = sr_results[0][num_valid:]
    sr_results[1] = sr_results[1][num_valid:]

    #data_sim = {'sim': sim_abortion[0].cpu().tolist(), 'sr': [item.item() for item in sr_results[0].cpu()],
    #            'true': [int(item) for item in sim_abortion[1].cpu()], 'predict': prediction_list[num_valid:].cpu()}
    #df = pd.DataFrame(data_sim)

    # df.to_excel('output.xlsx', index=False)

    abortion_scores, abortion_correct = torch.stack(abortion_results[0]).cpu(), torch.stack(abortion_results[1]).cpu()
    print(abortion_scores.shape)
    # sr_scores, sr_correct = torch.stack(abortion_results[0]), torch.stack(abortion_results[1])
    # sr_scores, sr_correct = torch.stack(sr_results[0]), torch.stack(sr_results[1])
    sr_scores, sr_correct = torch.stack(sr_results[0]).cpu(), torch.stack(sr_results[1]).cpu()
    print(sr_scores.shape)
    # Abstention Logit Results
    abortion_results = []
    bisection_method_LeoFeng(abortion_scores, abortion_correct, abortion_results, expected_coverage)
    with open(f"{folder_path}seed_{args.manualSeed}_selective_risk_ccl_sc.txt", 'w') as file:
        file.write("\nAbstention\tLogit\tTest\tCoverage\tError")
        print("\nAbstention\tLogit\tTest\tCoverage\tError")
        for idx, _ in enumerate(abortion_results):
            print('{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], abortion_results[idx][0] * 100.,
                                                      (1 - abortion_results[idx][1]) * 100))
            file.write('\n{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], abortion_results[idx][0] * 100.,
                                                             (1 - abortion_results[idx][1]) * 100))
        # Softmax Response Results
        sr_results = []
        bisection_method_LeoFeng(sr_scores, sr_correct, sr_results, expected_coverage)
        file.write("\n\nSoftmax\tResponse\tTest\tCoverage\tError")
        print("\Softmax\tResponse\tTest\tCoverage\tError")
        for idx, _ in enumerate(sr_results):
            # print('{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], abortion_results[idx][0]*100., (1 - abortion_results[idx][1])*100))
            print('{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], sr_results[idx][0] * 100.,
                                                      (1 - sr_results[idx][1]) * 100))
            file.write('\n{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], sr_results[idx][0] * 100.,
                                                             (1 - sr_results[idx][1]) * 100))

        if True:
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
        if True:
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

        if True:
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

        if True:
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
        if True:
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

        if True:
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

            bisection_method_LeoFeng(torch.Tensor(sim_scores), torch.Tensor(sim_correct), sim_results, expected_coverage)
            file.write("\nsim\tLogit\tTest\tCoverage\tError")
            print("\nsim\tLogit\tTest\tCoverage\tError")
            for idx, _ in enumerate(sim_results):
                print('{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], sim_results[idx][0] * 100.,
                                                          (1 - sim_results[idx][1]) * 100))
                file.write('\n{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], sim_results[idx][0] * 100.,
                                                                 (1 - sim_results[idx][1]) * 100))
        if True:
            sim_results = [(0, 0)] * len(expected_coverage)

            for i in range(0, num_classes):
                sim_results_temp = []
                sim_scores_class, sim_correct_class = torch.Tensor(sim_abortion_class[i][0]), torch.stack(
                    sim_abortion_class[i][1]).cpu()
                bisection_method_LeoFeng(sim_scores_class, sim_correct_class, sim_results_temp, expected_coverage)
                # print(sim_results_temp)
                for j, (a, b) in enumerate(sim_results_temp):
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
    exit(1)
    ####### Until here
    ################################################
    ################################################

    ################################################
    ################################################
    ####### Our code:

    abortion_scores, abortion_correct = torch.stack(abortion_results[0]), torch.stack(abortion_results[1])
    sr_scores, sr_correct = torch.stack(sr_results[0]), torch.stack(sr_results[1])


    # Abstention Logit Results
    # As we have our own validation set, it is not necessary to divide the set in two, then comment:
    """
    # use 2000 data points as the validation set (randomly shuffled)
    if args.dataset == "cacophony":
        abortion_valid, abortion_scores = abortion_scores[:569], abortion_scores[569:]
        correct_valid, abortion_correct = abortion_correct[:569], abortion_correct[569:]
    else:
        abortion_valid, abortion_scores = abortion_scores[:2000], abortion_scores[2000:]
        correct_valid, abortion_correct = abortion_correct[:2000], abortion_correct[2000:]
    results_valid = []; results = []
    bisection_method(abortion_valid, correct_valid, results_valid, expected_coverage)
    bisection_method(abortion_scores, abortion_correct, results, expected_coverage)
    """

    risk_coverage_results = {"selective_risk": [], "selective_risk_LeoFeng": [], "coverage": np.array([]) }
    abortion_results = []
    bisection_method(abortion_scores, abortion_correct, abortion_results, expected_coverage)
    # Risk-coverage curves
    selective_risk = [(1 - r[1]) * 100 for r in abortion_results]
    risk_coverage_results["selective_risk"] = selective_risk
    risk_coverage_curve_plotly(selective_risk, np.array(expected_coverage), "test (without idk class)", folder_path, loss=args.loss)

    risk_coverage_results["coverage"] = np.array(expected_coverage)

    abortion_results_LeoFeng = []
    bisection_method_LeoFeng(abortion_scores, abortion_correct, abortion_results_LeoFeng, expected_coverage)
    # Risk-coverage curves
    selective_risk_LeoFeng = [(1 - r[1]) * 100 for r in abortion_results_LeoFeng]
    risk_coverage_results["selective_risk_LeoFeng"] = selective_risk_LeoFeng
    risk_coverage_curve_plotly(selective_risk_LeoFeng, np.array(expected_coverage), "test (without idk class) - LF", folder_path, loss=args.loss)

    # Saving risk-coverage in csv
    df_risk_coverage = pd.DataFrame(risk_coverage_results)
    df_risk_coverage.to_csv(f"{folder_path}seed_{args.manualSeed}_risk_coverage.csv", index=False)


    # As we have our own validation set, it is not necessary to divide the set in two, then comment:
    """
    print("Vali\tCoverage\tError")
    for idx, _ in enumerate(results_valid):
        print('{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], results_valid[idx][0]*100., (1 - results_valid[idx][1])*100))
    print("\nTest\tCoverage\tError")
    for idx, _ in enumerate(results):
        print('{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], results[idx][0]*100., (1 - results[idx][1])*100))
    save_data(results_valid, results, reward_list, folder_path)
    """

    print("\nAbstention\tLogit\tTest\tCoverage\tError\tThreshold")
    for idx, _ in enumerate(abortion_results):
        print('{:.0f},\t{:.2f},\t\t{:.3f},\t\t{:.3f}'.format(expected_coverage[idx], abortion_results[idx][0] * 100.,
                                                  (1 - abortion_results[idx][1]) * 100,  abortion_results[idx][2]))
    #save_data(abortion_results, -1, reward_list, folder_path)

    print("\nAbstention\tLogit\tTest\tCoverage\tError\tThreshold \tLeoFeng")
    for idx, _ in enumerate(abortion_results_LeoFeng):
        print('{:.0f},\t{:.2f},\t\t{:.3f},\t\t{:.3f}'.format(expected_coverage[idx], abortion_results_LeoFeng[idx][0] * 100.,
                                                             (1 - abortion_results_LeoFeng[idx][1]) * 100,
                                                             abortion_results_LeoFeng[idx][2]))

    # Softmax Response Results
    sr_results = []
    bisection_method_LeoFeng(sr_scores, sr_correct, sr_results, expected_coverage)

    print("\Softmax\tResponse\tTest\tCoverage\tError\tThreshold")
    for idx, _ in enumerate(sr_results):
        print('{:.0f},\t{:.2f},\t\t{:.3f},\t\t{:.3f}'.format(expected_coverage[idx], sr_results[idx][0] * 100.,
                                                  (1 - sr_results[idx][1]) * 100, sr_results[idx][2]))
    #save_data(sr_results, -1, reward_list, folder_path)
    save_data_all_results("Test", abortion_results, abortion_results_LeoFeng,sr_results, expected_coverage, reward_list, folder_path)

    # GRAPHS
    images1, images2 = [], []


    for k, v in types_collate.items(): # types_collate = {"val":[val_batch, valloader], "test":[test_batch, testloader]}
        collate_model_info(model, None, folder_path, v[0], v[1], labels,
                           args, input_size=input_size, ds_type=k, use_cuda=use_cuda, num_classes=num_classes)

        """ #Remove
        images1.append(Image.open(folder_path + f"both_confusion_matrix_{k}_{args.loss}.png"))
        images2.append(Image.open(folder_path + f"count_both_confusion_matrix_{k}_{args.loss}.png"))
        """ #Remove

    """ #Remove
    fig, ax = plt.subplots(2, 1)
    fig.subplots_adjust(hspace=0, wspace=0)
    counter = 0
    for i in range(2):
        ax[i].imshow(images1[counter])
        ax[i].axis('off')
        counter += 1
    plt.savefig(folder_path + f"/all_cm_{args.loss}.png", dpi=120, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    fig, ax = plt.subplots(2, 1)
    fig.subplots_adjust(hspace=0, wspace=0)
    counter = 0
    for i in range(2):
        ax[i].imshow(images2[counter])
        ax[i].axis('off')
        counter += 1

    plt.savefig(folder_path + f"/all_count_cm_{args.loss}.png", dpi=120, bbox_inches='tight', pad_inches=0)
    plt.close()
    """ #Remove
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

    if args.loss != 'ce' or args.loss != 'focal':
        logits = F.softmax(torch.from_numpy(logits), dim=1).numpy()
        confidences = 1 - logits[:, -1]
        # print(np.histogram(confidences, bins=20, range=(0, 1), density=True)[0])
        # confidences = np.max(logits, axis=1)
        logits = logits[:, :10]
        # confidences *= np.max(logits, axis=1)
    else:
        logits = F.softmax(torch.from_numpy(logits), dim=1).numpy()
        confidences = np.max(logits, axis=1)
        # logits = logits[:, :10]

    # acc_at = eval_converage(logits, confidences, label)

    # print("\tClean")
    # for cov in acc_at:
    #     print("ERR@{}\t{:.2f}".format(cov, (1 - acc_at[cov]) * 100))
    acc_at = eval_converage(logits[:2000], confidences[:2000], label[:2000])

    print("\tVal")
    for cov in acc_at:
        print("ERR@{}\t{:.2f}".format(cov, (1 - acc_at[cov]) * 100))

    acc_at = eval_converage(logits[2000:], confidences[2000:], label[2000:])

    print("\tTest")
    for cov in acc_at:
        print("ERR@{}\t{:.2f}".format(cov, (1 - acc_at[cov]) * 100))
