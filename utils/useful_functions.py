"""
@User: sandruskyi
"""
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_curve, auc
import json
from datetime import datetime
from torchsummary import summary as m_sum
from io import StringIO
from torch import nn, no_grad, mean
import torch
import torch.nn.functional as F
from tqdm import tqdm
import cv2
from calibration.plot_reliability_diagram import plot_reliability_diagram, plot_reliability_diagram_plotly, calculate_confidence_accuracy_multiclass, calculate_AdaptiveECE_confidence_accuracy_multiclass, calculate_ClasswiseECE_confidence_accuracy_multiclass, AUC_shared_evaluation


from utils.bisection import bisection_method, bisection_method_LeoFeng
from utils.risk_coverage_curve import risk_coverage_curve_plotly
import random
from tqdm import trange


__all__ = ['collate_model_info', 'plot_history', 'plot_confusion_matrix', 'plot_confusion_matrix_number_of_instances', 'calculate_classification_metrics',
           'confidence_distro', 'make_roc',
           ]



def plot_history(history, folder):
    """
    Plotting training and validation learning curves.

    Args:
      history: model history with all the metric measures
      folder: folder to save figure to
  """
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2)
    SMALL_SIZE = 16
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 26
    fig.set_size_inches(18.5, 10.5)

    # Plot loss
    ax1.set_title('Train & Validation Losses', fontsize=BIGGER_SIZE)
    if type(history) == dict:
        ax1.plot(history['train_loss'], label='train')
        ax1.plot(history['train_NLLloss'], label='trainNLLloss')
        ax1.plot(history['val_loss'], label='test')
        ax1.plot(history['val_NLLloss'], label='valNLLloss')
    else:
        ax1.plot(history.history['train_loss'], label='train')
        ax1.plot(history.history['train_NLLloss'], label='trainNLLloss')
        ax1.plot(history.history['val_loss'], label='test')
        ax1.plot(history.history['val_NLLloss'], label='valNLLloss')
    ax1.set_ylabel('Loss', fontsize=SMALL_SIZE)
    ax1.tick_params(axis='both', which='major', labelsize=SMALL_SIZE)
    ax1.tick_params(axis='both', which='minor', labelsize=SMALL_SIZE)
    ax1.set_xlabel('Epoch', fontsize=SMALL_SIZE)
    ax1.legend(['Train', 'TrainNLL', 'Validation', 'ValidationNLL'], loc="lower left", fontsize=MEDIUM_SIZE)

    # Plot accuracy
    ax2.set_title('Train & Validation Accuracy', fontsize=BIGGER_SIZE)
    if type(history) == dict:
        ax2.plot(history['train_accuracy'], label='train')
        ax2.plot(history['val_accuracy'], label='test')
    else:
        ax2.plot(history.history['train_accuracy'], label='train')
        ax2.plot(history.history['val_accuracy'], label='test')
    ax2.set_ylabel('Accuracy', fontsize=SMALL_SIZE)
    ax2.set_xlabel('Epoch', fontsize=SMALL_SIZE)
    ax2.tick_params(axis='both', which='major', labelsize=SMALL_SIZE)
    ax2.tick_params(axis='both', which='minor', labelsize=SMALL_SIZE)
    ax2.legend(['Train', 'Validation'], loc="lower left", fontsize=MEDIUM_SIZE)
    plt.tight_layout()
    plt.savefig(folder + "train_history.png", dpi=120)
    plt.close()




def plot_confusion_matrix(actual, predicted, labels, ds_type, folder, num_classes, fraction_label=False, idkforpredator=False, loss="", labels_aux="", labels_current="", **kwargs):
    """Uses seaborn heatmap and scikit-learn confusion_matrix to make and plot a confusion matrix"""
    plt.clf()
    #print("PLOT CONFUSION MATRIX LABELS, actual:", np.unique(actual), "predicted:", np.unique(predicted))
    cm = confusion_matrix(actual, predicted, labels=labels_current, normalize='true')
    plt.figure(figsize=(16, 9))
    sns.set(font_scale=1.8)
    if fraction_label:
        ax = sns.heatmap(cm, annot=cm, fmt='.1%', annot_kws={"fontsize": 20})
    else:
        cm2 = confusion_matrix(actual, predicted, labels=labels_current)
        ax = sns.heatmap(cm, annot=cm2, fmt='g')
    ax.set_title('Confusion Matrix on ' + ds_type.capitalize() + ' Set ( '+ str(actual.shape[0]) + ')' , fontweight="bold", fontsize=36, pad=20)
    ax.set_xlabel('Predicted', fontsize=30)
    ax.set_ylabel('Actual', fontsize=30)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    #print("labels_aux", labels_aux, "labels", labels, len(ax.xaxis.get_major_locator().locs), len(ax.yaxis.get_major_locator().locs))

    if labels_aux!="" and (len(ax.xaxis.get_major_locator().locs) == (num_classes+1)):
    #if labels_aux!="":
        ax.xaxis.set_ticklabels(labels_aux, fontsize=30)
        ax.yaxis.set_ticklabels(labels_aux, fontsize=30)
    else:
        ax.xaxis.set_ticklabels(labels, fontsize=30)
        ax.yaxis.set_ticklabels(labels, fontsize=30)


    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    if "path" in kwargs.keys():
        plt.savefig(kwargs["path"], dpi=120)
    elif idkforpredator:
        plt.savefig(folder + f"idk_pred_confusion_matrix_{ds_type}_{loss}.png", dpi=120)
    else:
        plt.savefig(folder + f"confusion_matrix_{ds_type}_{loss}.png", dpi=120)
    plt.close()

def plot_confusion_matrix_number_of_instances(actual, predicted, labels, ds_type, folder, num_classes, fraction_label=False, idkforpredator=False, loss="", labels_aux="", labels_current="", **kwargs):
    """Uses seaborn heatmap and scikit-learn confusion_matrix to make and plot a confusion matrix"""
    plt.clf()
    cm = confusion_matrix(actual, predicted, labels=labels_current)
    plt.figure(figsize=(16, 9))
    sns.set(font_scale=1.8)
    if fraction_label:
        ax = sns.heatmap(cm, annot=cm, fmt='d', annot_kws={"fontsize": 20})
    else:
        cm2 = confusion_matrix(actual, predicted, labels=labels_current)
        ax = sns.heatmap(cm, annot=cm2, fmt='d')
    ax.set_title('Confusion Matrix on ' + ds_type.capitalize() + ' Set (' + str(len(actual)) + ')', fontweight="bold", fontsize=36, pad=20)
    ax.set_xlabel('Predicted Species', fontsize=30)
    ax.set_ylabel('Actual Species', fontsize=30)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    if labels_aux != "" and (len(ax.xaxis.get_major_locator().locs) == (num_classes+1)):
    #if labels_aux != "":
        ax.xaxis.set_ticklabels(labels_aux, fontsize=30)
        ax.yaxis.set_ticklabels(labels_aux, fontsize=30)
    else:
        ax.xaxis.set_ticklabels(labels, fontsize=30)
        ax.yaxis.set_ticklabels(labels, fontsize=30)


    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    if "path" in kwargs.keys():
        plt.savefig(kwargs["path"], dpi=120)
    elif idkforpredator:
        plt.savefig(folder + f"count_idk_pred_confusion_matrix_{ds_type}_{loss}.png", dpi=120)
    else:
        plt.savefig(folder + f"count_confusion_matrix_{ds_type}_{loss}.png", dpi=120)
    plt.close()


def calculate_classification_metrics(y_actual, y_pred, labels):
    """
    Calculate the precision and recall of a classification model using the ground truth and
    predicted values.

    Args:
      y_actual: Ground truth labels.
      y_pred: Predicted labels.
      labels: List of classification labels.

    Return:
      Precision and recall measures.
  """
    cm = confusion_matrix(y_actual, y_pred).astype(float)
    FP = cm.sum(axis=0) - np.diag(cm).astype(float)
    FN = cm.sum(axis=1) - np.diag(cm).astype(float)
    TP = np.diag(cm).astype(float)
    TN = cm.sum() - (FP + FN + TP).astype(float)

    # Recall
    TPR = TP / (TP + FN)
    TPR = np.nan_to_num(TPR)
    # True negative rate
    TNR = TN / (TN + FP)
    TNR = np.nan_to_num(TNR)
    # Precision
    PPV = TP / (TP + FP)
    PPV = np.nan_to_num(PPV)
    # Negative predictive value
    NPV = TN / (TN + FN)
    NPV = np.nan_to_num(NPV)
    # False positive rate
    FPR = FP / (FP + TN)
    FPR = np.nan_to_num(FPR)
    # False negative rate
    FNR = FN / (TP + FN)
    FNR = np.nan_to_num(FNR)
    # False discovery rate
    FDR = FP / (TP + FP)
    FDR = np.nan_to_num(FDR)
    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    ACC = np.nan_to_num(ACC)

    print("range(len(PPV))", range(len(PPV)))
    print("labels[i]", labels, len(labels))

    precision = {labels[i]: PPV[i] for i in range(len(PPV))}
    recall = {labels[i]: TPR[i] for i in range(len(TPR))}
    fps = {labels[i]: FPR[i] for i in range(len(FPR))}
    fns = {labels[i]: FNR[i] for i in range(len(FNR))}
    accuracy = {labels[i]: ACC[i] for i in range(len(ACC))}
    precision["average"] = sum(precision.values())/len(precision.values())
    recall["average"] = sum(recall.values())/len(recall.values())
    fps["average"] = sum(fps.values())/len(fps.values())
    fns["average"] = sum(fns.values())/len(fns.values())
    accuracy["average"] = sum(TP)/len(y_actual)
    return precision, recall, fps, fns, accuracy

def confidence_distro(folder, raw_data, actual_data, model_name):
    """Plots the distribution of raw confidence scores for binary predictions.
    The plot is split by correct and incorrect predictions."""
    raw_data = raw_data.flatten()
    actual_data = actual_data.flatten()
    correct = []
    incorrect = []
    for i, value in enumerate(actual_data):
        if abs(value - raw_data[i]) <= 0.5:
            correct.append(raw_data[i])
        else:
            incorrect.append(raw_data[i])

    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    f_pipe_part = " ".join(model_name.split("_")).title()
    plt.clf()
    plt.close()
    plt.figure(figsize=(10, 8))
    plt.title(f"Binary Confidence Scores for {f_pipe_part}", fontweight="bold", fontsize=20)
    plt.xlabel("Confidence", fontsize=18)
    plt.ylabel("Frequency", fontsize=18)
    plt.tick_params(axis="both", labelsize=18)
    plt.hist(correct, label="Correct Predictions", bins=bins, edgecolor="black", linewidth=1, color=(0, 0, 1, 0.1))
    plt.hist(incorrect, label="Incorrect Predictions", bins=bins, edgecolor="red", linewidth=1.3, ls="dashed",
             color=(1, 0, 0, 0.1))
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{folder}confidence_distro.png", dpi=120)


def make_roc(file, raw, actual, model_name):
    """Creates a Receiver Operating Characteristic graph for a binary model"""
    pred = np.array(raw).flatten()
    actual = actual.flatten()
    fpr, tpr, threshold = roc_curve(actual, pred)
    roc_auc = auc(fpr, tpr)
    model_name = " ".join(model_name.split("_")).title()
    plt.clf()
    plt.close()
    plt.figure(figsize=(9, 9))
    plt.title(f"{model_name} Receiver Operating Characteristic", fontweight="bold", fontsize=20)
    plt.plot(fpr, tpr, 'b', label=f"AUC = {roc_auc:.2f}")
    plt.legend(loc="lower right", fontsize=18)
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate", fontsize=18)
    plt.xlabel("False Positive Rate", fontsize=18)
    plt.tick_params(axis="both", labelsize=18)
    plt.tight_layout()
    plt.savefig(f"{file}roc.png", dpi=120)



def collate_model_info(model, history, file, batch_size, loader, labels, args, threshold=0.5,
                       input_size=(45, 3, 24, 24), ds_type = "test", use_cuda=False, num_classes=2, calibrator=None):
    """
    - Plot history if exist
    - For the dataset calculates confidences_without_idk, idk_confidences
    """
    epochs = args.epochs
    loss = args.loss
    idkforpredator = args.idkforpredator
    seed = args.manualSeed

    """Used after model training to evaluate the models performance on a test set"""
    if not os.path.exists(file):
        os.mkdir(file)

    if history is not None:
        plot_history(history, file)

    num_classes_current = num_classes
    labels_current_case_without_idk = np.arange(0, num_classes_current)
    if loss == 'sat' or loss=='FocalKnowThatIdk' or loss=='Socrates':
        num_classes_current+=1
    labels_current = np.arange(0, num_classes_current)

    if args.transfer_learning or (not args.clusterLessPower and (args.loss == 'csc' or args.loss == 'csc_entropy' or args.loss == 'csc_sat_entropy')) or args.temperatureScaling or args.dirichlet or args.vectorScaling or args.matrixScaling:
        model_part = model
    else:
        model_part = model.module

    # Prediction
    start = datetime.now()
    model.eval()
    with torch.no_grad():
        if 'evaluation' in file or args.arch=='resnet50' or args.arch=='resnet34' or args.arch == "resnet110":
            predicted_raw = []
            y_test = []
            for batch_idx, batch_data in tqdm(enumerate(loader), total=len(loader)):
                inputs, targets, indices = batch_data
                if use_cuda:
                    if args.clusterLessPower:
                        primary_device = list(map(int, args.gpu_id.split(',')))[0]
                        inputs, targets, indices = inputs.to(f'cuda:{primary_device}'), targets.to(f'cuda:{primary_device}'), indices.to(f'cuda:{primary_device}')
                    else:
                        inputs, targets, indices = inputs.cuda(), targets.cuda(), indices.cuda()
                targets = targets.to(torch.int64)
                y_test.append(targets.cpu().numpy())
                pr_raw1 = inputs
                if args.arch == "vit"  and not args.temperatureScaling and not args.dirichlet:
                    if args.transfer_learning:
                        outputs_logits_tl = model(pr_raw1)
                        pr_raw = outputs_logits_tl.logits
                    else:
                        pr_raw, _ = model(pr_raw1)
                else:
                    pr_raw = model(pr_raw1)

                if args.dirichlet:
                    if args.loss == 'Socrates':
                        pr_raw = pr_raw[:, :-1]  # Removing the idk class
                    pr_raw = calibrator.predict_proba(pr_raw)
                    pr_raw = torch.tensor(pr_raw)

                elif args.vectorScaling or args.matrixScaling:
                    pr_raw = calibrator.calibrate(pr_raw)
                    pr_raw = torch.from_numpy(pr_raw).to(inputs.device)

                pr_raw = F.softmax(pr_raw, dim=1)
                pr_raw = pr_raw.cpu()
                predicted_raw.append(pr_raw)
            predicted_raw = torch.from_numpy(np.concatenate(predicted_raw))
            y_test =  np.concatenate(y_test)
        elif issubclass(type(model), nn.Module):
            predicted_raw = []
            y_test = []
            for batch_idx, batch_data in tqdm(enumerate(loader), total=len(loader)):
                inputs, targets, indices = batch_data
                targets = targets.to(torch.int64)
                y_test.append(targets)

                outputs_pr = model_part.predict(inputs)
                if args.dirichlet:
                    if args.loss == 'Socrates':
                        outputs_pr = outputs_pr[:, :-1]  # Removing the idk class
                    outputs_pr = calibrator.predict_proba(outputs_pr)
                elif args.vectorScaling or args.matrixScaling:
                    outputs_pr = calibrator.calibrate(outputs_pr)


                predicted_raw.append(outputs_pr) # Has a softmax inside
            predicted_raw = torch.from_numpy(np.concatenate(predicted_raw))
            y_test = np.concatenate(y_test)


        if loss == 'sat' or loss == 'FocalKnowThatIdk' or loss=='Socrates':
            confidences_without_idk, idk_confidences = predicted_raw[:, :-1], predicted_raw[:, -1]
        else:
            confidences_without_idk = predicted_raw.clone()
        predicted_raw = predicted_raw.numpy()


    prediction_time = datetime.now() - start
    prediction_time = prediction_time.total_seconds()


    if predicted_raw.ndim > 1: # Multiclass case
        if predicted_raw.shape[-1] != 1: # Our case
            predicted = np.argmax(predicted_raw, axis=1) # get indices of the maximum values of all the predictions with the idk class
            higher_predicted_confidences_with_idk = predicted_raw[np.arange(len(predicted)), predicted] # List with the confidences of the argmax softmax

            predicted2 = np.argmax(confidences_without_idk, axis=1) # get indices of the maximum values without idk class. In this case the indices are the same than the class
            higher_predicted_confidences_without_idk = confidences_without_idk[np.arange(len(predicted2)), predicted2] # List with the confidences of the argmax softmax
        else:
            predicted = predicted_raw.flatten()
            np_func = np.vectorize(lambda x: 1 if x >= threshold else 0)
            predicted = np_func(predicted)
    else:
        predicted = predicted_raw.copy()

    if predicted_raw.ndim > 1 and predicted_raw.shape[-1] != 1: # Our case

        # 1 - Reliability Diagram with idk class
        accuracy_list_1_idk, confidence_list_1_idk, counter_list_1_idk, gap_list_1_idk, MCE_1_idk, ECE_1_idk, RMSCE_1_idk, min_list_1_idk, counter_zero_list_1_idk = calculate_confidence_accuracy_multiclass(y_test, predicted, higher_predicted_confidences_with_idk, 10)
        AdaECE_accuracy_list_1_idk, AdaECE_confidence_list_1_idk, AdaECE_counter_list_1_idk, AdaECE_gap_list_1_idk, AdaMCE_1_idk, AdaECE_1_idk, AdaRMSCE_1_idk, AdaECE_min_list_1_idk, AdaECE_counter_zero_list_1_idk, AdaECEaux_with_idk, AdaECE_n_perbin, AdaECE_bin_edges = calculate_AdaptiveECE_confidence_accuracy_multiclass(y_test, predicted, higher_predicted_confidences_with_idk, 10)
        classwiseECE_accuracy_list_1_idk, classwiseECE_confidence_list_1_idk, classwiseECE_counter_list_1_idk, classwiseECE_gap_list_1_idk, classwiseMCE_1_idk, classwiseECE_1_idk, classwiseRMSCE_1_idk, classwiseECE_min_list_1_idk, classwiseECE_counter_zero_list_1_idk = calculate_ClasswiseECE_confidence_accuracy_multiclass(y_test, predicted, higher_predicted_confidences_with_idk, 10, num_classes, predicted_raw, args)
        if not args.vectorScaling and not args.matrixScaling and not args.dirichlet:
            auc = AUC_shared_evaluation(y_test, predicted_raw, num_classes, args)
            df_ECE_MCE_RMSCE = pd.DataFrame({ "ECE":[f"{ECE_1_idk:.6}"], "MCE":[f"{MCE_1_idk:.6f}"], "RMSCE":[f"{RMSCE_1_idk:.6f}"],
                                              "AdaECE":[f"{AdaECE_1_idk:.6}"], "AdaMCE":[f"{AdaMCE_1_idk:.6f}"], "AdaRMSCE":[f"{AdaRMSCE_1_idk:.6f}"],
                                              "classwiseECE":[f"{classwiseECE_1_idk:.6}"], "classwiseMCE":[f"{float(classwiseMCE_1_idk):.6f}"], "classwiseRMSCE":[f"{float(classwiseRMSCE_1_idk):.6f}"],
                                              "auc":[f"{auc:.6f}"]})
        else:
            df_ECE_MCE_RMSCE = pd.DataFrame(
                {"ECE": [f"{ECE_1_idk:.6}"], "MCE": [f"{MCE_1_idk:.6f}"], "RMSCE": [f"{RMSCE_1_idk:.6f}"],
                 "AdaECE": [f"{AdaECE_1_idk:.6}"], "AdaMCE": [f"{AdaMCE_1_idk:.6f}"],
                 "AdaRMSCE": [f"{AdaRMSCE_1_idk:.6f}"],
                 "classwiseECE": [f"{classwiseECE_1_idk:.6}"], "classwiseMCE": [f"{float(classwiseMCE_1_idk):.6f}"],
                 "classwiseRMSCE": [f"{float(classwiseRMSCE_1_idk):.6f}"]})
        y_true = np.array(torch.stack(y_test).cpu()).tolist() if (
                    isinstance(y_test, torch.Tensor) or isinstance(y_test, list)) else y_test.tolist()
        df_ECE_MCE_RMSCE["Accuracy"] = np.mean(predicted == y_true)
        df_ECE_MCE_RMSCE.to_csv(f"{file}seed_{args.manualSeed}_AUC_ECE_MCE_RMSCE_NormalandADAandClassWise_{ds_type}.csv", index=False)
        plot_reliability_diagram_plotly(accuracy_list_1_idk, confidence_list_1_idk,"Accuracy", counter_list_1_idk, counter_zero_list_1_idk,
                                        ds_type, gap_list_1_idk, ECE_1_idk, MCE_1_idk, min_list_1_idk, model_name=ds_type + " (predictions with idk class)",calib_name="", onlyReliabilityDiagram=True, path=file, loss = loss)
        plot_reliability_diagram(accuracy_list_1_idk, confidence_list_1_idk, ds_type + " (predictions with idk class)", "Accuracy", ds_type, path=file, loss = loss)

        # 2 - Reliability Diagram without idk class
        if not ( args.dirichlet and args.loss == 'Socrates'):
            accuracy_list_1_noidk, confidence_list_1_noidk, counter_list_1_noidk, gap_list_1_noidk, MCE_1_noidk, ECE_1_noidk, RMSCE_1_noidk, min_list_1_noidk, counter_zero_list_1_noidk = calculate_confidence_accuracy_multiclass(y_test, predicted2, higher_predicted_confidences_without_idk, 10)

            AdaECE_accuracy_list_1_noidk, AdaECE_confidence_list_1_noidk, AdaECE_counter_list_1_noidk, AdaECE_gap_list_1_noidk, AdaMCE_1_noidk, AdaECE_1_noidk, AdaRMSCE_1_noidk, AdaECE_min_list_1_noidk, AdaECE_counter_zero_list_1_noidk, AdaECEaux_noidk, AdaECE_n_perbin_noidk, AdaECE_bin_edges_noidk = calculate_AdaptiveECE_confidence_accuracy_multiclass(y_test, predicted2, higher_predicted_confidences_without_idk, 10)
            classwiseECE_accuracy_list_1_noidk, classwiseECE_confidence_list_1_noidk, classwiseECE_counter_list_1_noidk, classwiseECE_gap_list_1_noidk, classwiseMCE_1_noidk, classwiseECE_1_noidk, classwiseRMSCE_1_noidk, classwiseECE_min_list_1_noidk, classwiseECE_counter_zero_list_1_noidk = calculate_ClasswiseECE_confidence_accuracy_multiclass(y_test, predicted2, higher_predicted_confidences_without_idk, 10, num_classes, confidences_without_idk)

            plot_reliability_diagram_plotly(accuracy_list_1_noidk, confidence_list_1_noidk,
                                            "Accuracy", counter_list_1_noidk, counter_zero_list_1_noidk,
                                            ds_type, gap_list_1_noidk, ECE_1_noidk, MCE_1_noidk, min_list_1_noidk, model_name=ds_type + " (predictions without idk class)",calib_name="", onlyReliabilityDiagram=True, path=file, loss = loss)
            plot_reliability_diagram(accuracy_list_1_noidk, confidence_list_1_noidk, ds_type + " (predictions without idk class)", "Accuracy", ds_type, path=file, loss = loss)
            if not args.vectorScaling and not args.matrixScaling:
                df_ECE_MCE_RMSCE_no_idk = pd.DataFrame({ "ECE":[f"{ECE_1_noidk:.6}"], "MCE":[f"{MCE_1_noidk:.6f}"], "RMSCE":[f"{RMSCE_1_noidk:.6f}"],
                                                  "AdaECE":[f"{AdaECE_1_noidk:.6}"], "AdaMCE":[f"{AdaMCE_1_noidk:.6f}"], "AdaRMSCE":[f"{AdaRMSCE_1_noidk:.6f}"],
                                                  "classwiseECE":[f"{classwiseECE_1_noidk:.6}"], "classwiseMCE":[f"{float(classwiseMCE_1_noidk):.6f}"], "classwiseRMSCE":[f"{float(classwiseRMSCE_1_noidk):.6f}"],
                                                  "auc":[f"{auc:.6f}"]})
            else:
                df_ECE_MCE_RMSCE_no_idk = pd.DataFrame(
                    {"ECE": [f"{ECE_1_noidk:.6}"], "MCE": [f"{MCE_1_noidk:.6f}"], "RMSCE": [f"{RMSCE_1_noidk:.6f}"],
                     "AdaECE": [f"{AdaECE_1_noidk:.6}"], "AdaMCE": [f"{AdaMCE_1_noidk:.6f}"],
                     "AdaRMSCE": [f"{AdaRMSCE_1_noidk:.6f}"],
                     "classwiseECE": [f"{classwiseECE_1_noidk:.6}"],
                     "classwiseMCE": [f"{float(classwiseMCE_1_noidk):.6f}"],
                     "classwiseRMSCE": [f"{float(classwiseRMSCE_1_noidk):.6f}"]})
            y_true = np.array(torch.stack(y_test).cpu()).tolist() if (
                        isinstance(y_test, torch.Tensor) or isinstance(y_test, list)) else y_test.tolist()
            df_ECE_MCE_RMSCE_no_idk["Accuracy"] = np.mean(predicted2.numpy() == y_true)
            df_ECE_MCE_RMSCE_no_idk.to_csv(f"{file}seed_{args.manualSeed}_AUC_ECE_MCE_RMSCE_NormalandADAandClassWise_{ds_type}_noidk.csv", index=False)
        if args.vectorScaling or args.matrixScaling or args.dirichlet:
            exit(1)
        if loss == 'sat' or loss == 'FocalKnowThatIdk' or loss=='Socrates':
            abortion_results = [[],[]]
            """"
            if loss == 'ce' or loss == 'focal':
                abortion_results[0] = list(1 - predicted_raw.max(1)[0])
            else:
            """
            abortion_results[0] = list(idk_confidences)
            abortion_results[1] = list(predicted2.eq(torch.tensor(y_test)))

            def shuffle_list(lst, seed=10):
                random.seed(seed)
                random.shuffle(lst)
            shuffle_list(abortion_results[0]); shuffle_list(abortion_results[1])
            abortion, correct = torch.stack(abortion_results[0]), torch.stack(abortion_results[1])
            #abortion = torch.tensor(predicted_raw[:,-1].copy())
            #correct = torch.tensor(predicted2 == actual)
            expected_coverage = np.round(np.arange(100,-5, -5),2) # It needs to start with the biggest coverage to optimize faster test_thres in the bisection method
            results = []
            bisection_method(abortion, correct, results, expected_coverage)
            selective_risk = [(1-r[1])*100 for r in results]
            # Risk-coverage curves
            risk_coverage_curve_plotly(selective_risk, expected_coverage, ds_type + " (without idk class)", file, loss=loss)

            results_LeoFeng = []
            bisection_method_LeoFeng(abortion, correct, results_LeoFeng, expected_coverage)
            selective_risk_LeoFeng = [(1 - r[1]) * 100 for r in results_LeoFeng]
            # Risk-coverage curves
            risk_coverage_curve_plotly(selective_risk_LeoFeng, expected_coverage, ds_type + " (without idk class) - LF", file,
                                       loss=loss)

        labels_idk = labels.copy()
        labels_idk.append("idk")
        """ # REMOVE
        #print("FIRST PLOTTING CONFUSION MATRIX WITH IDK: labels_idk", labels_idk, "actual:", np.unique(actual), "predicted", np.unique(predicted))
        plot_confusion_matrix(y_test, predicted2, labels, ds_type + " (without idk class)", file, num_classes, fraction_label=True, idkforpredator=False, loss = loss, labels_current=labels_current_case_without_idk)
        plot_confusion_matrix(y_test, predicted, labels, ds_type + " (with idk class)", file, num_classes, fraction_label=True, idkforpredator=False, loss = loss, labels_aux=labels_idk, labels_current=labels_current)
        plot_confusion_matrix_number_of_instances(y_test, predicted2, labels, ds_type + " (without idk class)", file, num_classes, fraction_label=True, idkforpredator=False, loss = loss, labels_current=labels_current_case_without_idk)
        plot_confusion_matrix_number_of_instances(y_test, predicted, labels, ds_type + " (with idk class)", file, num_classes, fraction_label=True, idkforpredator=False, loss = loss, labels_aux=labels_idk, labels_current=labels_current)
        """ # REMOVE


    if idkforpredator and args.dataset == "cacophony":
        #print(ds_type, "total shape", predicted.shape, "class 0", len(predicted[predicted==0]), "class 1:", len(predicted[predicted==1]),"class 2: ", len(predicted[predicted==2]))
        predicted[predicted==2] = 1
        predicted_confidences2 = predicted_raw[np.arange(len(predicted)), predicted]
        #print("len(predicted[predicted==1]) after change idk per predators", len(predicted[predicted==1]))


    #if predicted_raw.ndim > 1 and predicted_raw.shape[-1] != 1:

        # Realiability Diagram
        accuracy_list_2, confidence_list_2, counter_list_2, gap_list_2, MCE_2, ECE_2, RMSCE_2, min_list_2, counter_zero_list_2 = calculate_confidence_accuracy_multiclass(y_test, predicted, predicted_confidences2, 10)
        plot_reliability_diagram_plotly(accuracy_list_2, confidence_list_2,
                                        "Accuracy", counter_list_2, counter_zero_list_2,
                                        ds_type, gap_list_2, ECE_2, MCE_2, min_list_2, ds_type+ " (idk class as predator class)","", True, path=file, loss = loss)
        plot_reliability_diagram(accuracy_list_1, confidence_list_1, ds_type + " (without idk class)", "Accuracy", ds_type, path=file, loss = loss)

        plot_confusion_matrix(y_test, predicted, labels, ds_type + " (idk class as predator class)", file, num_classes, fraction_label=True, idkforpredator=idkforpredator, loss = loss, labels_current=labels_current_case_without_idk)
        plot_confusion_matrix_number_of_instances(y_test, predicted, labels, ds_type + " (idk class as predator class)", file, num_classes, fraction_label=True, idkforpredator=idkforpredator, loss = loss, labels_current=labels_current_case_without_idk)

        precision, recall, fpr, fnr, accuracy = calculate_classification_metrics(y_test, predicted2, labels)
    #elif args.dataset == "cifar10":
    #    precision, recall, fpr, fnr, accuracy = calculate_classification_metrics(y_test, predicted2, labels)
    else:
        """ # Remove
        precision, recall, fpr, fnr, accuracy = calculate_classification_metrics(y_test, predicted, labels)
        

    plot1 = Image.open(file + f"confusion_matrix_{ds_type} (without idk class)_{loss}.png")
    plot2 = Image.open(file + f"confusion_matrix_{ds_type} (with idk class)_{loss}.png")
    
    if idkforpredator and args.dataset == "cacophony":
        fig, ax = plt.subplots(1, 3)
        plot3 = Image.open(file + f"idk_pred_confusion_matrix_{ds_type} (idk class as predator class)_{loss}.png")
    else:
        fig, ax = plt.subplots(1, 2)

    fig.subplots_adjust(hspace=0, wspace=0)
    ax[0].imshow(plot1)
    ax[0].axis('off')
    ax[1].imshow(plot2)
    ax[1].axis('off')
    if idkforpredator and args.dataset == "cacophony":
        ax[2].imshow(plot3)
        ax[2].axis('off')
    plt.savefig(file + f"both_confusion_matrix_{ds_type}_{loss}.png", dpi=120, bbox_inches='tight', pad_inches=0)
    plt.close()
    


    plot1 = Image.open(file + f"count_confusion_matrix_{ds_type} (without idk class)_{loss}.png")
    plot2 = Image.open(file + f"count_confusion_matrix_{ds_type} (with idk class)_{loss}.png")
    
    if idkforpredator and args.dataset == "cacophony":
        plot3 = Image.open(file + f"count_idk_pred_confusion_matrix_{ds_type} (idk class as predator class)_{loss}.png")
        fig, ax = plt.subplots(1, 3)
    else:
        fig, ax = plt.subplots(1, 2)
    fig.subplots_adjust(hspace=0, wspace=0)
    ax[0].imshow(plot1)
    ax[0].axis('off')
    ax[1].imshow(plot2)
    ax[1].axis('off')
    if idkforpredator and args.dataset == "cacophony":
        ax[2].imshow(plot3)
        ax[2].axis('off')
    plt.savefig(file + f"count_both_confusion_matrix_{ds_type}_{loss}.png", dpi=120, bbox_inches='tight', pad_inches=0)
    plt.close()
    """ # Remove


    if history is not None and type(history) is not dict:
        history = {key: np.array(value, dtype=float).tolist() for key, value in history.history.items()} \
            if history is not None else {}
    """"
    summary = []
    try:
        model.summary(print_fn=lambda x: summary.append(x))
    except AttributeError:
        buffer = StringIO()
        sys.stdout = buffer
        if use_cuda:
            m_sum(model_part, input_size=input_size)
        else:
            m_sum(model, input_size=input_size)
        summary.extend(buffer.getvalue().splitlines()[27:30])
        sys.stdout = sys.__stdout__
    """
    if 'evaluation' not in file:
        model_info = {"model_name": model_part.name,
                      "seed": seed,
                      "inf_time": {"all_inf (s)": round(prediction_time, 3),
                                   "per_inf (ms)": round(prediction_time / len(y_test) * 1000, 3)},
                      "batch_size": batch_size,
                      "epochs": epochs,
                      "precision": precision,
                      "recall": recall,
                      #"complexity": summary[-4:-1],
                      "fpr": fpr,
                      "fnr": fnr,
                      "accuracy": accuracy,
                      "history": history,
                      "actual_test": y_test.tolist(),
                      "inf_test": predicted.tolist(),
                      "inf_raw": predicted_raw.flatten().tolist()}
        with open(file + "raw_model_data.json", "w") as raw_file:
            json.dump(model_info, raw_file, indent=4)
        if len(labels) == 2:
            confidence_distro(file, predicted_raw, y_test, model_part.name)
            #make_roc(file, predicted_raw, actual, model_part.name)

        return model_info
    return


if __name__ == "__main__":
    pass
