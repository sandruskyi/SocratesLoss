"""
@User: sandruskyi
Code to calculate the confidence and the accuracy (or the relative frequency) necesary for to print the reliability diagram.
Also, code to print the reliability diagram alone or with all the calibrators
"""
import numpy as np
import pandas as pd
import argparse
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from PIL import Image
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize


def calculate_confidence_accuracy_sklearn(y_true, y_pred, nbins):
    """
    Calculate confidence - Accuracy. Binary Methods. Sklearn

    Returns:
        prob_true_binary, prob_pred_binary: Calibration values for each bin

    """
    # calculate the values of calibration curve for bin 0 vs all
    prob_true_binary, prob_pred_binary = calibration_curve(y_true, y_pred, n_bins=nbins)
    return prob_true_binary, prob_pred_binary


def calculate_confidence_accuracy_multiclass(y_true, y_pred, confidences, nbins):
    """
    Confidence and accuracy calculation for multiclass reliability Diagram - Optimized
        # FOR THE MULTICLASS CASE IMPORTANT: IN THE BINARY CASE IT IS NOT THE ACCURACY!!!!!!!! IT IS THE RELATIVE FREQUENCY!!!!!!!! https://towardsdatascience.com/introduction-to-reliability-diagrams-for-probability-calibration-ed785b3f5d44

    """
    # First, part in M bins of size 1/M

    y_true = np.array(torch.stack(y_true).cpu()).tolist() if (isinstance(y_true, torch.Tensor) or isinstance(y_true, list)) else y_true.tolist()

    confidences = np.round(confidences, 7).tolist()

    bm = 1 / nbins  # size of each bin
    bin_edges = np.floor(np.linspace(0, 1, nbins + 1)*10)/10
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Digitize confidences into bins:
    trunc_confidences = np.floor(np.array(confidences) * 10) / 10  # To avoid that 0.6999 belongs to 0.7 (It belongs to [0.6,0.7) bin, no to [0.7,0.8))
    # bin_indices = np.digitize(confidences, bin_edges, right=False) - 1 # -1 because the list starts at 0
    bin_edges = np.array(bin_edges) if not isinstance(bin_edges, np.ndarray) else bin_edges
    trunc_confidences = np.array(trunc_confidences) if not isinstance(trunc_confidences,np.ndarray) else trunc_confidences
    bin_indices = np.searchsorted(bin_edges, trunc_confidences, side='left')
    bin_indices = np.clip(bin_indices, 0, nbins - 1)  # Ensure indices are within the range

    # Initialize arrays
    confidence_sums = np.zeros(nbins)
    accuracy_sums = np.zeros(nbins)
    counts = np.zeros(nbins)
    # Populate bins
    for i in range(len(confidences)):
        bin_idx = bin_indices[i]
        confidence_sums[bin_idx] += confidences[i]
        accuracy_sums[bin_idx] += (y_pred[i] == y_true[i])
        counts[bin_idx] += 1

    # Handle empty bins
    empty_bins = counts == 0
    confidence_sums[empty_bins] = bin_centers[empty_bins]
    accuracy_sums[empty_bins] = bin_centers[empty_bins]
    counts[empty_bins] = 1

    # confidence_list # accuracy_list # counter_list  # ECE_list, RMSCE_list, gap_list, min_list, counter_zero_list ????
    # Compute metrics:
    confidence_list = confidence_sums / counts
    accuracy_list = accuracy_sums / counts
    gaps = np.abs(accuracy_list - confidence_list)
    gaps_sq = (accuracy_list - confidence_list) ** 2
    MCE = np.max(gaps)
    ECE = np.sum(gaps * counts) / len(y_true)
    RMSCE = np.sqrt(np.sum(gaps_sq * counts) / len(y_true))
    min_np = np.minimum(accuracy_list, confidence_list)
    counts = counts.astype(int)  # For consistency
    counter_zero_list = empty_bins.tolist()

    return accuracy_list.tolist(), confidence_list.tolist(), counts.tolist(), gaps.tolist(), MCE.tolist(), ECE.tolist(), RMSCE.tolist(), min_np.tolist(), counter_zero_list

def calculate_AdaptiveECE_confidence_accuracy_multiclass(y_true, y_pred, confidences, nbins):
    """
    Confidence and accuracy calculation for multiclass reliability Diagram - Optimized
        # FOR THE MULTICLASS CASE IMPORTANT: IN THE BINARY CASE IT IS NOT THE ACCURACY!!!!!!!! IT IS THE RELATIVE FREQUENCY!!!!!!!! https://towardsdatascience.com/introduction-to-reliability-diagrams-for-probability-calibration-ed785b3f5d44

    Version from https://github.com/torrvision/focal_calibration/blob/475422a6754e20d25216158a874749c33e826e21/Metrics/metrics.py#L191
    """
    # First, part in M bins of size 1/M

    y_true = np.array(torch.stack(y_true).cpu()).tolist() if (isinstance(y_true, torch.Tensor) or isinstance(y_true, list)) else y_true.tolist()
    confidences = np.round(confidences, 7).tolist()

    # Histogram edges equal Number of instances
    len_confidences = len(confidences)
    edges = np.interp(np.linspace(0, len_confidences, nbins+1), np.arange(len_confidences), np.sort(confidences))
    n, bin_edges = np.histogram(confidences, edges) # n is the number of instances per bin


    #bm = 1 / nbins  # size of each bin
    #bin_edges = np.floor(np.linspace(0, 1, nbins + 1)*10)/10
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Digitize confidences into bins:
    #trunc_confidences = np.floor(np.array(confidences) * 10) / 10  # To avoid that 0.6999 belongs to 0.7 (It belongs to [0.6,0.7) bin, no to [0.7,0.8))
    # bin_indices = np.digitize(confidences, bin_edges, right=False) - 1 # -1 because the list starts at 0
    bin_edges = np.array(bin_edges) if not isinstance(bin_edges, np.ndarray) else bin_edges
    confidences_array = np.array(confidences) if not isinstance(confidences, np.ndarray) else confidences
    #trunc_confidences = np.array(trunc_confidences) if not isinstance(trunc_confidences,np.ndarray) else trunc_confidences
    #bin_indices = np.searchsorted(bin_edges, trunc_confidences, side='left')
    bin_indices = np.searchsorted(bin_edges, confidences_array, side='left')
    bin_indices = np.clip(bin_indices, 0, nbins - 1)  # Ensure indices are within the range

    # Initialize arrays
    confidence_sums = np.zeros(nbins)
    accuracy_sums = np.zeros(nbins)
    counts = np.zeros(nbins)
    # Populate bins
    for i in range(len(confidences)):
        bin_idx = bin_indices[i]
        confidence_sums[bin_idx] += confidences[i]
        accuracy_sums[bin_idx] += (y_pred[i] == y_true[i])
        counts[bin_idx] += 1

    # Handle empty bins
    empty_bins = counts == 0
    confidence_sums[empty_bins] = bin_centers[empty_bins]
    accuracy_sums[empty_bins] = bin_centers[empty_bins]
    counts[empty_bins] = 1

    # confidence_list # accuracy_list # counter_list  # ECE_list, RMSCE_list, gap_list, min_list, counter_zero_list
    # Compute metrics:
    confidence_list = confidence_sums / counts
    accuracy_list = accuracy_sums / counts
    gaps = np.abs(accuracy_list - confidence_list)
    gaps_sq = (accuracy_list - confidence_list) ** 2
    AdaMCE = np.max(gaps)
    AdaECE = np.sum(gaps * counts) / len(y_true)
    RMSCE = np.sqrt(np.sum(gaps_sq * counts) / len(y_true))
    min_np = np.minimum(accuracy_list, confidence_list)
    counts = counts.astype(int)  # For consistency
    counter_zero_list = empty_bins.tolist()


    # AdaECE AUX:
    len_confidences_aux = len(confidences)
    confidences = torch.tensor(confidences)
    edges_aux = np.interp(np.linspace(0, len_confidences_aux, nbins+1), np.arange(len_confidences_aux), np.sort(confidences))
    n_aux, bin_boundaries_aux = np.histogram(confidences.cpu().detach(), edges_aux) # n is the number of instances per bin
    # print(n,confidences,bin_boundaries)
    bin_lowers_aux = bin_boundaries_aux[:-1]
    bin_uppers_aux = bin_boundaries_aux[1:]
    y_pred_aux = torch.tensor(y_pred)
    y_true_aux = torch.tensor(y_true)
    accuracies_aux = y_pred_aux.eq(y_true_aux)
    ece_aux = torch.zeros(1, device=confidences.device)
    for bin_lower_aux, bin_upper_aux in zip(bin_lowers_aux, bin_uppers_aux):
        # Calculated |confidence - accuracy| in each bin
        in_bin_aux = confidences.gt(bin_lower_aux.item()) * confidences.le(bin_upper_aux.item())
        prop_in_bin_aux = in_bin_aux.float().mean()
        if in_bin_aux.any() > 0:
            accuracy_in_bin_aux = accuracies_aux[in_bin_aux].float().mean()
            avg_confidence_in_bin_aux = confidences[in_bin_aux].mean()
            ece_aux += torch.abs(avg_confidence_in_bin_aux - accuracy_in_bin_aux) * prop_in_bin_aux

    return accuracy_list.tolist(), confidence_list.tolist(), counts.tolist(), gaps.tolist(), AdaMCE.tolist(), AdaECE.tolist(), RMSCE.tolist(), min_np.tolist(), counter_zero_list, ece_aux.item(), n, bin_edges


def calculate_ClasswiseECE_confidence_accuracy_multiclass(y_true, y_pred, confidences, nbins, nclasses, all_confidences, args = None):
    """
    Confidence and accuracy calculation for multiclass reliability Diagram - Optimized
        # FOR THE MULTICLASS CASE IMPORTANT: IN THE BINARY CASE IT IS NOT THE ACCURACY!!!!!!!! IT IS THE RELATIVE FREQUENCY!!!!!!!! https://towardsdatascience.com/introduction-to-reliability-diagrams-for-probability-calibration-ed785b3f5d44

    Version from https://github.com/torrvision/focal_calibration/blob/475422a6754e20d25216158a874749c33e826e21/Metrics/metrics.py#L191
    """

    if args!=None:
        nclasses = nclasses if args.loss == 'ce' or args.loss == 'focal' or args.loss == 'focalAdaptive' or args.loss == 'dece' or args.loss == 'brierScore' or args.loss == 'csc' or args.dirichlet else nclasses+1

    # First, part in M bins of size 1/M
    y_true = np.array(torch.stack(y_true).cpu()).tolist() if (
                isinstance(y_true, torch.Tensor) or isinstance(y_true, list)) else y_true.tolist()

    bm = 1 / nbins  # size of each bin
    bin_edges = np.floor(np.linspace(0, 1, nbins + 1) * 10) / 10
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Bin lowers and bin uppers
    bin_edges = np.array(bin_edges) if not isinstance(bin_edges, np.ndarray) else bin_edges

    all_confidences = np.array(all_confidences)
    print(all_confidences.shape)
    for k in range(nclasses):
        # Initialize arrays
        confidence_sums = np.zeros(nbins)
        accuracy_sums = np.zeros(nbins)
        counts = np.zeros(nbins)

        # Confidences of the class:
        class_confidences = all_confidences[:, k]

        # Digitize confidences into bins:
        class_confidences = np.floor(np.array(
            class_confidences) * 10) / 10  # To avoid that 0.6999 belongs to 0.7 (It belongs to [0.6,0.7) bin, no to [0.7,0.8))
        # bin_indices = np.digitize(confidences, bin_edges, right=False) - 1 # -1 because the list starts at 0
        class_confidences = np.array(class_confidences) if not isinstance(class_confidences,
                                                                          np.ndarray) else class_confidences
        bin_indices = np.searchsorted(bin_edges, class_confidences, side='left')
        bin_indices = np.clip(bin_indices, 0, nbins - 1)  # Ensure indices are within the range


        # Populate bins
        for i in range(len(class_confidences)):
            bin_idx = bin_indices[i]
            confidence_sums[bin_idx] += class_confidences[i]
            accuracy_sums[bin_idx] += (k == y_true[i])
            counts[bin_idx] += 1


        # Handle empty bins
        empty_bins = counts == 0
        confidence_sums[empty_bins] = bin_centers[empty_bins]
        accuracy_sums[empty_bins] = bin_centers[empty_bins]
        counts[empty_bins] = 1

        # confidence_list # accuracy_list # counter_list  # ECE_list, RMSCE_list, gap_list, min_list, counter_zero_list ????
        # Compute metrics:
        confidence_list = confidence_sums / counts
        accuracy_list = accuracy_sums / counts
        gaps = np.abs(accuracy_list - confidence_list)
        gaps_sq = (accuracy_list - confidence_list) ** 2
        MCE = np.max(gaps)
        ECE = np.sum(gaps * counts) / len(y_true)
        RMSCE = np.sqrt(np.sum(gaps_sq * counts) / len(y_true))
        min_np = np.minimum(accuracy_list, confidence_list)
        counts = counts.astype(int)  # For consistency
        counter_zero_list = empty_bins.tolist()

        if (k == 0):
            per_class_sce = torch.Tensor([ECE.item()])
            accuracy_list_sce = torch.Tensor(accuracy_list)
            confidence_list_sce = torch.Tensor(confidence_list)
            counts_sce = torch.Tensor(counts)
            gaps_sce = torch.Tensor(gaps)
            MCE_sce = torch.Tensor([MCE.item()])
            RMSCE_sce = torch.Tensor([RMSCE.item()])
            min_np_sce = torch.Tensor(min_np)
            counter_zero_list_sce = torch.Tensor(counter_zero_list)
        else:
            per_class_sce = torch.cat((per_class_sce, torch.Tensor([ECE.item()])), dim = 0)
            accuracy_list_sce = torch.cat((accuracy_list_sce, torch.Tensor(accuracy_list)), dim = 0)
            confidence_list_sce = torch.cat((confidence_list_sce, torch.Tensor(confidence_list)), dim = 0)
            counts_sce = torch.cat((counts_sce, torch.Tensor(counts)), dim = 0)
            gaps_sce = torch.cat((gaps_sce, torch.Tensor(gaps)), dim = 0)
            MCE_sce = torch.cat((MCE_sce, torch.Tensor([MCE.item()])), dim = 0)
            RMSCE_sce = torch.cat((RMSCE_sce, torch.Tensor([RMSCE.item()])), dim = 0)
            min_np_sce = torch.cat((min_np_sce, torch.Tensor(min_np)), dim = 0)
            counter_zero_list_sce = torch.cat((counter_zero_list_sce, torch.Tensor(counter_zero_list)), dim = 0)

    sce = torch.mean(per_class_sce)
    mce_sce = torch.mean(MCE_sce)
    rmce_sce = torch.mean(RMSCE_sce)

    return str(np.array(accuracy_list_sce).tolist()), str(np.array(confidence_list_sce).tolist()), str(np.array(counts_sce).tolist()), str(np.array(gaps_sce).tolist()), str(mce_sce.tolist()), str(sce.tolist()), str(rmce_sce.tolist()), str(np.array(min_np_sce).tolist()), str(np.array(counter_zero_list_sce).tolist())

def AUC_shared_evaluation(y_test, y_all_confidences, num_classes, args=None):
    """
    Return AUC
    """
    # Calculate the AUC [One vs Rest]
    ## If multiclass, make the labels binary:
    y_gt = (lambda x: np.array(torch.stack(x).cpu()) if isinstance(x, list) and isinstance(x[0], torch.Tensor) else x) (y_test)

    if args!=None:
            y_all_confidences = y_all_confidences if args.loss == 'ce' or args.loss == 'focal' or args.loss == 'focalAdaptive' or args.loss == 'dece' or args.loss == 'brierScore' or args.loss == 'csc' else y_all_confidences[:, :num_classes]

    y_true_binary = label_binarize(y_gt, classes=list(range(num_classes)))
    ## AUC:
    auc = roc_auc_score(y_true_binary, y_all_confidences, multi_class='ovr')
    return auc

def calculate_confidence_relativefrequency_binary(y_true, y_pred, nbins):
    """
    Calculate confidence - Accuracy. Binary Methods. Sklearn
    Returns:
        relativefrequency_list, confidence_list, counter_list, gap_list, MCE, ECE, RMSCE, min_list
    """

    # GOOD CODE!! IMPORTANT: IN THE BINARY CASE IT IS NOT THE ACCURACY!!!!!!!! IT IS THE RELATIVE FREQUENCY!!!!!!!! https://towardsdatascience.com/introduction-to-reliability-diagrams-for-probability-calibration-ed785b3f5d44
    # First, part in M bins of size 1/M
    bm = 1/nbins
    confidence_list, relativefrequency_list, counter_list, ECE_list, RMSCE_list, gap_list, min_list, counter_zero_list = [], [], [], [], [], [], [], []
    # Calculate the confidence and relative frequency of positives per bin
    bin_fin = bm
    MCE = 0
    for bin_ini in  np.arange(0,1,bm):
        confidence_bm, relativefrequency_bm, counter, sum_TP_TN, counter_zero = 0, 0, 0, 0, False
        y_pred_standard = [1 if i>0.5 else 0 for i in y_pred]
        for i in range(len(y_pred)):
            if (y_pred[i] >= bin_ini and y_pred[i] < bin_fin) or (bin_fin == 1 and y_pred[i] == bin_fin):

                # Confidence
                confidence_bm+=y_pred[i]
                # relativefrequency
                sum_TP_TN+=y_true[i]
                counter+=1
        if counter == 0:
            confidence_bm = (bin_ini + bin_fin) / 2
            relativefrequency_bm = (bin_ini + bin_fin) / 2
            counter = 1
            counter_zero = True
        else:
            confidence_bm = confidence_bm/counter
            relativefrequency_bm = sum_TP_TN / counter
        confidence_list.append(confidence_bm)
        relativefrequency_list.append(relativefrequency_bm)
        counter_list.append(counter)
        counter_zero_list.append(counter_zero)
        bin_fin+=bm
        gap = np.abs(relativefrequency_bm - confidence_bm)
        gap_sq = (relativefrequency_bm - confidence_bm)**2
        if gap >= MCE:
            MCE = gap
        gap_list.append(gap)
        min_list.append(np.minimum(relativefrequency_bm, confidence_bm))
        ECE_list.append((abs(counter) * gap))
        RMSCE_list.append(abs(counter) * gap_sq)

    ECE = sum(ECE_list)/ len(y_true)
    RMSCE = np.sqrt(sum(RMSCE_list)/ len(y_true))

    return relativefrequency_list, confidence_list, counter_list, gap_list, MCE, ECE, RMSCE, min_list, counter_zero_list


def plot_reliability_diagram(prob_true, prob_pred, model_name, type_code, key, calib_name = "", path = "", loss = "", ax=None):
    # Plot the calibration curve for ResNet in comparison with what a perfectly calibrated model would look like
    if ax==None:
        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()
    else:
        plt.sca(ax)

    plt.plot([0, 1], [0, 1], color="#FE4A49", linestyle=":", label="Perfectly calibrated model")
    plt.plot(prob_pred, prob_true, "s-", label=model_name, color="#162B37")

    if type_code == "sklearn":
        plt.ylabel("Fraction of positives (Relative Fecuency)", fontsize=16)
    else:
        plt.ylabel("Fraction of positives ("+type_code+")", fontsize=16)

    plt.xlabel("Mean predicted value (Confidence)", fontsize=16,)

    plt.legend(fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.grid(True, color="#B2C7D9")

    plt.title("Reliability Diagram "+key)

    plt.tight_layout()

    if path != "":
        plt.savefig(path + f"plt_reliability_diagram_{model_name}_{loss}.png",)
    else:
        plt.savefig("calibration/reliab_diagrams/plt_"+calib_name+'reliability_diagram_both_'+key+'.png')

def plot_reliability_diagram_some_calibrators(results_list, calib_names, key , ax=None):
    # Plot the calibration curve for ResNet in comparison with what a perfectly calibrated model would look like
    if ax==None:
        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()
    else:
        plt.sca(ax)

    plt.plot([0, 1], [0, 1], color="#FE4A49", linestyle=":", label="Perfectly calibrated model")

    for id, r in enumerate(results_list): # relfreq_list_owncode2, confidence_list_owncode2
        plt.plot(r[1], r[0], "s-", label=calib_names[id])
    plt.ylabel("Fraction of positives (Relative Fecuency)", fontsize=16)
    plt.xlabel("Mean predicted value (Confidence)", fontsize=16,)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, color="#B2C7D9")
    plt.title("Reliability Diagram "+key)
    plt.tight_layout()
    plt.savefig('reliab_diagrams/reliability_diagram_allCalMethods_'+key+'.png')

def plot_reliability_diagram_plotly(prob_true, prob_pred, type_code, counter_list, counter_zero_list, key, gap_list, ECE, MCE, min_list, model_name = "", calib_name = "", onlyReliabilityDiagram = False, path = "", loss="", ax=None):
    try:
        print("Plotting reliability Diagram")
        num_times_0 = 0
        prob_true_b = prob_true.copy()
        prob_pred_b = prob_pred.copy()
        x = [0]*(5-num_times_0)+[1]*(5-num_times_0)
        y = [0]*(5-num_times_0)+[1]*(5-num_times_0)
        perf = ["no_perfectly"]*(10-num_times_0)+["perfectly"]*(10-2*num_times_0)
        prob_true_b = prob_true_b + y
        prob_pred_b = prob_pred_b + x
        #print(prob_true_b, prob_pred_b, perf)
        data = {"prob_true":prob_true_b, "prob_pred": prob_pred_b, "perf": perf}
        df = pd.DataFrame(data)
        trace1 = px.line(df, x="prob_pred", y="prob_true", color='perf',line_dash='perf', markers=True,
                         color_discrete_map={
                             'no_perfectly': 'blue',
                             'perfectly': 'green'
                         },
                         labels={
                             'perf': "Legend",
                             'prob_pred': 'Mean predicted value (Confidence)',
                             'prob_true': 'Fraction of positives (Relative Frequency)'
                         },
                         title="Reliability Diagram " + key
                         )
        newnames = {'no_perfectly': calib_name+'LCRNExtendTorch', 'perfectly': 'Perfectly calibrated model'}  # From the other post

        trace1.for_each_trace(lambda t: t.update(name=newnames[t.name]))

        trace1.update_xaxes(tickmode = 'array',
                     tickvals = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
        trace1.update_yaxes(tickmode='array',
                            tickvals=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        trace1.add_annotation(
            text=("ECE = %.2f" % (ECE*100)), showarrow=False, xref="paper",
            yref='paper', x=0.97, y=0.07, xshift=-1, yshift=-5, font=dict(size=20, color="black"))
        trace1.add_annotation(
            text=("MCE = %.2f" % (MCE*100)), showarrow=False, xref="paper",
            yref='paper', x=0.97, y=0.05, xshift=-1, yshift=-5, font=dict(size=20, color="black"))

        trace1.update_layout(font=dict(size=20), legend={'traceorder': 'normal'})


        bar_list = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        data = {"counter_list": gap_list, "bar_list": bar_list, "min_list": min_list,
                "prob_pred": prob_pred, "prob_true":prob_true}
        df_aux = pd.DataFrame(data)
        trace_bars = px.bar(df_aux,
                           x="bar_list",
                           y="prob_true",
                           labels={
                               "bar_list": "Mean Predicted value (Confidence)",
                               'counter_list': 'Count'
                           },
                           title="GAPS " + key
                           )
        trace_bars.update(layout_showlegend=True)
        trace_bars.update_layout(font=dict(size=20))
        trace_bars.update_xaxes(tickmode='array',
                            tickvals=[0.0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0])

        bar_list = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        color_list = ["red"] * len(bar_list)
        data = {"counter_list": gap_list,"counter_list_r": np.around(gap_list,4), "bar_list": bar_list, "min_list": min_list, "color_list":color_list, "prob_pred": prob_pred}
        df_aux = pd.DataFrame(data)
        trace_gap = px.bar(df_aux,
                                    x="prob_pred",
                                    y="counter_list",
                                    base=min_list,
                                    text="counter_list_r",
                                    color_discrete_sequence=color_list,
                                    labels={
                                        "bar_list": "Mean Predicted value (Confidence)",
                                        'counter_list': 'Count'
                                    },
                                    title="bars " + key
                                    )
        trace_gap.update(layout_showlegend=True)
        trace_gap.update_layout(font=dict(size=20))
        trace_gap.update_xaxes(tickmode="linear")
        trace_gap.update_traces(width = 0.005)
        trace_gap.update_traces(textposition='outside', textfont_size=10,textfont_color="black")
        trace_gap.update_layout(uniformtext_minsize=15, uniformtext_mode='show')


        all_figures = [trace1, trace_bars, trace_gap]
        fig = go.Figure(data=(trace1.data + trace_gap.data))
        fig.update_layout(font=dict(size=20), title=calib_name+"Reliability Diagram - " + key,
                              xaxis_title='Mean predicted value (Confidence)',
                              yaxis_title='Fraction of positives ('+type_code+')')
        fig.add_annotation(
            text=("ECE = %.2f" % (ECE*100)), showarrow=False, xref="paper",
            yref='paper', x=0.97, y=0.09, xshift=-1, yshift=-5, font=dict(size=20, color="black"))
        fig.add_annotation(
            text=("MCE = %.2f" % (MCE*100)), showarrow=False, xref="paper",
            yref='paper', x=0.97, y=0.05, xshift=-1, yshift=-5, font=dict(size=20, color="black"))
        fig.update_xaxes(tickmode='array',
                            tickvals=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        fig.update_yaxes(tickmode='array',
                            tickvals=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        fig.update_layout(uniformtext_minsize=12, uniformtext_mode='show')
        #fig.show()

        fig.write_image("calibration/reliab_diagrams/doing/"+calib_name+'reliability_diagram_plotly_bars_' + key + '.png',
                            width=1500, height=1000)


        bar_list = [0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]
        counter_list_distrib = np.array(counter_list) - (1 * np.array(counter_zero_list))
        data = {"counter_list":counter_list_distrib, "bar_list":bar_list}
        df_aux = pd.DataFrame(data)
        trace_distribution = px.bar(df_aux,
                                                x="bar_list",
                                                y="counter_list",
                                                text_auto='.0s',
                                                color_discrete_map={
                                                    'num_predators': 'red',
                                                    'num_no_predators': 'blue'
                                                },
                                                labels={
                                                    "bar_list": "Mean Predicted value (Confidence)",
                                                    'counter_list': 'Count'
                                                },
                                                title="Samples per bin LCRNExtendTorch Normalized " + key
                                                      )
        trace_distribution.update(layout_showlegend=True)
        trace_distribution.update_layout(font=dict(size=20))
        trace_distribution.update_xaxes(tickmode='array',
                                tickvals=[0.0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0])

        trace_distribution.write_image("calibration/reliab_diagrams/doing/"+calib_name+'reliability_diagram_distribution_' + key + '.png',
                                           width=1300, height=800)



        images = [Image.open(x) for x in ["calibration/reliab_diagrams/doing/"+calib_name+'reliability_diagram_plotly_bars_'+key+'.png', "calibration/reliab_diagrams/doing/"+calib_name+'reliability_diagram_distribution_'+key+'.png']]
        widths, heights = zip(*(i.size for i in images))

        total_width = max(widths)
        max_height = sum(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        y_offset = 0

        for im in images:
            new_im.paste(im, (0, y_offset))
            y_offset += im.size[1]

        if path != "":
            new_im.save(path + f"reliability_diagram_{model_name}_{loss}.png",)
        else:
            new_im.save("calibration/reliab_diagrams/"+str(onlyReliabilityDiagram)+"_"+calib_name+'reliability_diagram_both_'+key+'.png')
    except ValueError as ve:
        print("Impossible to print plotly reliability diagram. No same length")