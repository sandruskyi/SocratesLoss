"""
@User: sandruskyi
Code to print the precision-recall curve, FPR-FNR curve, ROC, counts-above-threshold, raw confidence vs scaled confidence
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
# ROC curve, precision-recall curve
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve


def plot_rawconf_vs_scaledconf_plotly(prob_true, prob_pred, prob_cal, key, calib_name = ""):

    data = {"prob_cal":prob_cal, "prob_pred": prob_pred}
    df = pd.DataFrame(data)
    trace1 = px.scatter(df, x="prob_pred", y="prob_cal",
                     labels={
                         'perf': "Legend",
                         'prob_pred': 'Raw - Mean predicted value (Confidence)',
                         'prob_true': 'Fraction of positives (Relative Frequency)',
                         'prob_cal': 'Calibrated confidence'
                     },
                     title="Raw confidence values against confidences scaled by each calibrator " + "_".join(calib_name.split("_")[:-1])
                     )
    trace1.update_layout(font=dict(size=20), legend={'traceorder': 'normal'})

    trace1.write_image("reliab_diagrams/doing/"+calib_name+'rawvsscaled_'+key + '.png',
                        width=1500, height=1000)

def counts_above_tresholds(prob_true, prob_pred, prob_cal, key, calib_name=""):
    data = {"prob_cal": prob_cal, "prob_true":prob_true, "prob_pred": prob_pred}
    df = pd.DataFrame(data)
    df["prediction"] = np.nan
    thresholds = np.arange (0, 1, 0.01)
    percentage_above_thresholds = []
    accuracies = []
    for t in thresholds:
        df.loc[df["prob_cal"]>=t,"prediction"] = 1.0
        df.loc[df["prob_cal"]<t, "prediction"] = 0.0

        porcentage_above_threshold = (df["prob_cal"]>=t).sum()/df.shape[0]
        if (df["prob_cal"]>=t).sum() == 0:
            accuracy = 1
        else:
            accuracy = (df.loc[df["prob_cal"]>=t,"prediction"] == df.loc[df["prob_cal"]>=t,"prob_true"]).sum() / (df["prob_cal"]>=t).sum()

        percentage_above_thresholds.append(porcentage_above_threshold)
        accuracies.append(accuracy)

    return percentage_above_thresholds, accuracies, thresholds


def counts_above_tresholds_plot(df,  key):
    if key == "train":
        trace1 = px.line(df, x="percentage_above_thresholds_train", y="accuracies_train" , markers=True,
                         #text="thresholds",
                         color="method",
                         labels={
                             'percentage_above_thresholds_train': "% above theshold",
                             'accuracies_train': 'Accuracy above theshold',
                             'method': 'Calibration method'
                         },
                         title="Counts Above Tresholds"
                         )
    else:
        trace1 = px.line(df, x="percentage_above_thresholds_val", y="accuracies_val", markers=True,
                         color="method",
                         labels={
                            'percentage_above_thresholds_val': "% above theshold",
                            'accuracies_val': 'Accuracy above theshold',
                            'method': 'Calibration method'
                         },
                         title="Counts Above Tresholds"
                         )

    trace1.update_layout(font=dict(size=20))
    trace1.add_annotation(
        text=("Thresholds = np.arange (0, 1, 0.01)"), showarrow=False, xref="paper",
        yref='paper', x=0.5, y=-0.08, xshift=-1, yshift=-5, font=dict(size=10, color="gray"))
    trace1.write_image("reliab_diagrams/counts_above_tresholds_" + key + ".png",
                       width=1500, height=1000)

def roc_curve_plot(y_true, y_pred, key, calib_name = ""):
    # calculate roc curves
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # One approach would be to test the model with each threshold returned from the call roc_auc_score() and select the
    # threshold with the largest G-Mean value.
    # Calculate the g-mean for each threshold:
    gmeans = np.sqrt(tpr * (1 - fpr))
    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    # Or calculate the J statistic J = TPR+FPR;

    # plot the roc curve for the model
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label='LCRN')
    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best - Threshold: '+ str(thresholds[ix]))
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC Curve "+ calib_name + key)
    plt.legend()
    # show the plot
    plt.savefig("reliab_diagrams/roc_curve_" + calib_name + key + ".png")
    #plt.show()

def FPRvsFNR_curve_plot(y_true, y_pred, key, calib_name = ""):
    # calculate roc curves
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    fnr = 1-tpr

    # Calculate the J statistic J = TPR+FPR;
    #fscore = np.sqrt((1-fpr) * fnr)
    # locate the index of the largest g-mean
    #ix = np.argmax(fscore)
    # Or calculate the J statistic J = TPR+FPR;

    # plot the roc curve for the model
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    plt.plot(fpr, fnr, marker='.', label='LCRN')
    #plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best - Threshold: '+ str(thresholds[ix]))
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.title("FPR vs FNR Curve "+ calib_name + key)
    plt.legend()
    # show the plot
    plt.savefig("reliab_diagrams/FPRvsFNR_curve_" + calib_name + key + ".png")
    #plt.show()

def precision_recall_curve_plot(y_true, y_pred, key, calib_name = ""):
    # calculate roc curves
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

    # Another naive approach to find a threshold with best balance of precision and recall is to use the F-measure that
    # summarizes the harmonic mean of both measures (precision and recall)
    # convert to f score
    fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ix = np.argmax(fscore)

    # plot the Precision-Recall curve  for the model
    no_skill = len(y_true[y_true == 1]) / len(y_true)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(recall, precision, marker='.', label='LCRN')
    plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best - Threshold: '+ str(thresholds[ix]))
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("Precision-Recall curve " + calib_name +key)
    plt.legend()
    # show the plot
    plt.savefig("reliab_diagrams/precision_recall_curve_" + calib_name + key + ".png")
    #plt.show()