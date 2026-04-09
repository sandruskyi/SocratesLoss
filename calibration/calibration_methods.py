"""
@User: sandruskyi

Calibration methods Fit and Calibration Phase
"""
import numpy as np
import pandas as pd
#Calibration:
from sklearn.linear_model import LogisticRegression # Platt Scaling
from scipy.optimize import minimize # Platt Scaling v2
from scipy.special import logit, expit # Platt Scaling v2
from sklearn.isotonic import IsotonicRegression # IsotonicRegression
from betacal import BetaCalibration # BetaCalibration
np.random.seed(42) # To have the same model all the time


methods_names = ["Original", "Platt_scaling_LR", "Platt_scaling_KDD", "IsotonicRegression", "HistogramBinning",
                 "BetaCalibration"]


def get_y_pred_true_calibrated(y_pred_calibrated, y_true):
    y_pred_true_calibrated = [[], []]
    if len(y_pred_calibrated) == len(y_true):
        y_pred_true_calibrated[0] = list(y_pred_calibrated)
        y_pred_true_calibrated[1] = list(y_true)
    else:
        print(len(y_pred_calibrated), len(y_true))
        exit("CALIBRATION ERROR")
    return y_pred_true_calibrated

def platt_scaling_fit(y_pred, y_gt):
    # Fit a Linear Regression model to have the parameters A and B
    lr = LogisticRegression(C=99999999999, solver='lbfgs')
    lr.fit(y_pred.reshape(-1,1), y_gt)
    return lr

def platt_scaling_calibration(lr_calibrated, y_pred, y_true):
    y_pred_calibrated = lr_calibrated.predict_proba(y_pred.reshape(-1,1))[:,1]
    y_pred_true_calibrated = get_y_pred_true_calibrated(y_pred_calibrated, y_true)
    return y_pred_calibrated, y_pred_true_calibrated


def KDD_workshop_PlattScaling_fit(y_pred_logits, y_true):
    #y_logits = logit(y_pred)  # To step back the logistic function of the last layer of the model

    def scale_fun_bce(x, *args):
        a, b = x
        y_logit_scaled = a * y_pred_logits + b
        y_pred_inner = expit(y_logit_scaled)
        bce = sum(
            [-(y_t * np.log(y_p) + (1 - y_t) * np.log(1 - y_p)) for y_t, y_p in zip(y_true[:1000], y_pred_inner)
             if not y_p == 0])
        return bce

    min_obj = minimize(scale_fun_bce, [1, 0], method='Nelder-Mead', options={'xatol': 1e-8, 'disp': True})
    return min_obj

def KDD_workshop_PlattScaling_calibration(min_obj, y_pred_logits, y_true):
    y_pred_calib = expit(min_obj.x[0] * y_pred_logits + min_obj.x[1])
    y_pred_true_calibrated = get_y_pred_true_calibrated(y_pred_calib, y_true)
    return y_pred_calib, y_pred_true_calibrated

def KDD_workshop_IsotonicRegression_fit(confidence_list, relfreq_list):
    iso_reg = IsotonicRegression().fit(confidence_list, relfreq_list)
    return iso_reg

def KDD_workshop_IsotonicRegression_calibration(iso_reg_fit, y_pred, y_true):
    y_pred_calibrated = iso_reg_fit.predict(y_pred)
    y_pred_true_calibrated = get_y_pred_true_calibrated(y_pred_calibrated, y_true)
    return y_pred_calibrated, y_pred_true_calibrated

def BetaCalibration_fit(y_pred, y_true):
    # https://github.com/REFRAME/betacal/blob/master/python/tutorial/Python%20tutorial.ipynb
    # Fit three-parameter beta calibration
    bc = BetaCalibration(parameters="abm").fit(y_pred.reshape(-1, 1), y_true)
    return bc

def BetaCalibration_calibration(bc, y_pred, y_true):
    y_pred_calibrated = bc.predict(y_pred)
    y_pred_true_calibrated = get_y_pred_true_calibrated(y_pred_calibrated, y_true)
    return y_pred_calibrated, y_pred_true_calibrated

def HistogramBinning_fit(y_pred, y_true, M = 10):
    """
    Fit the calibration model, finding optimal confidences for all the bins.
    # From: https://github.com/markus93/NN_calibration/blob/91f056361d65b972c9defd7193fe393f3018ebd5/scripts/calibration/cal_methods.py#L19

    Params:
        y_pred: probabilities of data
        y_true: true labels of data
        M: number of bins
    """
    bin_size = 1/M
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)  # Set bin bounds for intervals. With M=10 => [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ]

    # Got through intervals and add confidence to list
    conf = []
    for conf_threshold in upper_bounds:
        conf_threshold_lower = conf_threshold - bin_size
        conf_threshold_upper = conf_threshold
        # Filter labels within probability range
        y_true_bin = [x[0] for x in zip(y_true, y_pred) if (
                    (x[1] > conf_threshold_lower and x[1] <= conf_threshold_upper) or (
                        x[1] == 0 and x[1] == conf_threshold_lower))]
        nr_elems = len(y_true_bin)
        if nr_elems < 1:
            conf_bin = 0
        else:
            # In essence the confidence equals to the average accuracy of a bin
            conf_bin = sum(y_true_bin)/nr_elems # Sums positive classes
        conf.append(conf_bin)
    return conf

def HistogramBinning_calibration(conf_list, y_pred, y_true, M = 10):
    bin_size = 1 / M
    upper_bounds = np.arange(bin_size, 1 + bin_size,
                             bin_size)  # Set bin bounds for intervals. With M=10 => [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ]

    for i, prob in enumerate(y_pred):
        id = np.searchsorted(upper_bounds, prob) # Search the index of each prediction inside the bins
        y_pred[i] = conf_list[id] # We change the predictions for the relative frequencies
    y_pred_true_calibrated = get_y_pred_true_calibrated(y_pred, y_true)
    return y_pred, y_pred_true_calibrated
