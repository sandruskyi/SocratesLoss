"""
@User: sandruskyi

"""
import numpy as np
import pandas as pd
import argparse
from sklearn.calibration import calibration_curve
from plotly.subplots import make_subplots
import os
from PIL import Image

np.random.seed(42) # To have the same model all the time

# Calibration
from calibration_methods import platt_scaling_fit, platt_scaling_calibration, KDD_workshop_PlattScaling_fit, KDD_workshop_PlattScaling_calibration, KDD_workshop_IsotonicRegression_fit, KDD_workshop_IsotonicRegression_calibration, BetaCalibration_fit, BetaCalibration_calibration, HistogramBinning_fit, HistogramBinning_calibration
from scipy.special import logit # In Platt Scaling and Temperature Scaling we need to use the logits, no the final predictions

# Reliability Diagram
from plot_reliability_diagram import calculate_confidence_accuracy_sklearn, calculate_confidence_accuracy_multiclass, calculate_confidence_relativefrequency_binary, plot_reliability_diagram, plot_reliability_diagram_some_calibrators, plot_reliability_diagram_plotly

# Raw confidences vs scaled confidences curves, counts above thresholds curve (Zac Paper), ROC curve, precision-recall curve, FPRvsFNR_curve
from UsefulCurves import plot_rawconf_vs_scaledconf_plotly, counts_above_tresholds, counts_above_tresholds_plot, roc_curve_plot, FPRvsFNR_curve_plot, precision_recall_curve_plot

# IMPORTANT: IN  THE BINARY CASE IT IS NOT THE ACCURACY!!!!!!!! IT IS THE RELATIVE FREQUENCY!!!!!!!! https://towardsdatascience.com/introduction-to-reliability-diagrams-for-probability-calibration-ed785b3f5d44

def load_pred(path, onlyReliabilityDiagram = False, debug = "False"):
    """
    This method loads the predictions and their ids
    """
    data = np.load(path)

    if debug:
        train_pred = data["train"][:,:10]
        test_pred = data["test"][:,10]
        train_ids_pred = data["train_ids"][:10]
        test_ids_pred = data["test_ids"][:10]
    else:
        train_pred = data["train"]
        test_pred = data["test"]
        train_ids_pred = data["train_ids"]
        test_ids_pred = data["test_ids"]
        if onlyReliabilityDiagram == False:
            val_pred = data["val"]
            val_ids_pred = data["val_ids"]
            return train_pred, val_pred, test_pred, train_ids_pred, val_ids_pred, test_ids_pred
    return train_pred, test_pred, train_ids_pred, test_ids_pred


def main(debug, pred_modelx_path, n_bins, onlyReliabilityDiagram):
    if onlyReliabilityDiagram:
        train_pred_model, test_pred_model, train_ids_pred_model, test_ids_pred_model = load_pred(pred_modelx_path, onlyReliabilityDiagram, debug)
    else:
        train_pred_model, val_pred_model, test_pred_model, train_ids_pred_model, val_ids_pred_model, test_ids_pred_model = load_pred(pred_modelx_path, onlyReliabilityDiagram, debug)

    print(train_pred_model)
    print(train_ids_pred_model)

    if onlyReliabilityDiagram:
        dict_reli = {"train": train_pred_model, "test": test_pred_model}
        dict_reli_ids = {"train": train_ids_pred_model, "test": test_ids_pred_model}
    else:
        dict_reli = {"train":train_pred_model, "val":val_pred_model, "test":test_pred_model}
        dict_reli_ids = {"train": train_ids_pred_model, "val": val_ids_pred_model, "test": test_ids_pred_model}
    methods_names = ["Original", "Platt_scaling_LR", "Platt_scaling_KDD", "IsotonicRegression", "HistogramBinning", "BetaCalibration"]
    train_results = []
    val_results = []
    calib = False
    for key in dict_reli:
        y_true = dict_reli[key][1]
        y_pred = dict_reli[key][0]

        roc_curve_plot(y_true, y_pred, key)
        precision_recall_curve_plot(y_true, y_pred, key)
        FPRvsFNR_curve_plot(y_true, y_pred, key)

        relfreq_list_owncode2, confidence_list_owncode2, counter_list, gap_list, MCE, ECE, RMSCE, min_list, counter_zero_list = calculate_confidence_relativefrequency_binary(y_true, y_pred, n_bins)
        if key=="train":
            train_results.append([relfreq_list_owncode2, confidence_list_owncode2])
        elif key=="val":
            val_results.append([relfreq_list_owncode2, confidence_list_owncode2])
        print("ECE, MCE, RMSCE ",ECE, MCE, RMSCE)
        plot_reliability_diagram_plotly(relfreq_list_owncode2, confidence_list_owncode2, pred_modelx_path.split("/")[-1].split("_")[0], counter_list, counter_zero_list, key, gap_list, ECE, MCE, min_list, "", onlyReliabilityDiagram)

    if onlyReliabilityDiagram:
        exit(1)

    fit_methods = {"Platt_scaling_LR_":platt_scaling_fit, "Platt_scaling_KDD_": KDD_workshop_PlattScaling_fit, "IsotonicRegression_": KDD_workshop_IsotonicRegression_fit, "HistogramBinning_": HistogramBinning_fit, "BetaCalibration_": BetaCalibration_fit}
    calib_methods = {"Platt_scaling_LR_":platt_scaling_calibration, "Platt_scaling_KDD_": KDD_workshop_PlattScaling_calibration, "IsotonicRegression_": KDD_workshop_IsotonicRegression_calibration, "HistogramBinning_": HistogramBinning_calibration, "BetaCalibration_": BetaCalibration_calibration}
    dfs_above_trsh_final = pd.DataFrame()
    for method_key in fit_methods:
        df_above_trsh = {}
        print("Train dataset " + method_key)
        key = "train"
        y_true = dict_reli[key][1]
        y_pred = dict_reli[key][0]
        calib = True

        if method_key == "Platt_scaling_LR_" or method_key=="Platt_scaling_KDD_":
            y_pred = logit(y_pred)  # To step back the logistic function of the last layer of the model

        # Fit
        fitting_minimiser = fit_methods[method_key](y_pred, y_true)
        # Predict train
        y_pred_calibrated, y_pred_true_calibrated = calib_methods[method_key](fitting_minimiser, y_pred, y_true)
        y_pred_true_calibrated_train = y_pred_true_calibrated

        roc_curve_plot(y_true, y_pred_calibrated, key, method_key)
        precision_recall_curve_plot(y_true, y_pred_calibrated, key, method_key)
        FPRvsFNR_curve_plot(y_true, y_pred_calibrated, key, method_key)


        relfreq_list_owncode2, confidence_list_owncode2, counter_list, gap_list, MCE, ECE, RMSCE, min_list, counter_zero_list = calculate_confidence_relativefrequency_binary(y_true, y_pred_calibrated, n_bins)
        train_results.append([relfreq_list_owncode2, confidence_list_owncode2])
        print("relfreq_list_owncode2", relfreq_list_owncode2)
        print("confidence_list_owncode2", confidence_list_owncode2)
        plot_reliability_diagram_plotly(relfreq_list_owncode2, confidence_list_owncode2, pred_modelx_path.split("/")[-1].split("_")[0], counter_list, counter_zero_list, key, gap_list, ECE, MCE, min_list, method_key)
        print("1")
        plot_rawconf_vs_scaledconf_plotly(y_true, y_pred, y_pred_calibrated,key , method_key)
        print("2")

        percentage_above_thresholds, accuracies, thresholds = counts_above_tresholds(y_true, y_pred, y_pred_calibrated, key, method_key)
        df_above_trsh["percentage_above_thresholds_train"] = percentage_above_thresholds
        df_above_trsh["accuracies_train"] = accuracies
        df_above_trsh["method"] = method_key
        df_above_trsh["thresholds"] = thresholds
        print("Train finished")
        # Predict val
        print("Val dataset " + method_key)
        key = "val"
        y_true = dict_reli[key][1]
        y_pred = dict_reli[key][0]


        if method_key == "Platt_scaling_LR_" or method_key=="Platt_scaling_KDD_" :
            y_pred = logit(y_pred)  # To step back the logistic function of the last layer of the model

        y_pred_calibrated, y_pred_true_calibrated = calib_methods[method_key](fitting_minimiser, y_pred, y_true)
        y_pred_true_calibrated_val = y_pred_true_calibrated

        ##
        # To plot the final plot threshold meeting
        np.savez("calibrated_predictions/"+method_key + "_predictions_LCRNExtendTorch.npz", train = y_pred_true_calibrated_train, test = y_pred_true_calibrated_val, train_ids = dict_reli_ids["train"], test_ids = dict_reli_ids["val"])
        ##


        roc_curve_plot(y_true, y_pred_calibrated, key, method_key)
        precision_recall_curve_plot(y_true, y_pred_calibrated, key, method_key)
        FPRvsFNR_curve_plot(y_true, y_pred_calibrated, key, method_key)



        relfreq_list_owncode2, confidence_list_owncode2, counter_list, gap_list, MCE, ECE, RMSCE, min_list, counter_zero_list = calculate_confidence_relativefrequency_binary(y_true, y_pred_calibrated, n_bins)
        print("relfreq_list_owncode2", relfreq_list_owncode2)
        print("confidence_list_owncode2", confidence_list_owncode2)

        val_results.append([relfreq_list_owncode2, confidence_list_owncode2])
        plot_reliability_diagram_plotly(relfreq_list_owncode2, confidence_list_owncode2, pred_modelx_path.split("/")[-1].split("_")[0], counter_list, counter_zero_list, key, gap_list, ECE, MCE, min_list, method_key)

        plot_rawconf_vs_scaledconf_plotly(y_true, y_pred, y_pred_calibrated, key, method_key)
        percentage_above_thresholds, accuracies, thresholds = counts_above_tresholds(y_true, y_pred, y_pred_calibrated, key, method_key)
        df_above_trsh["percentage_above_thresholds_val"] = percentage_above_thresholds
        df_above_trsh["accuracies_val"] = accuracies
        dfs_above_trsh = pd.DataFrame(df_above_trsh)
        if dfs_above_trsh_final.shape[0]==0:
            dfs_above_trsh_final = dfs_above_trsh.copy()
        else:
            dfs_above_trsh_final = pd.concat([dfs_above_trsh_final, dfs_above_trsh], axis = 0)
        print("Val finished")

    plot_reliability_diagram_some_calibrators(train_results, methods_names, "train", ax=None)
    plot_reliability_diagram_some_calibrators(val_results, methods_names, "val", ax=None)

    print("ALL CALIBRATION PLOTTED")
    # Create raw vs calibrated plot and % above threshold plot
    keys = ["train","val"]
    for key in keys:
        raw_vs_cal_list = []
        for method_key in fit_methods:
            raw_vs_cal_list.append("reliab_diagrams/doing/"+method_key+'rawvsscaled_'+key + '.png')
        images = [Image.open(x) for x in raw_vs_cal_list]
        widths, heights = zip(*(i.size for i in images))
        total_width = max(widths)
        max_height = sum(heights)
        new_im = Image.new('RGB', (total_width, max_height))
        y_offset = 0
        for im in images:
            new_im.paste(im, (0, y_offset))
            y_offset += im.size[1]
        new_im.save("reliab_diagrams/rawvsscaled_"+key + ".png")

        counts_above_tresholds_plot(dfs_above_trsh_final, key)
    print("ALL CURVES PLOTTED")



if __name__ == '__main__':
    directory = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=bool, default=False)

    parser.add_argument("--predModel4Path", type=str, default="/".join(directory.split("\\")[:-1]) + "/predictions.npz")

    parser.add_argument("--nbins", type=int, default=10)
    parser.add_argument("--onlyReliabilityDiagram", type=bool, default=True)
    args = parser.parse_args()
    main(args.debug, args.predModel4Path, args.nbins, args.onlyReliabilityDiagram)