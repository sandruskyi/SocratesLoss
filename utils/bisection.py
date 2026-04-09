"""
Extended by
@User: sandruskyi
"""
import torch
import math
import numpy as np


__all__ = ['bisection_method', 'bisection_method_LeoFeng']

# this function is used to evaluate the accuracy on validation set and test set per coverage
def bisection_method(abortion, correct, results, expected_coverage):

    print("Bisection method")
    upper = 1.
    while True:
        mask_up = abortion <=  upper

        passed_up = torch.sum(mask_up.long()).item()
        if passed_up/len(correct)*100.<expected_coverage[0]: upper *= 2.
        else: break
    test_thres = 1.
    for coverage in expected_coverage:
        print("Doing coverage ", coverage)

        if coverage == 0:
            results.append((0, 1)) # results[0] = final coverage, results[0] = accuracy, 1-acc=empirical risk
        else:
            mask = abortion <=  test_thres
            passed = torch.sum(mask.long()).item()
            # bisection method start
            lower = 0.
            #print("Start while")
            #print("abortion shape", abortion.shape)
            while (math.fabs(passed/len(correct)*100.-coverage) > 0.3) and (round(upper, 5)!=round(lower, 5)):
                #print("passed", passed, "passed/len(correct)*100.-coverage", passed/len(correct)*100.-coverage, "passed/len(correct)*100.", passed/len(correct)*100., "coverage", coverage)
                #print("test_thres", test_thres, "upper", upper, "lower", lower )
                if passed/len(correct)*100.>coverage:
                    upper = min(test_thres,upper)
                    test_thres=(test_thres+lower)/2
                    #print("1 test_thres", "upper", upper, "lower", lower)

                elif passed/len(correct)*100. < coverage:
                    lower = max(test_thres,lower)
                    test_thres=(test_thres+upper)/2
                    #print("2 test_thres", "upper", upper, "lower", lower)
                if (round(upper, 5)==round(lower, 5)):
                    print("Finishing bisection method for upper == lower")
                mask = abortion <=  test_thres
                passed = torch.sum(mask.long()).item()
                # bisection method end
            #print("Finish while")
            masked_correct = correct[mask]
            correct_data = torch.sum(masked_correct.long()).item()
            if passed == 0:
                passed_acc = 0
            else:
                passed_acc = correct_data/passed
            results.append((passed/len(correct), passed_acc, test_thres)) # results[0] = final coverage, results[0] = accuracy, 1-acc=empirical risk
            # print('coverage {:.0f} done'.format(coverage))

    print("Bisection method finished")


def bisection_method_LeoFeng(abortion, correct, results, expected_coverage):

    print("Bisection method LeoFeng")
    def calc_threshold(val_tensor,cov): # Coverage is a perentage in this input
        threshold=np.percentile(np.array(val_tensor), 100-cov*100)
        return threshold


    neg_score = -abortion
    for coverage in expected_coverage: # Coverage is a number from 0 to 100 here
        threshold = calc_threshold(neg_score, coverage/100)

        mask = (neg_score >= threshold)

        nData = len(correct)
        nSelected = mask.long().sum().item()
        isCorrect = correct[mask]
        nCorrectSelected = isCorrect.long().sum().item()
        passed_acc = nCorrectSelected/nSelected
        results.append((nSelected/nData, passed_acc, threshold))

    print("Bisection method LeoFeng finished")

