import csv
import numpy as np
import argparse
from math import sqrt


def evaluation(numFolder):
    gt = np.zeros((480,), dtype=np.int8)
    pred = np.zeros((480,), dtype=np.int8)
    with open('..\\dataset\\dreyeve\\{}\\{}_gt.txt'.format(numFolder, numFolder)) as gt_file:
        reader = csv.reader(gt_file, delimiter=',')
        i = 0
        for row in reader:
            if i == 480:
                continue
            gt[i] = row[1]
            i += 1
    with open('..\\dataset\\dreyeve\\{}\\{}_pred.txt'.format(numFolder, numFolder)) as pred_file:
        reader = csv.reader(pred_file, delimiter=',')
        i = 0
        for row in reader:
            if i == 480:
                continue
            pred[i] = row[1]
            i += 1
    tp = np.sum(np.logical_and(gt == 1, pred == 1))
    tn = np.sum(np.logical_and(gt == 0, pred == 0))
    fp = np.sum(np.logical_and(gt == 0, pred == 1))
    fn = np.sum(np.logical_and(gt == 1, pred == 0))

    if tp + fp != 0:
        precision = np.around(tp / (tp + fp), decimals=2)
    else:
        precision = 0

    if tp + fn != 0:
        recall_tpr_sensitivity = np.around(tp / (tp + fn), decimals=2)
    else:
        recall_tpr_sensitivity = 0

    if fp + tn != 0:
        tnr_specificity = np.around(tn / (fp + tn), decimals=2)
    else:
        tnr_specificity = 0

    if tp + fn + fp + tn != 0:
        accuracy = np.around((tp + tn) / (tp + fn + fp + tn), decimals=2)
    else:
        accuracy = 0

    if precision + recall_tpr_sensitivity != 0:
        f1 = np.around(2 * precision * recall_tpr_sensitivity / (precision + recall_tpr_sensitivity), decimals=2)
    else:
        f1 = 0

    gmean = sqrt(tnr_specificity * recall_tpr_sensitivity)

    print('TOT = ' + str(tp + fn + tn + fp))
    print('P = ' + str(tp + fn) + '\tN = ' + str(tn + fp))
    print('TP = ' + str(tp) + '\tTN = ' + str(tn))
    print('FP = ' + str(fp) + '\tFN = ' + str(fn))
    print('Precision = ' + str(precision) + '\tRecall = ' + str(recall_tpr_sensitivity))
    print('Sensitivity|TPR = ' + str(recall_tpr_sensitivity) + '\tSpecificity|TNR = ' + str(tnr_specificity))
    print('G-mean = {:0.2f}'.format(gmean) + '\tF1 = ' + str(f1))
    print('Accuracy = ' + str(accuracy))


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--numFolder", help="number of folder: 05, 06, 26, or 35")
    args = a.parse_args()
    if args.numFolder is None:
        print('No numFolder')
    elif int(args.numFolder) in (5, 6, 26, 35):
        evaluation(args.numFolder)
    else:
        print('Error')
