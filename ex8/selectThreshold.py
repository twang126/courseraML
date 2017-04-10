import numpy as np
import math

def selectThreshold(yval, pval):
    """
    finds the best
    threshold to use for selecting outliers based on the results from a
    validation set (pval) and the ground truth (yval).
    """

    bestEpsilon = 0
    bestF1 = 0

    stepsize = (np.max(pval) - np.min(pval)) / 1000.0
    for epsilon in np.arange(np.min(pval),np.max(pval), stepsize):

        predictions = (pval < epsilon).reshape((-1,1))

        X = np.column_stack((predictions, yval))

        fp = np.sum((X[:,0] == 1) & (X[:,1] == 0))
        tp = np.sum((X[:,0] == 1) & (X[:,1] == 1))
        fn = np.sum((X[:,0] == 0) & (X[:,1] == 1))
        if (tp + fp == 0):
            F1 = 0
        else:
            prec = float(tp) / (tp + fp)
            rec = float(tp) / (tp + fn)

            if(prec + rec == 0):
                F1 = 0
            else:
                F1 = (2 * prec * rec) / (float(prec + rec))

        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon

    return bestEpsilon, bestF1






