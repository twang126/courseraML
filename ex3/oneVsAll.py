import numpy as np
from scipy.optimize import minimize

from lrCostFunction import lrcostFunctionReg
from gradientFunctionReg import lrgradientReg
from sigmoid import sigmoid

def oneVsAll(X, y, n_labels, Lambda):
    initial_theta = np.zeros((X.shape[1],1))  # 401x1
    all_theta = np.zeros((n_labels, X.shape[1])) #10x401

    for c in np.arange(1, n_labels+1):
        res = minimize(lrcostFunctionReg, initial_theta, args=(Lambda, X, (y == c)*1), method=None,
                       jac=lrgradientReg, options={'maxiter':50})
        all_theta[c-1] = res.x

    return all_theta
