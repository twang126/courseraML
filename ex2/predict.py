from sigmoid import sigmoid
import numpy as np


def predict(theta, X):

    """ computes the predictions for X using a threshold at 0.5
    (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
    """

    p = sigmoid(X.dot(theta.T)) >= .5
    return(p.astype('int'))
    
