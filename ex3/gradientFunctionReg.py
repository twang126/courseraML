import numpy as np
from sigmoid import sigmoid


def lrgradientReg(theta, reg, X,y):
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1,1)))
      
    grad = (1.0/m)*X.T.dot(h-y) + (float(reg)/m)*np.r_[[[0]],theta[1:].reshape(-1,1)]
        
    return grad.flatten()