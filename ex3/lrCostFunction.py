import numpy as np
from sigmoid import sigmoid

def lrcostFunctionReg(theta, reg, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))
    
    J = -1*(1.0/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y)) + (float(reg)/(2*m))*np.sum(np.square(theta[1:]))
    
    return J
