import numpy as np
from sigmoid import sigmoid


def gradientFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1,1)))
      
    grad = (1.0/m)*X.T.dot(h-y) + (float(Lambda)/m)*np.r_[[[0]],theta[1:].reshape(-1,1)]
        
    return grad.flatten()