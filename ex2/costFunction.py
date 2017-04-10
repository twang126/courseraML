import numpy as np
from sigmoid import sigmoid

def costFunction(theta, X,y):
    """ computes the cost of using theta as the
    parameter for logistic regression and the
    gradient of the cost w.r.t. to the parameters."""
    m = y.size
    h = sigmoid(X.dot(theta))

    J =  (1.0/m) * ((-y.T).dot(np.log(h)) - (1-y.T).dot(np.log(1-h)))

    return J
