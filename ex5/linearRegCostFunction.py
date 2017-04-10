import numpy as np
def linearRegCostFunction(X, y, theta, Lambda):
    """computes the
    cost of using theta as the parameter for linear regression to fit the
    data points in X and y. Returns the cost in J and the gradient in grad
    """
# Initialize some useful values

    m = y.size # number of training examples

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost and gradient of regularized linear 
#               regression for a particular choice of theta.
#
#               You should set J to the cost and grad to the gradient.
#


# =========================================================================
    h = X.dot(theta)
    J = (1.0/(2*m))*np.sum(np.square(h-y)) + (float(Lambda)/(2*m))*np.sum(np.square(theta[1:]))
    grad = (1.0/m) * (X.T.dot(h-y)) + (float(Lambda) / m) * np.r_[[[0]],theta[1:].reshape(-1,1)]

    return J, grad.flatten()