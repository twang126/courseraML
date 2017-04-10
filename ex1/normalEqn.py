import numpy as np
from numpy.linalg import inv


def normalEqn(X,y):
    """ Computes the closed-form solution to linear regression
       normalEqn(X,y) computes the closed-form solution to linear
       regression using the normal equations.
    """

# ====================== YOUR CODE HERE ======================
# Instructions: Complete the code to compute the closed form solution
#               to linear regression and put the result in theta.
#

# ---------------------- Sample Solution ----------------------


# -------------------------------------------------------------
    return np.dot((np.linalg.inv(np.dot(X.T,X))),np.dot(X.T,y))

# ============================================================

