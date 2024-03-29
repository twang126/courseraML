import numpy as np


def pca(X):
    """computes eigenvectors of the covariance matrix of X
      Returns the eigenvectors U, the eigenvalues (on diagonal) in S
    """

    # You need to return the following variables correctly.

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should first compute the covariance matrix. Then, you
    #               should use the "svd" function to compute the eigenvectors
    #               and eigenvalues of the covariance matrix.
    #
    # Note: When computing the covariance matrix, remember to divide by m (the
    #       number of examples).
    #


# =========================================================================

    covar = np.dot(X.T,X) / len(X)
    U, S, V = np.linalg.svd(covar)

    return U, S, V