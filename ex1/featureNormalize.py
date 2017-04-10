import numpy as np


def featureNormalize(X):
    """
       returns a normalized version of X where
       the mean value of each feature is 0 and the standard deviation
       is 1. This is often a good preprocessing step to do when
       working with learning algorithms.
    """
    X_norm, mu, sigma = 0,0,0

    X_norm = X.copy()
    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X_norm = np.divide((X_norm - mu), sigma)

    return X_norm, mu, sigma
