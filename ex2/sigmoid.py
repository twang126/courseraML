from numpy import e

def sigmoid(z):
    """computes the sigmoid of z."""

    return 1.0/(1 +  e**(-z))