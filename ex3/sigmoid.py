from numpy import e

def sigmoid(z):
    return(1 / (1 + e**(-z)))