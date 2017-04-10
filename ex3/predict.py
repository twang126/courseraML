import numpy as np

from sigmoid import sigmoid

def predict(Theta1, Theta2, X):
    """ outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2)
    """
    m = X.shape[0]

    X = np.c_[np.ones((m,1)), X]
    z2 = Theta1.dot(X.T)
    a2 = np.c_[np.ones((m,1)), sigmoid(z2).T]
    
    z3 = a2.dot(Theta2.T)
    a3 = sigmoid(z3)
        
    return(np.argmax(a3, axis=1)+1) 

