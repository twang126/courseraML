import numpy as np  
from computeCost import computeCost



def gradientDescent(X, y, theta, alpha, num_iters):
    """
     Performs gradient descent to learn theta
       theta = gradientDescent(x, y, theta, alpha, num_iters) updates theta by
       taking num_iters gradient steps with learning rate alpha
    """

    # Initialize some useful values
    J_history = []
    m = y.size  # number of training examples

    for i in range(num_iters):
        hypothesis = np.dot(X, theta)
        theta -= (alpha/m) * np.dot((hypothesis-y), X)
        J_history.append(computeCost(X, y, theta))

    return theta, J_history
