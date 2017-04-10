from computeCostMulti import computeCostMulti


def gradientDescentMulti(X, y, theta, alpha, num_iters):
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
        theta = theta - (alpha/m) * np.dot(X.T, (hypothesis - y))

        J_history.append(computeCostMulti(X, y, theta))

    return theta, J_history