import numpy as np
import pandas as pd
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, reg):

    """computes the cost and gradient of the neural network. The
  parameters for the neural network are "unrolled" into the vector
  nn_params and need to be converted back into the weight matrices.

  The returned parameter grad should be a "unrolled" vector of the
  partial derivatives of the neural network.
    """

# Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
# for our 2 layer neural network
# Obtain Theta1 and Theta2 back from nn_params
    theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                       (hidden_layer_size, input_layer_size + 1), order='F').copy()

    theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                       (num_labels, (hidden_layer_size + 1)), order='F').copy()

    m = X.shape[0]
    X = np.column_stack((np.ones((m, 1)), X))    
    y_matrix = pd.get_dummies(y.ravel()).as_matrix()

    a1 = X # 5000x401
        
    z2 = theta1.dot(a1.T) # 25x401 * 401x5000 = 25x5000 
    a2 = np.c_[np.ones((X.shape[0],1)),sigmoid(z2.T)] # 5000x26 
    
    z3 = theta2.dot(a2.T) # 10x26 * 26x5000 = 10x5000 
    a3 = sigmoid(z3) # 10x5000
    
    J = -1*(1.0/m)*np.sum((np.log(a3.T)*(y_matrix)+np.log(1-a3).T*(1-y_matrix))) + (float(reg)/(2*m))*(np.sum(np.square(theta1[:,1:])) + np.sum(np.square(theta2[:,1:])))

    # Gradients
    d3 = a3.T - y_matrix # 5000x10
    d2 = theta2[:,1:].T.dot(d3.T)*sigmoidGradient(z2) # 25x10 *10x5000 * 25x5000 = 25x5000
    
    delta1 = d2.dot(a1) # 25x5000 * 5000x401 = 25x401
    delta2 = d3.T.dot(a2) # 10x5000 *5000x26 = 10x26
    
    theta1_ = np.c_[np.zeros((theta1.shape[0],1)),theta1[:,1:]]
    theta2_ = np.c_[np.zeros((theta2.shape[0],1)),theta2[:,1:]]
    
    theta1_grad = (delta1 + (theta1_*float(reg)))/float(m)
    theta2_grad = (delta2 + (theta2_*float(reg)))/float(m)

    # Unroll gradient
    grad = np.hstack((theta1_grad.T.ravel(), theta2_grad.T.ravel()))


    return J, grad