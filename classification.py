import numpy as np
import scipy

# return a probability value [0, 1] for a given input z
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def classification(X):
    #reshape photo data
    X = X.reshape((X.shape[0], np.prod(X.shape[1:4]))) #need to check dimensions
    X = X / 255

    #not sure what we're doing here yet, assuming w and b will be exlicit 
    w = scipy.io.loadmat('Py_Project_2_Data/ibc1_weights.mat')['weights']
    b = scipy.io.loadmat('Py_Project_2_Data/ibc1_biases.mat')['biases'] 

    # calculate probabilities for testing data 
    A = sigmoid(np.dot(X, w) + b)

    # assign labels to the samples using their probabilities
    Y = (A >= 0.5) * 1.0
    return Y



    

