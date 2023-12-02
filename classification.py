import numpy as np
import scipy

w = scipy.io.loadmat('training_weights.mat')['weights'] 
b = scipy.io.loadmat('training_biases.mat')['biases'] 

def classification(image):
    #reshape photo data
    X = image.reshape((image.shape[0], np.prod(image.shape[1:4])))
    X = X / 255

    # calculate probabilities for testing data 
    A = 1 / (1 + np.exp(-(np.dot(X, w) + b)))

    # assign labels to the samples using their probabilities
    Y = (A >= 0.5) * 1.0
    return Y



    

