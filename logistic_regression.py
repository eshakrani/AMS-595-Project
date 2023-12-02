import numpy as np 
import scipy
from tqdm import tqdm

def sigmoid(z):
    """Returns a probability value [0, 1] for an input z

    Passes z as an input into the sigmoid function,
        sigmoid(z) = 1 / (1 + e^-z)
    and returns the output value which ranges from 0 to 1
    """
    return 1 / (1 + np.exp(-z))



def classification(image, w_path, b_path):
    """Assigns a label to input image based on a previously trained model

    w_path and b_path are the locations of a weight term and a bias term 
    that were both previously tuned during the training process of a model.
    These terms are used as extra parameters in the sigmoid function. 
    This essentially uses the training data as a basis for figuring out the 
    probability that the input image belongs to a specific class.

    The assigned label and its probability are returned from the function
    """

    # obtain w and b from their respective files
    w = scipy.io.loadmat(w_path)['weights']
    b = scipy.io.loadmat(b_path)['biases']

    # reshape photo data
    X = image.reshape((1, np.prod(image.shape[0:3])))

    # flatten the data
    X = X / 255
    print('X shape: ', X.shape)
    print('w shape: ', w.shape)

    # calculate probabilities for testing data 
    # A = 1 / (1 + np.exp(-(np.dot(X, w) + b)))
    A = sigmoid(np.dot(X, w) + b)
    print("PROBABILITY: ", A)
    # assign labels to the samples using their probabilities
    Y = (A >= 0.5) * 1.0

    return {'label': Y, 'prob': A}



def propagate(w, b, X, Y):
    """Implement the cost function and its gradient

    Arguments:
    w: weights - numpy array of size (num_px * num_px * 3, 1)
    b: bias - scalar
    X: data of size (num. samples, num_px * num_px * 3)
    Y: true "label" vector

    Return:
    dw: gradient of loss function w.r.t. w - same shape as w
    db: gradient of loss function w.r.t. b - same shape as b
    """

    m = X.shape[0]

    # Forward propagation (X -> cost)
    A = sigmoid(np.dot(X, w) + b)

    # Backward propagation (to find gradient)
    dw = (np.dot(X.T, (A-Y))) / m 
    db = (np.sum(A-Y)) / m 

    return dw, db 


def train(X_train, y_train, w_path, b_path):
    """Trains a logistic regression classifier on the given training data

    Performs gradient descent to optimize the the weight and bias parameters
    in the sigmoid function using the training data as a basis.

    The optimized weight and bias terms are saved to the current directory 
    in separate files as well as returned from the function in a dictionary
    """

    # num. dims (cols) for each sample
    dim = X_train.shape[1]

    # initialize weights vector
    w = np.zeros((dim, 1))

    # initialize bias term
    b = 0

    # hyperparameters
    num_iterations = 10000
    learning_rate = 0.006

    print('Training model....')

    for i in tqdm(range(num_iterations)):
        dw, db = propagate(w, b, X_train, y_train)

        # gradient descent
        w = w - (learning_rate * dw)
        b = b - (learning_rate * db)

    # save the final weight and bias terms to their respective files
    scipy.io.savemat(w_path, {'weights': w})
    scipy.io.savemat(b_path, {'biases': b})

    # return the results from the function
    return {'w': w, 'b': b}