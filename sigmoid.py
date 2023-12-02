import numpy as np 

def sigmoid(z):
    """Returns a probability value [0, 1] for an input z

    Passes z as an input into the sigmoid function,
        sigmoid(z) = 1 / (1 + e^-z)
    and returns the output value which ranges from 0 to 1
    """
    return 1 / (1 + np.exp(-z))