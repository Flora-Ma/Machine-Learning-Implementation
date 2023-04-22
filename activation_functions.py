import numpy as np

def sigmoid(z):
    """
    Compute the sigmod of z.
    Args:
        z (ndarray): A scalar, numpy array of any size.
    Returns:
        g (ndarray): sigmoid(z), with the same shape as z.
    """
    return 1 / (1 + np.exp(-z))

def relu(z):
    """
    Compute Relu(Rectified Linear Unit) function of z.
    Args:
        z (ndarray): A scalar, numpy array of any size.
    Returns:
        g (scalar): relu(z), with the same shape as z.
    """
    g = np.maximum(z, 0)
    return g

def softmax(z):
    """
    Compute the softmax of z.
    Args:
        z (ndarray): numpy array of any size.
    Returns:
        a (ndarray): a vector of probability
    """
    ez = np.exp(z)
    return ez / np.sum(ez)






