import numpy as np

def dense(a_in, W, b, activation_fun):
    """
    Compute dense layer (one kind of layers whoes each neuron output is a function 
    of all the activation outputs of the previous layer).
    Args:
        a_in (ndarray(1, n)): Data, 1 example
        W (ndarray(n, j)): Weight matrix, n features, j units
        b (ndarray(1, j)): bias vector, j units
    Returns:
        a_out (ndarray(1, j)): j units
    """
    z = np.matmul(a_in, W) + b
    return activation_fun(z)


def sequential(x, W1, b1, W2, b2, afun1, afun2):
    """
    Build a two layer neural network
    """
    a1 = dense(x, W1, b1, afun1)
    a2 = dense(a1, W2, b2, afun2)
    return a2

def predict(X, W1, b1, W2, b2, afun1, afun2):
    m = X.shape[0]
    p = np.zeros((m, 1))
    for i in range(m):
        p[i, 0] = sequential(X[i], W1, b1, W2, b2, afun1, afun2)
    return p

