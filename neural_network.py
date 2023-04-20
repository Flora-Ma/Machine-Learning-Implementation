import numpy as np
from cost_function import sigmoid

def dense(a_in, W, b, activation_fun):
    """
    Compute dense layer.
    Args:
        a_in (ndarray(1, n)): Data, 1 example
        W (ndarray(n, j)): Weight matrix, n features, j units
        b (ndarray(1, j)): bias vector, j units
    Returns:
        a_out (ndarray(1, j)): j units
    """
    z = np.matmul(a_in, W) + b
    return activation_fun(z)


def sequential(x, W1, b1, W2, b2, afun1=sigmoid, afun2=sigmoid):
    """
    Build a two layer neural network
    """
    a1 = dense(x, W1, b1, afun1)
    a2 = dense(a1, W2, b2, afun2)
    return a2

def predict(X, W1, b1, W2, b2, afun1=sigmoid, afun2=sigmoid):
    m = X.shape[0]
    p = np.zeros((m, 1))
    for i in range(m):
        p[i, 0] = sequential(X[i], W1, b1, W2, b2, afun1, afun2)
    return p

X = np.array([[200, 17]])
W = np.array([[1, -3, 5],[-2, 4, 6]])
B = np.array([[-1, 1, 2]])
out = dense(X, W, B)

