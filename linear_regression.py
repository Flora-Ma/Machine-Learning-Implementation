import numpy as np
import copy
import math

def compute_cost(X, y, w, b, regularized=False, lambda_=1):
    """
    Mean sqaured error cost function.
    Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m,)): target values
        w (ndarray (n,)): model parameters
        b (scalar): model parameters
        regularized (boolean): with regularization or not
        lambda_ (scalar): Controls amount of regularization
    Returns:
        cost (scalar): cost
    """
    m = X.shape[0]
    cost = np.sum((np.matmul(X, w) + b - y) ** 2)
    if regularized:
        cost += lambda_ * np.sum(w ** 2)
    cost /= 2 * m
    return cost

def compute_gradient(X, y, w, b, regularized=False, lambda_=1):
    """
    Compute gradient for linear regression.
    Args:
        X (ndarray (m, n)): Data, m examples with n features.
        y (ndarray (m,)): target values
        w (ndarray (n,)): model parameters
        b (scalar): model parameters
        regularized (boolean): with regularization or not
        lambda_ (scalar): Controls amount of regularization
    Returns:
        dj_dw (ndarray (n,)): The gradient of the cost w.r.t the parameter w
        dj_db (scalar): The gradient of the cost w.r.t the parameter b
    """
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0
    for i in range(m):
        err = np.dot(X[i], w) + b - y[i]
        dj_dw += err * X[i]
        if regularized:
            dj_dw += lambda_ * w
        dj_db += err
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, regularized=False, lambda_=1):
    """
    Perform batch gradient descent to learn w and b.
    Args:
        X (ndarray (m, n)): Data, m examples with n features.
        y (ndarray (m,)): target values
        w_in (ndarray (n,)): initial model parameters
        b_in (scalar): initial model parameter
        cost_function: function to compute cost
        gradient_function: function to compute gradient
        alpha (float): learning rate
        num_iters (int): number of iterations to run gradient descent
        regularized (boolean): with regularization or not
        lambda_ (scalar): Controls amount of regularization
    Returns:
        w (ndarray (n,)): updated values of parameters
        b (scalar): updated values of parameters
    """
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b, regularized, lambda_)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i % math.ceil(num_iters / 10) == 0:
            cost = compute_cost(X, y, w, b, regularized, lambda_)
            print(f'Iteration {i:4d}: Cost {cost:8.2f}')
    return w, b

def fit(X, y, num_iters, alpha=0.1, regularized=False, lambda_=1):
    """
    Fit train data X, y to learn parameters w, b of linear regression model.
    Args:
        X (ndarray (m, n)): Data, m examples with n features.
        y (ndarray (m,)): target values
        num_iters (int): number of iterations to run gradient descent
        alpha (float): learning rate
        regularized (boolean): with regularization or not
        lambda_ (scalar): Controls amount of regularization
    Returns:
        w (ndarray (n,)): updated values of parameters
        b (scalar): updated values of parameters
    """
    n = X.shape[1]
    initial_w = np.zeros(n)
    initial_b = 0
    w, b = gradient_descent(X, y, initial_w, initial_b, compute_cost, compute_gradient, alpha, num_iters, regularized, lambda_)
    return w, b

def predict(x, w, b):
    """
    Single prefict using linear regression.
    Args:
        x (ndarray (n,)): single example with n features
        w (ndarray (n,)): model parameters
        b (scalar): model parameter
    Returns:
        y_hat (scalar): prediction
    """
    y_hat = np.dot(w, x) + b
    return y_hat
