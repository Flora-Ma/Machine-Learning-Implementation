import numpy as np
import math
from functions import sigmoid
import copy

def compute_cost_logistic(X, y, w, b, regularized=False, lambda_=1):
    """
    Compute cost function of logistic regression. 
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
    f_wb = sigmoid(np.dot(X, w) + b)
    cost = - np.multiply(y, np.log(f_wb)) - np.multiply(1 - y, np.log(1 - f_wb))
    cost = sum(cost) / m
    if regularized:
        cost += lambda_ * np.sum(w ** 2) / (2 * m)
    return cost

def compute_gradient_logistic(X, y, w, b, regularized=False, lambda_=1):
    """
    Compute the gradient for logistic regression.
    Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m,)): target values
        w (ndarray (n,)): model parameters
        b (scalar): model parameters
        regularized (boolean): with regularization or not
        lambda_ (scalar): Controls amount of regularization
    Returns:
        dj_dw (ndarray (n,)): The gradient of the cost w.r.t the parameters w.
        dj_db (scalar): The gradient of the cost w.r.t the parameter b.
    """
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0
    f_wb = sigmoid(np.dot(X, w) + b)
    for i in range(m):
        dj_db += f_wb[i] - y[i]
        for j in range(n):
            dj_dw[j] += (f_wb[i] - y[i]) * X[i, j]
    for j in range(n):
        dj_dw[j] += lambda_ * w[j]
    dj_db /= m
    dj_dw /= m
    return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, alpha, num_iters, regularized=False, lambda_=1):
    """
    Performs batch gradient descent
    Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m,)): target values
        w_in (ndarray (n,)): Initial values of model parameters
        b_in (scalar): Initial values of model parameter
        alpha (float): Learning rate
        num_iters (scalar): number of iterations to run gradient descent
        regularized (boolean): with regularization or not
        lambda_ (scalar): Controls amount of regularization
    Returns:
        w (ndarray (n, )): Updated values of parameters
        b (scalar): Updated value of paramter        
    """
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_logistic(X, y, w, b, regularized, lambda_)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i % math.ceil(num_iters / 10) == 0:
            print(f'Iteration {i:4d}: Cost {compute_cost_logistic(X, y, w, b, regularized, lambda_)}')
    return w, b

def fit(X, y, num_iters, alpha=0.1, regularized=False, lambda_=1):
    """
    Fit train data X, y to learn parameters w, b of logistic regression model.
    Args:
        X (ndarray (m, n)): Data, m examples with n features.
        y (ndarray (m,)): target values, 0 or 1
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
    w, b = gradient_descent(X, y, initial_w, initial_b, alpha, num_iters, regularized, lambda_)
    return w, b

def predict(x, w, b, threshold):
    """
    Single Predict whether the label is 0 or 1 using learned logistic regression parameters w and b.
    Args:
        X (ndarray (n, ): data, m examples by n features
        w (ndarray (n,)): values of parameters of the model      
        b (scalar): value of bias parameter of the model
        threshold (scalar): 
    Returns:
        p (float): possibility of label is 1
        y_hat (int): predicted label
    """
    z_wb = np.dot(x, w) + b
    p = sigmoid(z_wb)
    y_hat = 0
    if p >= threshold:
        y_hat = 1 
    return p, y_hat        
