import numpy as np
import math

def sigmoid(z):
    """
    Compute the sigmod of z
    Args:
        z (ndarray): A scalar, numpy array of any size.
    Returns:
        g (ndarray): sigmoid(z), with the same shape as z.
    """
    return 1 / (1 + np.exp(-z))

def compute_cost_logistic(X, y, w, b):
    """
    Compute cost function of logistic regression.
    Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m,)): target values
        w (ndarray (n,)): model parameters
        b (scalar): model parameters

    Returns:
        cost (scalar): cost
    """
    m = X.shape[0]
    cost = 0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += - y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)
    cost /= m
    return cost

def compute_cost_logistic_2(X, y, w, b):
    """
    Compute cost function of logistic regression. More vectorized.
    Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m,)): target values
        w (ndarray (n,)): model parameters
        b (scalar): model parameters

    Returns:
        cost (scalar): cost
    """
    m = X.shape[0]
    f_wb = sigmoid(np.dot(X, w) + b)
    cost = - np.multiply(y, np.log(f_wb)) - np.multiply(1 - y, np.log(1 - f_wb))
    cost = sum(cost) / m
    return cost

def compute_cost_logistic_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost for regularized logistic regression
    Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m,)): target values
        w (ndarray (n,)): model parameters
        b (scalar): model parameters
        lambda_ (scalar): Controls amount of regularization

    Returns:
        total_cost (scalar): cost
    """
    cost = compute_cost_logistic(X, y, w, b)
    m = X.shape[0]
    reg_cost = np.sum(w ** 2) * lambda_ / (2 * m)
    return cost + reg_cost

def compute_gradient_logistic(X, y, w, b):
    """
    Compute the gradient for logistic regression.
    Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m,)): target values
        w (ndarray (n,)): model parameters
        b (scalar): model parameters

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
    dj_db /= m
    dj_dw /= m
    return dj_dw, dj_db

def compute_gradient_logistic_2(X, y, w, b):
    """
    Compute the gradient for logistic regression. More vectorized.
    Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m,)): target values
        w (ndarray (n,)): model parameters
        b (scalar): model parameters

    Returns:
        dj_dw (ndarray (n,)): The gradient of the cost w.r.t the parameters w.
        dj_db (scalar): The gradient of the cost w.r.t the parameter b.
    """
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0
    f_wb = sigmoid(np.matmul(X, w) + b)
    err = f_wb - y
    dj_dw = np.matmul(np.transpose(X), err) / m
    dj_db = sum(err) / m
    return dj_dw, dj_db

def compute_gradient_logistic_reg(X, y, w, b, lambda_):
    """
    Compute the gradient for regularized logistic regression.
    Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m,)): target values
        w (ndarray (n,)): model parameters
        b (scalar): model parameters
        lambda_ (scalar): Controls amount of regularization

    Returns:
        dj_dw (ndarray (n,)): The gradient of the cost w.r.t the parameters w.
        dj_db (scalar): The gradient of the cost w.r.t the parameter b.
    """
    dj_dw, dj_db = compute_gradient_logistic(X, y, w, b)
    m, n = X.shape
    for i in range(n):
        dj_dw[i] += (lambda_ / m) * w[i]
    return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    """
    Performs batch gradient descent
    Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m,)): target values
        w_in (ndarray (n,)): Initial values of model parameters
        b_in (scalar): Initial values of model parameter
        alpha (float): Learning rate
        num_iters (scalar): number of iterations to run gradient descent

    Returns:
        w (ndarray (n, )): Updated values of parameters
        b (scalar): Updated value of paramter        
    """
    cost_history = []
    w, b = w_in, b_in
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_logistic(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i < 100000: # prevent resource exhaustion
            cost_history.append(compute_cost_logistic(X, y, w, b))
        if i % math.ceil(num_iters / 10) == 0:
            print(f'Iteration {i:4d}: Cost {cost_history[-1]}')
    return w, b, cost_history

def predict(X, w, b):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters w and b.
    Args:
        X (ndarray Shape (m,n): data, m examples by n features
        w (ndarray Shape (n,)): values of parameters of the model      
        b (scalar): value of bias parameter of the model
    Returns:
        p (ndarray (m,)): predictions for X using a threshold at 0.5; if f(x) >= 0.5, predict 1 else 0.
    """
    z_wb = np.dot(X, w) + b
    f_wb = sigmoid(z_wb)
    p = [1 if x >= 0.5 else 0 for x in f_wb]
    return p
        

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])
w_in = np.zeros_like(X_train[0])
b_in = 0
alpha = 0.1
iters = 10000
w_out, b_out, history = gradient_descent(X_train, y_train, w_in, b_in, alpha, iters)
print(f'w = {w_out}, b = {b_out}, cost = {history[-1]}')
p = predict(X_train, w_out, b_out)
print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))