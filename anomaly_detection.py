import numpy as np

def estimate_gaussian(X):
    """
    Estimate mean and variance of all features in the data set.
    Args:
        X (ndarray(m, n)): Data, m examples with n features.
    Returns:
        mu (ndarray(n,)): Mean of all the features
        var (ndarray(n,)): Variance of all the features
    """
    mu = np.mean(X, axis=0)
    var = np.mean((X - mu) ** 2, axis=0)
    return mu, var

def multivariate_gaussian(X, mu, var):
    """
    Computes the probability density function of the examples X under
    the multivariate gaussian distribution with parameter mu and var.
    If var is a matrix, it is treated as the covariance matrix. If var
    is a vector, it is treated as the var values of the variances in 
    each dimension.
    Args:
        X (ndarray(m, n)): Data, m examples with n features
        mu (ndarray(n,)): Mean
        var (ndarray(n,)): Variance 
    Returns:
        p (ndarray(m,)): probability of m examples
    """
    k = len(mu)
    
    if var.ndim == 1:
        var = np.diag(var)
        
    X = X - mu
    p = (2* np.pi)**(-k/2) * np.linalg.det(var)**(-0.5) * \
        np.exp(-0.5 * np.sum(np.matmul(X, np.linalg.pinv(var)) * X, axis=1))
    
    return p

def select_threshold(y_val, p_val):
    """
    Finds the best threshold to use for selecting outliers
    based on the results from a validation set (p_val) and 
    the ground truth (y_val).
    Args:
        y_val (ndarray): Ground truth on validation set
        p_val (ndarray): Results on validation set
    Returns:
        epsilon (float): Threshold chosen
        F1 (float): F1 score by choosing epsilon as threshold
    """
    best_epsilon, best_F1 = 0, 0
    F1 = 0
    step_size = (max(p_val) - min(p_val)) / 1000
    for epsilon in np.arange(min(p_val), max(p_val), step_size):
        predictions = p_val < epsilon
        tp = np.sum((predictions == 1) & (y_val == 1))
        fp = np.sum((predictions == 1) & (y_val == 0))
        fn = np.sum((predictions == 0) & (y_val == 1))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        F1 = 2 * precision * recall / (precision + recall)

        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon

    return best_epsilon, best_F1

