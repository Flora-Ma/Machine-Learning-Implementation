import numpy as np

def zscore_normalize_scaler(X):
    """
    Return z-score normalization scaler paramerters for X by column and normalized X. This normalization is also called standardization or mean removal and variance.
    Args:
        X (ndarray (m, n)): data
    Returns:
        X_norm (ndarray (m, n)): normalized data by column
        mean_ (ndarray(m,)): mean by column
        std_ (ndarray(m,)): standard deviation
    """
    mean_ = np.mean(X, axis = 0)
    std_ = np.std(X, axis = 0)
    X_norm = (X - mean_) / std_ # TODO: what if std_[i] == 0
    return X_norm, mean_, std_

def zscore_scaler_transform(X, mean_, std_):
    """
    Use the given mean_ and std_ to do z_score normalization to X by column. 
    Args:
        X (ndarray (m, n)): data
        mean_ (ndarray(m,)): normalization parameter
        std_ (ndarray(m,)): normalization parameter
    Returns:
        X_norm (ndarray (m, n)): normalized result
    """
    X_norm = (X - mean_) / std_
    return X_norm

def min_max_scaler(X):
    """
    Return min-max normalization scaler parameters for X by column and normalized X.
    Args:
        X (ndarray (m, n)): data
    Returns:
        X_norm (ndarray (m, n)): normalized data by column
        min_ (ndarray(m,)): min values by column
        max_ (ndarray(m,)): max values deviation
    """
    min_ = np.min(X, axis = 0)
    max_ = np.max(X, axis = 0)
    X_norm = (X - min_) / (max_ - min_) # TODO: what if max_[i] - min_[i] == 0
    return X_norm, min_, max_

def min_max_scaler_transform(X, min_, max_):
    """
    Use the given parameters to do min_max normalization to X by column. 
    Args:
        X (ndarray (m, n)): data
        min_ (ndarray(m,)): normalization parameter
        max_ (ndarray(m,)): normalization parameter
    Returns:
        X_norm (ndarray (m, n)): normalized result
    """
    X_norm = (X - min_) / (max_ - min_)
    return X_norm

