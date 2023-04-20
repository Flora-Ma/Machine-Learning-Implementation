import numpy as np

def sigmoid(z):
    """
    Compute the sigmod of z
    Args:
        z (ndarray): A scalar, numpy array of any size.
    Returns:
        g (ndarray): sigmoid(z), with the same shape as z.
    """
    return 1 / (1 + np.exp(-z))

def softmax(z):
    """
    Compute the softmax of z
    Args:
        z (ndarray): numpy array of any size.
    Returns:
        a (ndarray): a vector of probability
    """
    ez = np.exp(z)
    return ez / np.sum(ez)

def entropy(p):
    if p == 0 or p == 1:
        return 0
    return - p * np.log2(p) - (1 - p) * np.log2(1 - p)

def split_indices(X, index_feature):
    """
    Given a dataset and a index_feature, return two lists for the two split nodes, the left node has items that have
    that feature = 1 and the righ node has items that have that feature = 0.
    Args:
        X (ndarray(m, n)): data set, m examples with n features
        index_feature (scalar): index of feature
    Returns:
        left_indices (ndarray): left node indices
        right_indices (ndarray): right node indices
    """
    left_indices = []
    right_indices = []
    for i, x in enumerate(X):
        if x[index_feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    return left_indices, right_indices

def information_gain(X, y, left_indices, right_indices):
    p_node = sum(y) / len(y)
    h_node = entropy(p_node)

    w_left = len(left_indices) / len(X)
    w_right = len(right_indices) / len(X)
    entropy_left = entropy(sum(y[left_indices]) / len(left_indices)) if len(left_indices) > 0 else 0
    entropy_right = entropy(sum(y[right_indices]) / len(right_indices)) if len(right_indices) > 0 else 0
    weighted_entropy = w_left * entropy_left + w_right * entropy_right

    return h_node - weighted_entropy





