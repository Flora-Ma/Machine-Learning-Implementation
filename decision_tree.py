import numpy as np

# Here for ease of implementation, the features are all one-hot encoded 
# and target labels are 0 or 1 (binary classification).

def entropy(p_vec):
    """
    Given a possibility vector, compute the entropy as a measure of impurity.
    Args:
        p_vec (list): possibility vector whoes sum up to 1
    Returns:
        entropy (float): entropy
    """
    entropy = 0
    for p in p_vec:
        if p != 0:
            entropy += - p * np.log2(p)
    return entropy

def split_dataset(X, node_indices, feature):
    """
    Splits the data at given node into left and right branches.
    Args:
        X (ndarray(m, n)): data set, m examples with n features
        node_indices (list): List containing the active indices, the samples being considered at this step
        feature (scalar): index of feature to split on
    Returns:
        left_indices (ndarray): Indices with feature value == 1
        right_indices (ndarray): Indices with feature value == 0
    """
    left_indices = []
    right_indices = []
    for i in node_indices:
        if X[i][feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    return left_indices, right_indices

def compute_entropy(node):
    """
    Compute entropy of node.
    Args:
        node (ndarray(n,)): array containing the target labels (0 or 1).
    Returns:
        entropy (float) 
    """
    p1 = 0
    if len(node) > 0:
        p1 = sum(node) / len(node)
    return entropy([1 - p1, p1])

def compute_information_gain(X, y, node_indices, feature):
    """
    Compute the information gain of splitting the node on a given feature.
    Args:
        X (ndarray(m, n)): data set, m examples with n features
        node_indices (list): List containing the active indices, the samples being considered at this step
        feature (scalar): index of feature to split on
    Returns:
        information_gain(float): information gain
    """
    left_indices, right_indices = split_dataset(X, node_indices, feature)

    y_node = y[node_indices]
    left_node = y[left_indices]
    right_node = y[right_indices]

    node_entropy = compute_entropy(y_node)
    left_entropy = compute_entropy(left_node)
    right_entropy = compute_entropy(right_node)

    left_weight = len(left_node) / len(y_node)
    right_weight = len(right_node) / len(y_node)

    information_gain = node_entropy - left_weight * left_entropy - right_weight * right_entropy
    return information_gain

def get_best_split(X, y, node_indices):
    """
    Return the optimal feature and threshold to split the node data.
    Args:
        X (ndarray):            data set, m examples with n features
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        best_feature (int):     The index of the best feature to split
    """
    num_feature = X.shape[1]
    information_gain = []
    for i in range(num_feature):
        information_gain.append(compute_information_gain(X, y, node_indices, i))
    best_feature = np.argmax(information_gain)
    return best_feature

def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth, tree):
    """
    Build decision tree recursively.
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
        branch_name (string):   Name of the branch. ['Root', 'Left', 'Right']
        max_depth (int):        Max depth of the resulting tree. 
        current_depth (int):    Current depth. Parameter used during recursive call.
        trees (list):           Store tree node.
    """
    if current_depth == max_depth:
        formatting = ' ' * current_depth + '-' * current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return
    
    y_node = y[node_indices]
    if np.all(y_node == 0) or np.all(y_node == 1):
        formatting = ' ' * current_depth + '-'*current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return
    
    best_feature = get_best_split(X, y, node_indices)

    formatting = "-"*current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))

    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    tree.append((left_indices, right_indices, best_feature))

    build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth + 1, tree)
    build_tree_recursive(X, y, right_indices, "Right", max_depth,  current_depth + 1, tree)


