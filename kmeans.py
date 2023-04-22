import numpy as np
def kMeans_init_centroids(X, K):
    """
    This function initializes K centroids to be used in K-Means on the dataset X.
    Args:
        X (ndarray): Data points
        K (int): number of centroids/clusters
    Returns:
        centroids (ndarray(K, n)): Initialized centroids
    """
    randidx = np.random.permutation(X)
    centroids = randidx[:K]
    return centroids

def find_closest_centroids(X, centroids):
    """
    Computes the closest centroids for each example.
    Args:
        X (ndarray(m, n)): Data points
        centroids (ndarray(k, n)): centroids
    Returns:
        idx (ndarray(m, )): closest centroids
        cost (ndarray(m, )): squared L2 norm distance
    """
    m = X.shape[0]
    K = centroids.shape[0]
    idx = np.zeros(m, dtype=int)
    cost = np.zeros(m)
    for i in range(m):
        distances = []
        for k in range(K):
            diff = X[i] - centroids[k]
            distances.append(np.dot(diff, diff))
        idx[i] = np.argmin(distances)
        cost[i] = distances[idx[i]]
    return idx, cost

def compute_centroids(X, idx, K):
    """
    Return the centroids by computing the means of the data
    points assigned to each cluster.
    Args:
        X (ndarray(m, n)): Data points
        idx (ndarray(m, )): cluster id (closest centroid id) for each
                            example in X.
        K (int): cluster numbers
    Returns:
        centroids (ndarray(K, n)): New centroids of each cluster
    """
    n = X.shape[1]
    centroids = np.zeros((K, n))
    for k in range(K):
        cluster = X[np.where(idx == k)]
        if len(cluster) > 0:
            centroids[k] = np.mean(cluster, axis=0)
    return centroids    

def run_kMeans(X, initial_centroids, max_iters=10):
    """
    Run K-Means algorithm on data matrix X to get k clusters, 
    where each row of X is a single example.
    Args:
        X (ndarray(m, n)): Data points
        initial_centroids (ndarray(k, n)): initial centroids
        max_iters (int): max interations
    Returns:
        centroids (ndarray(k, n)): final centroids of clusters
        idx (ndarray(m,)): cluster id [0, k) for each data point.
    """
    centroids = initial_centroids
    m = X.shape[0]
    k = len(centroids)

    for iter in range(max_iters):        
        idx, cost = find_closest_centroids(X, centroids)
        print(f'K-Means iteration {iter}/{max_iters - 1}, cost = {np.sum(cost)}')
        centroids = compute_centroids(X, idx, k)
    return centroids, idx

                


