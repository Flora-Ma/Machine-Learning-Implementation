import numpy as np
def kMeans_init_centroids(X, K):
    """
    This function initializes K centroids to be used in K-Means on the dataset X.
    Args:
        X (ndarray): Data points
        K (int): number of centroids/clusters
    Returns:
        centroids (ndarray): Initialized centroids
    """
    randidx = np.random.permutation(X)
    centroids = X[randidx[:K]]
    return centroids

def run_kMeans(X, initial_centroids, max_iters):
    pass

