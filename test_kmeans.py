import numpy as np
import kmeans

# test self implemented k-means
X = np.array([[-1, -1], [-1.5, -1.5], [-1.5, 1],
                  [-1, 1.5], [2.5, 1.5], [-1.1, -1.7], [-1.6, 1.2]])
init_centroids = kmeans.kMeans_init_centroids(X, 3)
centroids, idx = kmeans.run_kMeans(X, init_centroids, max_iters=10)
print(f'idx={idx}, centroids={centroids}')

# sklean.cluster.KMeans
import sklearn.cluster
result = sklearn.cluster.KMeans(n_clusters=3, random_state=0, n_init='auto').fit(X)
print(f'idx={result.labels_}, centroids={result.cluster_centers_}')
