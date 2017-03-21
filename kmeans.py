import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import dask.array as da
from dask.dot import dot_graph
from sklearn import metrics
import multiprocessing

def euclidean(XA, XB):
    
    """Returns the distance between points using
       Euclidean distance (2-norm) as the distance metric between the
       points.

       Find the Euclidean distances between four 2-D coordinates:
        >>> coords = [(35.0456, -85.2672),
        ...           (35.1174, -89.9711),
        ...           (35.9728, -83.9422),
        ...           (36.1667, -86.7833)]
        >>> euclidean(coords, coords)
        array([[ 0.    ,  4.7044,  1.6172,  1.8856],
            [ 4.7044,  0.    ,  6.0893,  3.3561],
            [ 1.6172,  6.0893,  0.    ,  2.8477],
            [ 1.8856,  3.3561,  2.8477,  0.    ]])

    """
    mA = (XA.shape)[0]
    mB = (XB.shape)[0]

    distances = []

    for i in xrange(0, mA):
        dm = np.zeros(shape = (1, mB), dtype=np.double)
        for j in xrange(0, mB):
            XA_XB = XA[i, :] - XB[j, :]
            dm[0, j] = da.sqrt(da.dot(XA_XB, XA_XB))

        distances.append(da.from_array(dm, chunks = (mA + mB)/multiprocessing.cpu_count())) 

    return da.concatenate(distances, axis= 0)

def cluster_centroids(data, clusters, k=None):
    """Return centroids of clusters & clusters in data.

    data is an array of observations with shape (A, B, ...).

    clusters is an array of integers of shape (A,) giving the index
    (from 0 to k-1) of the cluster to which each observation belongs.
    The clusters must all be non-empty.

    k is the number of clusters. If omitted, it is deduced from the
    values in the clusters array.

    The result is an array of shape (k, B, ...) containing the
    centroid of each cluster.

    >>> data = np.array([[12, 10, 87],
    ...                  [ 2, 12, 33],
    ...                  [68, 31, 32],
    ...                  [88, 13, 66],
    ...                  [79, 40, 89],
    ...                  [ 1, 77, 12]])
    >>> cluster_centroids(data, np.array([1, 1, 2, 2, 0, 1]))
    array([[ 79.,  40.,  89.],
           [  5.,  33.,  44.],
           [ 78.,  22.,  49.]])

    """
    if k is None:
        k = (da.max(clusters)).compute() + 1

    result = []

    result = [da.mean(data[clusters.compute() == i], axis=0) for i in xrange(k)]
    
    return da.reshape(da.concatenate(result, axis=0), shape=(k,) + data.shape[1:])

def kmeans(data, k=None, centroids=None, steps=100):
    """Divide the observations in data into clusters using the k-means
    algorithm, and return an array of integers assigning each data
    point to one of the clusters.

    centroids, if supplied, must be an array giving the initial
    position of the centroids of each cluster.

    If centroids is omitted, the number k gives the number of clusters
    and the initial positions of the centroids are selected randomly
    from the data.

    The k-means algorithm adjusts the centroids iteratively for the
    given number of steps, or until no further progress can be made.

    >>> data = np.array([[12, 10, 87],
    ...                  [ 2, 12, 33],
    ...                  [68, 31, 32],
    ...                  [88, 13, 66],
    ...                  [79, 40, 89],
    ...                  [ 1, 77, 12]])
    >>> np.random.seed(73)
    >>> kmeans(data, k=3)
    array([1, 1, 2, 2, 0, 1])

    """
    
    if centroids is not None and k is not None:
        assert(k == len(centroids))
    elif centroids is not None:
        k = len(centroids)
    elif k is not None:
        # Forgy initialization method: choose k data points randomly.
        centroids = data[np.random.choice(np.arange(len(data)), k, False)]
    else:
        raise RuntimeError("Need a value for k or centroids.")

    da_data = da.from_array(data, chunks = multiprocessing.cpu_count())
    da_centroids = da.from_array(centroids, chunks = multiprocessing.cpu_count())

    i = 0
    for _ in range(max(steps, 1)):
        print "Iteration : ", i
        i += 1
        # Squared distances between each point and each centroid.
        sqdists = euclidean(da_centroids, da_data)

        # Index of the closest centroid to each data point.
        da_clusters = da.argmin(sqdists, axis=0)

        da_new_centroids = cluster_centroids(da_data, da_clusters, k)
        if np.array_equal(da_new_centroids.compute(), da_centroids.compute()):
            break

        da_centroids = da_new_centroids

    return da_clusters, da_centroids

if __name__ == '__main__':

    # Generate sample data
    centers = [[1, 1], [-1, -1], [1, -1]]

    X, labels_true = make_blobs(n_samples=50, centers=centers, cluster_std=0.5,
		                    random_state=0)

    result = kmeans(X, k=10)
    dot_graph(result[0].dask, filename='clusters')
    dot_graph(result[1].dask, filename='centroids')
    
    print "Result:\nClusters"
    print result[0].compute()

    print "Centroids"
    print result[1].compute()

    print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X.tolist(), result[0].compute().tolist(), metric='euclidean'))
