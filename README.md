#Scalable K-means Clustering

Dask is a flexible parallel computing library for analytic computing and provides parallelized NumPy array and Pandas DataFrame objects.

While the ordinary Lloyd's algorithm could fit into memory on most modern machines for small datasets, here we'll take a more scalable approach, utilizing Dask to do our data ingestion and manipulation out-of-core.

A good place to start with : https://jakevdp.github.io/blog/2015/08/14/out-of-core-dataframes-in-python/

Sample Instruction to begin with:

```

    import numpy as np
    from sklearn.datasets.samples_generator import make_blobs
    import dask.array as da
    from dask.dot import dot_graph
    from sklearn import metrics

    from kmeans import kmeans

    # Generate sample data
    centers = [[1, 1], [-1, -1], [1, -1]]

    X, labels_true = make_blobs(n_samples=30, centers=centers, cluster_std=0.5,
		                    random_state=0)

    result = kmeans(X, k=10)

    # Print Task-Scheduling graph
    dot_graph(result[0].dask, filename='clusters')
    dot_graph(result[1].dask, filename='centroids')
    
    print "Result:\nClusters"
    print result[0].compute()

    print "Centroids"
    print result[1].compute()

    print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X.tolist(), result[0].compute().tolist(), metric='euclidean'))

    

```
