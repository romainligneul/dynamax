from jax import Array
from jax import numpy as jnp
from jax import random as jr


def kmeans_centers_sklearn(k:int, X:Array, key:Array):
    """
    Compute the cluster centers using the K-means algorithm.

    Args:
        k (int): The number of clusters.
        X (Array(N, D)): The input data array. N samples of dimension D.
        key (Array): The random seed array.

    Returns:
        Array(k, D): The cluster centers.
    """
    from sklearn.cluster import KMeans

    key, subkey = jr.split(key)  # Create a random seed for SKLearn.
    sklearn_key = jr.randint(subkey, shape=(), minval=0, maxval=2147483647)  # Max int32 value.
    km = KMeans(k, random_state=int(sklearn_key)).fit(X)
    return jnp.array(km.cluster_centers_)

