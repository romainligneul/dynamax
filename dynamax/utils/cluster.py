from typing import NamedTuple

from jax import Array, lax
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


class KMeansState(NamedTuple):
    centroids: Array
    assignments: Array
    prev_centroids: Array
    itr: int


def kmeans_centers_jax(
    X: Array, k: int, key: Array = jr.PRNGKey(0), max_iters: int = 1000, 
) -> KMeansState:
    """
    Perform k-means clustering using JAX.

    Args:
        X (Array): The input data array of shape (n_samples, n_features).
        k (int): The number of clusters.
        max_iters (int, optional): The maximum number of iterations. Defaults to 1000.
        key (PRNGKey, optional): The random key for initialization. Defaults to jr.PRNGKey(0).

    Returns:
        KMeansState: A named tuple containing the final centroids array of shape (k, n_features),
        the assignments array of shape (n_samples,) indicating the cluster index for each sample,
        the previous centroids array of shape (k, n_features), and the number of iterations.
    """

    def _update_centroids(X: Array, assignments: Array):
        group_counts = jnp.bincount(assignments, minlength=k, length=k)
        group_sums = jnp.where(
            assignments[:, None, None] == jnp.arange(k)[None, :, None],
            X[:, None, :],
            0.0,
        ).sum(axis=0)
        return group_sums / group_counts[:, None]

    def _update_assignments(X, centroids):
        return jnp.argmin(jnp.linalg.norm(X[:, None] - centroids, axis=2), axis=1)

    def body(carry: KMeansState):
        centroids, assignments, *_ = carry
        new_centroids = _update_centroids(X, assignments)
        new_assignments = _update_assignments(X, new_centroids)
        return KMeansState(new_centroids, new_assignments, centroids, carry.itr + 1)

    def cond(carry: KMeansState):
        return jnp.any(carry.centroids != carry.prev_centroids) & (
            carry.itr < max_iters
        )

    def init():
        # Initialize centroids as random data points
        init_centroids = X[jr.choice(key, X.shape[0], (k,), replace=False)]
        # TODO: Maybe I need to do one step of assignment before starting the loop
        init_assignments = _update_assignments(X, init_centroids)
        # Perform one iteration to update centroids
        centroids = _update_centroids(X, init_assignments)
        assignments = _update_assignments(X, centroids)
        return KMeansState(centroids, assignments, init_centroids, 0)

    state = lax.while_loop(cond, body, init())

    return state

