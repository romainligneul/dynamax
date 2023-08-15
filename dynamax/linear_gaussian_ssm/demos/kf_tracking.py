# This demo provides a basic example of Kalman filtering and
#  smoothing with dynamax.
from jax import numpy as jnp
from jax import random as jr
from matplotlib import pyplot as plt

from dynamax.utils.plotting import plot_uncertainty_ellipses
from dynamax.linear_gaussian_ssm import LinearGaussianSSM


def kf_tracking():
    """Kalman filter tracking demo."""
    state_dim = 4
    emission_dim = 2
    delta = 1.0

    lgssm = LinearGaussianSSM(state_dim, emission_dim)

    # Manually chosen parameters
    initial_mean = jnp.array([8.0, 10.0, 1.0, 0.0])
    initial_covariance = jnp.eye(state_dim) * 0.1
    dynamics_weights = jnp.array([[1, 0, delta, 0], [0, 1, 0, delta], [0, 0, 1, 0], [0, 0, 0, 1]])
    dynamics_covariance = jnp.eye(state_dim) * 0.001
    emission_weights = jnp.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    emission_covariance = jnp.eye(emission_dim) * 1.0

    # Initialize model
    params, _ = lgssm.initialize(
        jr.PRNGKey(0),
        initial_mean=initial_mean,
        initial_covariance=initial_covariance,
        dynamics_weights=dynamics_weights,
        dynamics_covariance=dynamics_covariance,
        emission_weights=emission_weights,
        emission_covariance=emission_covariance,
    )

    num_timesteps = 15
    key = jr.PRNGKey(310)
    x, y = lgssm.sample(params, key, num_timesteps)
    lgssm_posterior = lgssm.smoother(params, y)
    return x, y, lgssm_posterior


def plot_lgssm_posterior(post_means, post_covs, ax=None, ellipse_kwargs={}, legend_kwargs={}, **kwargs):
    """Plot posterior means and covariances for the first two dimensions of
     the latent state of a LGSSM.

    Args:
        post_means: array(T, D).
        post_covs: array(T, D, D).
        ax: matplotlib axis.
        ellipse_kwargs: keyword arguments passed to matplotlib.patches.Ellipse().
        **kwargs: passed to ax.plot().
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Select the first two dimensions of the latent space.
    post_means = post_means[:, :2]
    post_covs = post_covs[:, :2, :2]

    # Plot the mean trajectory
    ax.plot(post_means[:, 0], post_means[:, 1], **kwargs)
    # Plot covariance at each time point.
    plot_uncertainty_ellipses(post_means, post_covs, ax, **ellipse_kwargs)

    ax.axis("equal")

    if "label" in kwargs:
        ax.legend(**legend_kwargs)

    return ax


def plot_kf_tracking(x, y, lgssm_posterior):
    """Plot the results of the Kalman filter tracking demo."""
    observation_marker_kwargs = {"marker": "o", "markerfacecolor": "none", "markeredgewidth": 2, "markersize": 8}
    dict_figures = {}

    # Plot Data
    fig1, ax1 = plt.subplots()
    ax1.plot(*x[:, :2].T, marker="s", color="C0", label="true state")
    ax1.plot(*y.T, ls="", **observation_marker_kwargs, color="tab:green", label="emissions")
    ax1.legend(loc="upper left")

    # Plot Filtering
    fig2, ax2 = plt.subplots()
    ax2.plot(*y.T, ls="", **observation_marker_kwargs, color="tab:green", label="observed")
    ax2.plot(*x[:, :2].T, ls="--", color="darkgrey", label="true state")
    plot_lgssm_posterior(
        lgssm_posterior.filtered_means,
        lgssm_posterior.filtered_covariances,
        ax2,
        color="tab:red",
        label="filtered means",
        ellipse_kwargs={"edgecolor": "k", "linewidth": 0.5},
        legend_kwargs={"loc": "upper left"},
    )

    # Plot Smoothing
    fig3, ax3 = plt.subplots()
    ax3.plot(*y.T, ls="", **observation_marker_kwargs, color="tab:green", label="observed")
    ax3.plot(*x[:, :2].T, ls="--", color="darkgrey", label="true state")
    plot_lgssm_posterior(
        lgssm_posterior.smoothed_means,
        lgssm_posterior.smoothed_covariances,
        ax3,
        color="tab:red",
        label="smoothed means",
        ellipse_kwargs={"edgecolor": "k", "linewidth": 0.5},
        legend_kwargs={"loc": "upper left"},
    )

    dict_figures["kalman_tracking_truth"] = fig1
    dict_figures["kalman_tracking_filtered"] = fig2
    dict_figures["kalman_tracking_smoothed"] = fig3

    return dict_figures


def main(test_mode=False):
    x, y, lgssm_posterior = kf_tracking()
    if not test_mode:
        dict_figures = plot_kf_tracking(x, y, lgssm_posterior)
        plt.show()


if __name__ == "__main__":
    main()
