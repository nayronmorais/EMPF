"""Module contains functions for compute similarity."""

from numba import njit
import numpy as np


@njit
def imk(dist, alpha):
    """
    Inverse Multiquadratic Kernel (IMK).

    Parameters
    ----------
    dist : float or numpy.ndarray, shape=(m,)
        Distances.

    alpha : float
        Control the decay of the function.

    Results
    -------
    sim : float or numpy.ndarray, shape=(m)
        Similatity result.

    """
    return 1 / np.sqrt(1 + alpha ** 2 * dist ** 2)


@njit
def wsf(dist, alpha):
    """
    Wegerich Similarity Function (WSF).

    Parameters
    ----------
    dist : float or numpy.ndarray, shape=(m,)
        Distances.

    alpha : float
        Control the decay of the function.

    Returns
    -------
    sim : float or numpy.ndarray, shape=(m)
        Similatity result.

    """
    return 1 / (1 + alpha * dist)


@njit
def cck(dist, alpha):
    """
    Cauchy Kernel Function (CCK).

    Parameters
    ----------
    dist : float or numpy.ndarray, shape=(m,)
        Distances.

    alpha : float
        Control the decay of the function.

    Returns
    -------
    sim : float or numpy.ndarray, shape=(m)
        Similatity result.

    """
    return 1 / (1 + (alpha ** 2) * (dist ** 2))


@njit
def lk(dist, alpha):
    """
    Exponential or Laplace Kernel (LK).

    Parameters
    ----------
    dist : float or numpy.ndarray, shape=(m,)
        Distances.

    alpha : float
        Control the decay of the function.

    Results
    -------
    sim : float or numpy.ndarray, shape=(m)
        Similatity result.

    """
    return np.exp(-alpha * dist)


@njit
def rbf(dist, alpha):
    """
    Radial Basis Function (RBF).

    Parameters
    ----------
    dist : float or numpy.ndarray, shape=(m,)
        Distances.

    alpha : float
        Control the decay of the function.

    Results
    -------
    sim : float or numpy.ndarray, shape=(m)
        Similatity result.

    """
    return np.exp(-alpha * dist ** 2)


def sto(dist, d, epsilon=1e-10):
    """
    Saturated Triangular Operator (STO).

    Eq.:
        d - dist, if dist <= d + epsilon
        epsilon, if dist > d + epsilon

    Parameters
    ----------
    dist : float or numpy.ndarray, shape=(m,)
        Distances.

    d : float
        Threshold based on observartions varince.

    epsilon : float
        Value for similarity when the distance is greater than `d`.

    Results
    -------
    sim : float or numpy.ndarray, shape=(m)
        Similatity result.

    """
    less_eq_d = np.less_equal(dist, d + epsilon)

    sim = dist.copy()
    sim[less_eq_d] = 1 - dist[less_eq_d] / d  # Adjust to interval [0, 1]
    sim[~less_eq_d] = epsilon

    return sim
