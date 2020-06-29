"""Module contains functions for incremental update of statistics values."""

import numpy as np
from numba import njit


@njit
def mean(x, old_mean, old_num_samples):
    """
    Incremental update (using unbiased formula) of sample mean.

    Parameters
    ----------
    x : numpy.ndarray, shape=(1, n)
        New point.

    old_mean : numpy.ndarray, shape=(1, n)
        Old sample mean.

    old_num_samples : int
        Old number of samples.

    Returns
    -------
    new_mean : numpy.ndarray, shape=(1, n)
        Updated sample mean.

    References
    ----------
    Lughofer, Edwin. Evolving fuzzy systems-methodologies, advanced concepts
    and applications. Vol. 53. Berlin: Springer, 2011.

    """
    new_num_samples = old_num_samples + 1

    new_mean = (old_num_samples / new_num_samples) * old_mean + \
               ((1 / new_num_samples) * x)

    return new_mean


@njit
def inv_cov(x, mean, old_inv_cov, old_num_samples):
    """
    Incremental update of inverse sample covariance.

    Parameters
    ----------
    x : numpy.ndarray, shape=(1, n)
        New point.

    mean : numpy.ndarray, shape=(1, n)
        Current sample mean.

    old_inv_cov : numpy.ndarray, shape=(n, n)
        Old inverse sample covariance matrix.

    old_num_samples : int
        Old number of samples.

    Returns
    -------
    new_inv_cov : numpy.ndarray, shape=(n, n)
        Updated inverse sample covariance matrix.

    References
    ----------
    Lughofer, Edwin. Evolving fuzzy systems-methodologies, advanced concepts
    and applications. Vol. 53. Berlin: Springer, 2011.

    """
    alpha = 1 / (old_num_samples + 1)
    diff = x - mean
    dot_inv_cov_x = np.dot(old_inv_cov, diff.T)

    part_1 = old_inv_cov / (1 - alpha)
    part_2 = alpha / (1 - alpha)

    part_3_num = np.dot(dot_inv_cov_x, dot_inv_cov_x.T)
    part_3_den = 1 + alpha * np.dot(diff, dot_inv_cov_x)

    new_inv_cov = part_1 - part_2 * (part_3_num / part_3_den)

    return new_inv_cov
