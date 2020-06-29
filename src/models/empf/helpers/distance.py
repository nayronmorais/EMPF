"""Module contains functions for compute distances."""

from numba import njit
import numpy as np


@njit
def mahalanobis(x, mean, inv_cov, square=False):
    """
    Compute the Mahalanobis distance.

    Parameters
    ----------
    x : numpy.ndarray, shape=(m, n)
        N-dimensional set of points.

    mean : numpy.ndarray, shape=(1, n)
        Mean point of the data.

    square : bool, optional
        If returned distance is on same scale of `x` (`square=False`) or it is
        squared (`square=True`). The default is False.

    Returns
    -------
    dist : numpy.array, shape=(m, 1)
        Distance among `mean` and the points in `x`.

    """
    diff = x - mean

    dist = np.dot(diff, np.dot(inv_cov, diff.T))
    dist = np.diag(dist)

    if not square:
        dist = np.sqrt(dist)

    return dist.reshape(mean.shape[0], 1)


@njit
def euclidean(x, y, square=False):
    """
    Compute the Euclidean distance.

    Parameters
    ----------
    x : numpy.ndarray, shape=(1, n)
        N-dimensional point.

    y : numpy.ndarray, shape=(m, n)
        Set of points N-dimensional .

    square : bool, optional
        If returned distance is on same scale of `x` (`square=False`) or it is
        squared (`square=True`). The default is False.

    Returns
    -------
    dist : numpy.array, shape=(m,)
        Distance among `x` and the points in `y`.

    """
    diff = x - y

    dist = np.dot(diff, diff.T)
    dist = np.diag(dist)

    if not square:
        dist = np.sqrt(dist)

    return dist
