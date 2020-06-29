"""Module contains helpers functions for clustering methods."""

import numpy as np
from matplotlib.patches import Ellipse
from matplotlib import pyplot as plt
from scipy.stats import chi2


def plot_2d_gauss_cluster(center, inv_cov, gamma=0.99,
                          ax=None, ls=':', lw=2, color='green'):
    """
    Plot a 2D cluster with gaussian shape.

    Parameters
    ----------
    center : numpy.ndarray, shape=(1, n)
        Center of the cluster.

    inv_cov : numpy.ndarray, shape(n, n)
        Inverse of the covariance matrix.

    ax : matplotlib.axes.Axes
        The axes when the ellipse will be shown.

    gamma : float, optional
        Confidence interval. The default is 0.99.

    ls : str, optional
        The matplotlib line style. Can be any of
        `matplotlib.lines.lineStyles`. The default is `:`.

    lw : float, optional
        The width of the edge (in pt). The default is 2.

    color : str or iterator, optional
        The color of the edge. Can be any matplotlib color name or
        RGB values. The default is `green`.

    Returns
    -------
    ellipse : matplotlib.patches.Ellipse
        The cluster representation.

    """
    if inv_cov.shape[0] != 2:
        raise Exception('Available only in 2d problems.')

    if ax is None:
        ax = plt.gca()

    radii = chi2.ppf(gamma, 2)

    eigenvalues, eigenvectors = np.linalg.eigh(inv_cov)

    ord_ = np.argsort(eigenvalues)

    eigenvalues = eigenvalues[ord_]
    eigenvectors = eigenvectors[:, ord_]

    theta = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width = 2 * np.sqrt(radii * (1 / eigenvalues[0]))
    height = 2 * np.sqrt(radii * (1 / eigenvalues[1]))

    ellipse = Ellipse(center[0], width, height, angle=theta, fill=False,
                      zorder=10 ** 2, edgecolor=color, ls=ls, lw=lw, alpha=0.8)

    ax.add_artist(ellipse)

    return ellipse


def plot_3d_gauss_cluster():
    """Plot a 3D cluster with gaussian shape."""
    raise NotImplementedError
