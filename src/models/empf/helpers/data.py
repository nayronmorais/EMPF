"""Module contains functions for build and preprocess dataset."""

from os import path

import numpy as np
import pandas as pd
from scipy.stats import norm

from .file import download_files, DEFAULT_DATA_DIR_NAME


def build_lagged_matrix(X, lags=3):
    """
    Build a lagged variable for each ones in X.

    Parameters
    ----------
    X : numpy.ndarray, shape=(m, n)
        Reference points.

    lags : int, optional
        The desired lags. The default is 3.

    Returns
    -------
    lagged_matrix : numpy.ndarray, shape=(m, n * (1 + l))
        Matrix com lagged variables.

    """
    if lags == 0:
        return X

    def apply_column(row_idxs, data_row):

        return data_row[row_idxs]

    rows, cols = X.shape
    new_cols = cols + cols * lags

    lagged_mat = np.zeros(shape=(rows, new_cols), dtype=np.float64)
    lagged_mat[:, :cols] = X
    lagged_mat[0, cols:] = 0

    # Building matrix with lagged indexes.
    rows_idxs = np.arange(rows)[:, None]
    lag_vec = list(range(1, lags + 1))
    lag_mat_idxs = np.tile(lag_vec, reps=(rows, 1))
    lag_mat_idxs = rows_idxs - lag_mat_idxs
    lag_mat_idxs_zeros = np.triu_indices(lags)

    l = cols

    for i in range(cols):

        row = lagged_mat[:, i]

        new_lagged_rows = np.apply_along_axis(apply_column, 0, lag_mat_idxs, row)
        new_lagged_rows[lag_mat_idxs_zeros] = 0
        lagged_mat[:, l:l + lags] = new_lagged_rows

        l += lags

    return lagged_mat


def add_lagged_vars(x, matrix, lags=3):
    """
    Build a sample with lagged variables.

    Parameters
    ----------
    x : numpy.ndarray, shape=(1, n)
        Point to add lagged vars.

    matrix : numpy.ndarray, shape=(m, n)
        Reference points.

    lags : int, optional
        The desired lags. The default is 3.

    Returns
    -------
    lagged_sample : numpy.ndarray, shape=(1, n * (1 + l))
        Sample com lagged variables.

    """
    if lags == 0:
        return x

    cols = x.shape[1]
    new_cols = cols + cols * lags

    new_x = np.zeros(shape=(1, new_cols), dtype=np.float64)
    new_x[:, :cols] = x

    l = cols

    lag_idxs = np.array([-i for i in range(1, lags + 1)])[None, :]
    apply = lambda idx, r: r[idx]

    for i in range(cols):

        column = matrix[:, i]

        new_vars = np.apply_along_axis(apply, 0, lag_idxs, column)
        new_x[0, l: l + lags] = new_vars

        l += lags

    return new_x

def moving_average(data, w=3):
    """
    Compute a moving average.

    Parameters
    ----------
    data : numpy.ndarray, shape=(n, m)
        The data samples
        .
    w : int, optional
        Window lenght. The default is 3.

    Returns
    -------
    data_averaged : numpy.ndarray, shape=(n - w, m)
        The valid data samples

    """
    weights = np.ones(w) / w
    return np.convolve(data, weights, mode='valid')


def fetch_remote_data(in_out_filenames, url, sep='\s+', header=None,
                      forcedownload=False):
    """
    Get remote dataset.

    Parameters
    ----------
    in_out_filenames : dict<str, str>
        The out name file as key and the filename as value.

    url : str
        The url to fetch the fileanames.

    sep : str, optional
        Delimiter of the file. The default is spaces.

    header : int, optional
        If 0 use the firt row as header. The default is None (without header).

    forcedownload : bool, optional
        Do redowonload, even if the operations alredy downloaded.
        The default is False.

    Returns
    -------
    data : dict<str, pandas.DatFrame>
        The out file names and the data for each file.

    """
    data = {}

    for out_name, filename in in_out_filenames.items():

        fpath = path.join(DEFAULT_DATA_DIR_NAME, filename)

        if not path.exists(fpath) or forcedownload:
            download_files(url, (filename,))

        data_aux = pd.read_csv(fpath, sep=sep, header=header)

        data[out_name] = data_aux

    return data
