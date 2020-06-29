"""Module provide simulation data of bechmarks processes and synthetic data."""

import numpy as np

from scipy.stats import norm

from .data import fetch_remote_data, build_lagged_matrix


def build_2d_gauss_data(mu_1, mu_2, sig_1, sig_2, samples, changes={},
                        w=50, alpha=0.1, lags=0):
    """Build a bivarite dataset following a Gaussian distribution.

    Parameters
    ----------
    mu_1 : float
        Mean of x_1.

    mu_2 : float
        Mean of x_2.

    sig_1 : float
        Standard deviation of x_1.

    sig_2 : float
        Standard deviation of x_2.

    samples : int
        Size of samples will be generated.

    changes : dict, optional
        Sudden or incipients changes to be were add. The default is
        an empty dict.

        Can be:
        - incip: Incipient.
        - sudden: Sudden.

        Example:
            {
                'incip': [{'add': 50, 'where': (start, stop)}],
                'sudden': [{'add': 50, 'where': (start, stop)}]
            }

        In `add` give the magnitude of the change and in `where`, where
        the change will be added. start >=0 and stop < samples.


    w : int, optional
        Size of window to compute the moving average in incipients changes.
        The default is 50.

    alpha : float, optional
        Weight for linear dependence (correlation) between x_1 and x_2.

        x_2 = ... alpha * x_1 ...

    lags : int, optional
        If greater than 0 it's added time dependence.

    Notes
    -----
    - x_1 is defined as following:

        x_1 = N(mu_1, sig_1 ** 2),

        where N is the normal distribuition with mean `mu_1' and variance
        `sig_1 ** 2`.

    - x_2 is defined as following:

        x_2 = N(mu_2, sig_2 ** 2) + alpha * x_1 + N(0, 1),

        where `alpha` is a weight and `N(0, 1)` is an white noise.

    Returns
    -------
    x_1 : numpy.ndarray, shape(samples,)
        1th random variable.

    x_2 : numpy.ndarray, shape(samples,)
        2th random variable..

    """
    white_noise = norm.rvs(loc=0, scale=1, size=samples)
    x_1 = norm.rvs(loc=mu_1, scale=sig_1, size=samples)

    for change_name, changes_to_apply in changes.items():

        change_name = change_name.lower()

        for change in changes_to_apply:

            to_sum = change['add']
            start, stop = change['where']

            num_samples = stop - start
            mean_est = np.mean(x_1[start - w: start])

            if change_name == 'incip':

                add = np.linspace(start=0, stop=to_sum, num=num_samples)

                x_1[start: stop] = norm.rvs(loc=mean_est,
                                            scale=sig_1,
                                            size=num_samples) + add

            elif change_name == 'sudden':

                x_1[start: stop] += norm.rvs(loc=to_sum,
                                             scale=sig_1,
                                             size=num_samples)

    x_2 = norm.rvs(loc=mu_2, scale=sig_2, size=samples) + \
            alpha * x_1 + white_noise

    # Time dependence.
    if lags > 0:

        lagged_mat = build_lagged_matrix(np.c_[x_1, x_2], lags)

        end_1th = 2 + lags
        end_2th = end_1th + lags

        x_1 += np.sum(alpha * lagged_mat[:, 2: end_1th], axis=1)
        x_2 += np.sum(alpha * lagged_mat[:, end_1th: end_2th], axis=1)

    return x_1, x_2


def fetch_damadics(operations=None, forcedownload=False):
    """
    Get the DAMADICS Simulation Dataset.

    Parameters
    ----------
    operations : str, optional
        The desired operation name. If None, get all. The default is None.

        Can be:
            - normal
            - f01/[s,m,b or i]
            - f02/[s,m,b or i]
                .
                .
                .
            - f19/[s,m,b or i]

        more the type, s (small), m (medium), b (big) or i (incipient).

    forcedownload : bool, optional
        Do redowonload, even if the files alredy downloaded.
        The default is False.

    Raises
    ------
    Exception
        Case operation not recognized.

    Returns
    -------
    all_data : dict<dict<list<str>>, dict<numpy.ndarray>>
        The variables names and the data for each desired operation.

    Notes
    -----
        - Attention on selection faulty type, the type  should be allowed.
        - The faltys start from 900s. The variable `Fi` indicate when is in
            faulty.

    """

    url = 'https://raw.githubusercontent.com/nayronmorais/EMPF/' + \
          'master/data/external/damadics/'

    operations_desc = fetch_remote_data({'desc': 'operations.info'},
                                        url=url, sep=';', header=0,
                                        forcedownload=False)
    operations_desc = operations_desc['desc']

    all_operationmodes = {'normal': 'normal.csv'}

    sufix = {'small': 's', 'medium': 'm', 'big': 'b', 'incipient': 'i'}

    for i, row in operations_desc.iterrows():
        opname = row['fault_name']
        for fsf, sf in sufix.items():
            if bool(row[fsf]):
                all_operationmodes[opname + sf] = opname + sf + '.csv'

    operationmodes = all_operationmodes

    if operations is not None:
        operationmodes = {}
        try:
            for operation in operations:
                operationmodes[operation] = all_operationmodes[operation]
        except KeyError:
            raise Exception(f'Operation `{operation}` not recognized.')

    all_data = {'names': ['CV', 'P1', 'P2', 'X', 'F', 'T1', 'Fi']}
    dict_numeric_data = fetch_remote_data(operationmodes, url,
                                          forcedownload=forcedownload,
                                          sep=';',
                                          header=0)

    for operation, data in dict_numeric_data.items():
        dict_numeric_data[operation] = data.values.astype(np.float64)

    all_data['data'] = dict_numeric_data

    return all_data


def fetch_tep(operations=None, forcedownload=False):
    """
    Get the Tennessee Eastman Process Simulation Dataset.

    Parameters
    ----------
    operations : str, optional
        The desired operation name. If None, get all. The default is None.

        Can be:
            - normal
            - f01
            - f02
                .
                .
                .
            - f21

    forcedownload : bool, optional
        Do redowonload, even if the files alredy downloaded.
        The default is False.

    Raises
    ------
    Exception
        Case operation not recognized.

    Returns
    -------
    all_data : dict<dict<list<str>>, dict<numpy.ndarray>>
        The variables names and the data for each desired operation.

    """
    url = 'https://raw.githubusercontent.com/' + \
        'nayronmorais/EMPF/master/data/external/te/train/'

    idxs_f = [str(idx_f).zfill(2) for idx_f in range(1, 22)]
    faultys = {f'f{idx}': f'd{idx}.dat' for idx in idxs_f}

    all_operationmodes = {'normal': 'd00.dat'}
    all_operationmodes.update(faultys)

    operationmodes = all_operationmodes

    if operations is not None:
        operationmodes = {}
        try:
            for operation in operations:
                operationmodes[operation] = all_operationmodes[operation]
        except KeyError:
            raise Exception(f'Operation `{operation}` not recognized.')

    columns_names = [f'XMEAS_{idx}' for idx in range(1, 42)] + \
                    [f'XMV_{idx}' for idx in range(1, 12)]

    all_data = {'names': columns_names}

    dict_numeric_data = fetch_remote_data(operationmodes, url,
                                          forcedownload=forcedownload)

    for operation, data in dict_numeric_data.items():

        data = data.values.astype(np.float64)
        dict_numeric_data[operation] = data

        if data.shape[1] > 52:
            dict_numeric_data[operation] = data.T

    all_data['data'] = dict_numeric_data

    return all_data


def fetch_synthetic_2d(lags=0):
    """
    Build synthetic 2d data.

    Parameters
    ----------
    lags : int, optional
        If greater than 0 it's added time dependence.
        The default is 0.

    Returns
    -------
    data : numpy.ndarray, shape=(3000, 2)
        Synthetic data.

    """
    seed = 98
    np.random.seed(seed)

    mu_1, mu_2 = 1, 30
    sigma_1, sigma_2 = 3, 1

    num_samples = 3000

    changes = {'incip': [
                         {'add': 50, 'where': (1000, 1300)},
                         {'add': 0, 'where': (1300, 1600)},
                         {'add': -50, 'where': (1600, 1650)}
                    ],
               'sudden': [
                          {'add': -50, 'where': (2000, 2200)}
                    ]
               }

    labels = np.ones(num_samples, dtype=np.uint8)
    labels[1070:1250] = 2
    labels[1250:1602] = 3
    labels[1602:1640] = 2
    labels[2000:2200] = 4

    x_1, x_2 = build_2d_gauss_data(mu_1, mu_2, sigma_1, sigma_2,
                                   samples=num_samples, changes=changes,
                                   alpha=0.15, w=10, lags=lags)

    return np.c_[np.arange(1, 3001), x_1, x_2, labels]


def synthetic_base_example():
    """Build e plot synthetic base."""
    from matplotlib import pyplot as plt

    seed = 98
    np.random.seed(seed)

    mu_1, mu_2 = 1, 30
    sigma_1, sigma_2 = 2, 0.5

    num_samples = 3000

    changes = {'incip': [
                         {'add': 50, 'where': (1000, 1300)},
                         {'add': 0, 'where': (1300, 1600)},
                         {'add': -50, 'where': (1600, 1650)}
                    ],
               'sudden': [
                          {'add': -50, 'where': (2000, 2200)}
                    ]
               }

    x_1, x_2 = build_2d_gauss_data(mu_1, mu_2, sigma_1, sigma_2,
                                   samples=num_samples, changes=changes,
                                   alpha=0.15, w=50, lags=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.plot(x_1, lw=1, color='b', label='$x_1$', ls='-')

    ax1.plot(x_2, lw=1, color='g', label='$x_2$', ls='--')
    ax1.legend(labelspacing=0.25)

    ax2.scatter(x_1, x_2, s=2*10, color='b', edgecolor='k', marker='.',
                linewidths=0.1)
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')

    plt.tight_layout()
    plt.show()

    print(np.corrcoef(x_1, x_2))


if __name__ == '__main__':

    synthetic_base_example()
