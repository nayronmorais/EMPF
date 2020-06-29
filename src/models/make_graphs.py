import numpy as np
import sklearn
import math
from scipy.stats import chi2
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from random import *
from matplotlib.patches import Ellipse
from numpy.linalg import cholesky

import pandas as pd

import matplotlib as mpl
import seaborn as snss


# https://github.com/joferkington/oost_paper_code/blob/master/error_ellipse.py
def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).
    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * np.sqrt(nstd * vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

# generate a bivariate Gaussian distribution points set
def generate_gaussian(mu, sigma, sample_num = 300):
    # mu = np.array([[1, 5]])
    # sigma = np.array([[1, 0.5], [1.5, 3]])
    R = cholesky(sigma)
    s = np.dot(np.random.randn(sample_num, 2), R) + mu
    return s
    # plt.plot(s[:,0], s[:,1], 'o')
    # plt.show()

def plot_bivariate_gaussian(**kwargs):
    mu = np.array([[10, 5]])
    sigma = np.array([[3, 0.7], [0.7, 2]])
    s1 = generate_gaussian(mu, sigma, sample_num=1000)

    mu = np.array([[15, 20]])
    sigma = np.array([[3, -0.3], [-0.3, 6]])
    s2 = generate_gaussian(mu, sigma, sample_num=1000)

    # kwrg = {'edgecolor':'k', 'linewidth':0.5}


    plt.scatter(s1[:,0], s1[:,1], c='b', marker='o', s=5)
    plt.scatter(s2[:,0], s2[:,1], c='b', marker='o', s=5)
    plot_point_cov(s1, nstd = np.sqrt(chi2.ppf(0.99, 2)), **kwargs)
    plot_point_cov(s1, nstd = np.sqrt(chi2.ppf(0.999, 2)), **kwargs)
    plot_point_cov(s1, nstd = np.sqrt(chi2.ppf(0.9999, 2)), **kwargs)

    plot_point_cov(s2, nstd = np.sqrt(chi2.ppf(0.99, 2)), **kwargs)
    plot_point_cov(s2, nstd = np.sqrt(chi2.ppf(0.999, 2)), **kwargs)
    plt.show()

    plt.xlim(0, 25)
    plt.ylim(-5, 30)

    return plt.gca()

def plot_single_gaussian():
    mu = np.array([[1, 5]])
    sigma = np.array([[1, 0.5], [1.5, 3]])
    s1 = generate_gaussian(mu, sigma)

    mu = np.array([[4, 11]])
    sigma = np.array([[2.4, 3.1], [1.5, 3.7]])
    s2 = generate_gaussian(mu, sigma)

    X = np.hstack((s1[:,0], s2[:,0]))
    Y = np.hstack((s1[:,1], s2[:,1]))
    X.shape = (600, 1)
    Y.shape = (600, 1)
    points = np.c_[X,Y]

    kwrg = {'edgecolor':'k', 'linewidth':0.5}

    plt.plot(s1[:,0], s1[:,1], 'go')
    plt.plot(s2[:,0], s2[:,1], 'go')
    plot_point_cov(points, nstd = 2, alpha = 0.7, color = 'pink', **kwrg)
    plt.show()

# %% Exibe exemplo de gaussianas

from scipy.spatial import distance

np.random.seed(10000)

mu1, mu2 = np.array([[10, 5]]), np.array([[10, 17]])
sigma1, sigma2 = np.array([[3, 0.7], [0.7, 2]]), np.array([[3, -0.3], [-0.3, 6]])

sr1 = generate_gaussian(mu1, sigma1, sample_num=50)
sr2 = generate_gaussian(mu2, sigma2, sample_num=50)

nstd_80 = np.sqrt(chi2.ppf(0.8, 2))
nstd_99 = np.sqrt(chi2.ppf(0.99, 2))
nstd_999 = np.sqrt(chi2.ppf(0.999, 2))

fig, ax = plt.subplots()

ax.set_xlim(3, 16)
ax.set_ylim(-1, 12)

ax.scatter(sr1[:, 0], sr1[:, 1], s=10, zorder=10**2, marker='o')
ax.scatter(mu1[0, 0], mu1[0, 1], color='tab:red', s=17, zorder=10**2, marker='D')

# ax.scatter(sr2[:, 0], sr2[:, 1], s=4, zorder=10**2)

ellkargs = {'edgecolor': 'b', 'linewidth': 0.5, 'fill': True, 'linestyle': '-', 'facecolor': 'tab:blue', 'alpha': 0.2}

plot_cov_ellipse(sigma1, mu1[0, :], nstd_99, ax=ax, **ellkargs)

dist = distance.cdist(sr1, mu1, metric='mahalanobis', VI=np.linalg.pinv(sigma1))

idx = np.argmax(dist)

max_x = sr1[idx]

ax.text(max_x[0] - 3, max_x[1] + 0.6, '$s_{lk}(\\mu, x) < \\tau$', fontdict={'fontsize': 10, 'fontweight': 'normal'})
ax.text(max_x[0] + 6, max_x[1] + 1.5, '$s_{lk}(\\mu, x_i) \\geq \\tau$', fontdict={'fontsize': 10, 'fontweight': 'normal'})

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
# plot_cov_ellipse(sigma2, mu2[0, :], nstd_99, ax=ax, **ellkargs)

# %% Distância entre todos

from scipy.spatial import distance

np.random.seed(10000)

mu1 = np.array([[10, 5]])
sigma1 = np.array([[3, 0.7], [0.7, 2]])
sr1 = generate_gaussian(mu1, sigma1, sample_num=50)

nstd_80 = np.sqrt(chi2.ppf(0.8, 2))
nstd_99 = np.sqrt(chi2.ppf(0.99, 2))
nstd_999 = np.sqrt(chi2.ppf(0.999, 2))

fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)

# ax.scatter(sr2[:, 0], sr2[:, 1], s=4, zorder=10**2)

ellkargs_m = {'edgecolor': 'b', 'linewidth': 0.5, 'fill': True, 'linestyle': '-', 'facecolor': 'tab:blue', 'alpha': 0.2}


dist = distance.cdist(sr1, mu1, metric='mahalanobis', VI=np.linalg.pinv(sigma1))

idx = np.argmax(dist)

max_x = sr1[idx]

sr1_all = sr1.copy()

sr1 = sr1[np.random.randint(0, 50, 10)[5:], :]

sr1 = np.vstack((sr1, max_x[None, :]))

# ax.scatter(mu1[0, 0], mu1[0, 1], color='tab:red', s=17, zorder=10**2, marker='D')

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:cyan']
markers = ['o', 's', '8', 'D', '*', 'P']
t = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

for i, x in enumerate(sr1):

    ellkargs = {'edgecolor': 'b', 'linewidth': 0.5,
                'fill': True, 'linestyle': '-', 'facecolor': colors[0], 'alpha': 0.2}

    col = min(i, 2)

    if (i / 6) < 0.5:
        row = 0
    else:
        row = 1
        col = i - 4

    # fig, ax = plt.subplots()
    ax[row, col].scatter(sr1_all[:, 0], sr1_all[:, 1], s=10, zorder=10**2, marker='o',  color='b')
    # plot_cov_ellipse(sigma1, mu1[0, :], nstd_99, ax=ax[row, col], **ellkargs_m)



    plot_cov_ellipse(sigma1, x, nstd_99, ax=ax[row, col], **ellkargs)
    ax[row, col].scatter(sr1[i, 0], sr1[i, 1], s=10, zorder=100**2, marker='o', color='tab:red')
    ax[row, col].set_xlim(0, 20)
    ax[row, col].set_ylim(-2, 14)

    ax[row, col].spines['top'].set_color('white')
    ax[row, col].spines['right'].set_color('white')
    ax[row, col].spines['top'].set_color('white')
    ax[row, col].spines['right'].set_color('white')

    # ax[row, col].set_xticks([])
    # ax[row, col].set_yticks([])
    # ax[row, col].set_title('$%s$' % t[i], fontdict={'fontsize': 10})

    # for xa in sr1_all:
    #     ax[row, col].plot([x[0], xa[0]], [x[1], xa[1]], linestyle=':', alpha=0.5, linewidth=1, color='tab:green')

    # ax.scatter(sr1_all[:, 0], sr1_all[:, 1], s=10, zorder=10**2, marker='o')
    # plot_cov_ellipse(sigma1, mu1[0, :], nstd_99, ax=ax, **ellkargs_m)



    # plot_cov_ellipse(sigma1, x, nstd_99, ax=ax, **ellkargs)
    # ax.scatter(sr1[i, 0], sr1[i, 1], s=17, zorder=100**2, marker='D', color='tab:red')
    # ax.set_xlim(0, 19)
    # ax.set_ylim(-1, 15)
    # # ax.set_xticks([])
    # # ax.set_yticks([])
    # # ax.set_title('$%s$' % t[i], fontdict={'fontsize': 10})

    # for xa in sr1_all:
    #     ax.plot([x[0], xa[0]], [x[1], xa[1]], linestyle=':', alpha=0.5, linewidth=1, color='tab:green')

# ax.set_xlabel('$x_1$')
# ax.set_ylabel('$x_2$')


# %% Região de composição (Apenas uma imagem)

from scipy.spatial import distance


np.random.seed(10000)

mu1 = np.array([[10, 5]])
sigma1 = np.array([[3, 0.7], [0.7, 2]])
sr1 = generate_gaussian(mu1, sigma1, sample_num=50)

nstd_95 = np.sqrt(chi2.ppf(0.99, 2))
nstd_90 = np.sqrt(chi2.ppf(0.9, 2))
nstd_10 = np.sqrt(chi2.ppf(0.05, 2))

fig, ax = plt.subplots()

dist = distance.cdist(sr1, mu1, metric='mahalanobis', VI=np.linalg.pinv(sigma1))

idxs = dist > (nstd_95)

no_in = sr1[idxs[:, 0]]

sr1 = sr1[np.invert(idxs[:, 0])]

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:cyan']
markers = ['o', 's', '8', 'D', '*', 'P']
t = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

ellkargs = {'edgecolor': 'purple', 'linewidth': 0.5,
                'fill': True, 'linestyle': '-', 'facecolor': 'c', 'alpha': 0.4}

ellkargs_m = {'edgecolor': 'b', 'linewidth': 0.5, 'fill': True, 'linestyle': '-',
              'facecolor': 'tab:blue', 'alpha': 0.2}


ax.scatter(mu1[0, 0], mu1[0, 1], color='tab:red', s=20, zorder=10**2, marker='D')
ax.scatter(sr1[:, 0], sr1[:, 1], s=10, zorder=10**2, marker='o', color='b')
ax.scatter(no_in[:, 0], no_in[:, 1], s=10, zorder=10**2, marker='o', color='b')

plot_cov_ellipse(sigma1, mu1[0, :], nstd_95, ax=ax, **ellkargs_m)


ax.set_xlim(0, 17)
ax.set_ylim(-1, 15)


for i, x in enumerate(sr1):

    plot_cov_ellipse(sigma1, x, nstd_10, ax=ax, **ellkargs)

ax.text(10, 10, 'class region $(d^2 \\leq d^2_{class})$', fontdict={'fontsize': 10})
ax.text(10, 12, 'point region  $(d^2 \\leq d^2_{point})$', fontdict={'fontsize': 10})

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')

# %% Região de composição (exemplos variando \gamma_{point})

from scipy.spatial import distance
import numpy as np

np.random.seed(10000)

mu1 = np.array([[10, 5]])
sigma1 = np.array([[3, 0.7], [0.7, 2]])
sr1 = generate_gaussian(mu1, sigma1, sample_num=50)

nstd_95 = np.sqrt(chi2.ppf(0.99, 2))
nstd_90 = np.sqrt(chi2.ppf(0.9, 2))
nstd_10 = np.sqrt(chi2.ppf(0.05, 2))

ns_ = [0.05, 0.2]
nstd = np.sqrt(chi2.ppf(ns_, 2))

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
plt.subplots_adjust(wspace=0.05, hspace=0.2, left=0.1, right=0.95, top=0.95, bottom=0.1)

dist = distance.cdist(sr1, mu1, metric='mahalanobis', VI=np.linalg.pinv(sigma1))

idxs = dist > (nstd_95)

no_in = sr1[idxs[:, 0]]

sr1 = sr1[np.invert(idxs[:, 0])]

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:cyan']
markers = ['o', 's', '8', 'D', '*', 'P']
t = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

ellkargs = {'edgecolor': 'purple', 'linewidth': 0.5,
                'fill': True, 'linestyle': '-', 'facecolor': 'c', 'alpha': 0.4}

ellkargs_m = {'edgecolor': 'b', 'linewidth': 0.5, 'fill': True, 'linestyle': '-',
              'facecolor': 'tab:blue', 'alpha': 0.2}


def get_representative(mat, ns):

    idx = np.random.randint(0, mat.shape[0])

    src = mat.copy()

    c = src[idx]

    src = np.delete(src, idx, axis=0)

    p = []

    while True:

        p.append(c)

        d = distance.cdist(src, c[None, :], metric='mahalanobis', VI=np.linalg.pinv(sigma1))

        src = src[(d > ns)[:, 0], :]

        if src.shape[0] == 0:
            break

        else:

            if src.shape[0] == 1:
                p.append(src[0])
                break

            idx = np.random.randint(0, src.shape[0])
            c = src[idx]

            src = np.delete(src, idx, axis=0)

    return np.array(p)


for i, ns in enumerate(nstd):

    ax[i, 0].scatter(sr1[:, 0], sr1[:, 1], s=10, zorder=10**2, marker='o', color='b')
    ax[i, 0].scatter(no_in[:, 0], no_in[:, 1], s=10, zorder=10**2, marker='o', color='b')

    plot_cov_ellipse(sigma1, mu1[0, :], nstd_95, ax=ax[i, 0], **ellkargs_m)
    plot_cov_ellipse(sigma1, mu1[0, :], nstd_95, ax=ax[i, 1], **ellkargs_m)

    ax[i, 0].set_xlim(0, 19)
    ax[i, 0].set_ylim(-1, 15)

    ax[i, 1].set_xlim(0, 19)
    ax[i, 1].set_ylim(-1, 15)

    ax[i, 0].spines['top'].set_color('white')
    ax[i, 0].spines['right'].set_color('white')
    ax[i, 1].spines['top'].set_color('white')
    ax[i, 1].spines['right'].set_color('white')

    fig.text(10,10, '$\gamma_{point} = %.2f$' % ns_[i], zorder=10**100, fontdict={'fontsize': 10})
    ax[i, 1].set_title('$\gamma_{point} = %.2f$' % ns_[i], fontdict={'fontsize': 10})

    ax[0, 0].set_ylabel('$x_2$')
    ax[1, 0].set_ylabel('$x_2$')
    ax[1, 0].set_xlabel('$x_1$')
    ax[1, 1].set_xlabel('$x_1$')

    for j, x in enumerate(sr1):
        plot_cov_ellipse(sigma1, x, ns, ax=ax[i, 0], **ellkargs)

    src = get_representative(sr1, ns)
    ax[i, 1].scatter(src[:, 0], src[:, 1], s=10, zorder=10**2, marker='o', color='b')

    for j, x in enumerate(src):
        plot_cov_ellipse(sigma1, x, ns, ax=ax[i, 1], **ellkargs)


# %%

alphas = lambda d, t: (-np.log(t) / (d))

t = 0.1

d95 = chi2.ppf(.95, 2)
d99 = chi2.ppf(.99, 2)

alpha95 = alphas(d95, t)
alpha99 = alphas(d99, t)

x = np.linspace(0, max(d95, d99) + 2)

y95 = np.exp(-alpha95 * x)
y99 = np.exp(-alpha99 * x)

# plt.plot(x, y95, lw=1.5)
plt.plot(x, y99, lw=1.5)

plt.yticks(np.arange(0, 1.1, 0.1))
# plt.xticks((d99, ))

plt.grid(linestyle=':')

# plt.scatter(d95, t, s=6 ** 2, zorder=1000)
plt.scatter(d99, t, s=6 ** 2, zorder=1000)

plt.text(d99 - 0.35, t + 0.03, '$s_{lk}(d^2_{max})$', fontdict={'fontsize': 10, 'fontweight': 'normal'})
plt.ylabel('$s_{lk}(d^2)$')
plt.xlabel('statistic distance $(d^2)$')


# %% Exibe clusters (2d) em `alg`.

import numpy as np
import sklearn
import math
from scipy.stats import chi2
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from random import *
from matplotlib.patches import Ellipse
from numpy.linalg import cholesky

import pandas as pd

import matplotlib as mpl
import seaborn as snss



# https://github.com/joferkington/oost_paper_code/blob/master/error_ellipse.py
def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).
    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * np.sqrt(nstd * vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


np.random.seed(9)

ellkargs = {'edgecolor': 'b', 'linewidth': 0.5, 'fill': True, 'linestyle': '-', 'facecolor': 'tab:blue', 'alpha': 0.2}

fig, ax = plt.subplots()
# ax.set_xlim(-10, 170)
# ax.set_ylim(-10, 180)

data = pd.read_csv('data/ruspini.csv', sep=';')
# data = (data - data.min()) / (data.max() - data.min())
# data = pd.read_csv('data/entrada.csv', sep=';').iloc[:, [37, 38]]

IdCurto = np.array(output)

idx  = np.random.choice(np.arange(0, 148), 148, replace=False)
colors = np.array(list(mpl.colors.cnames.keys()))[idx.tolist()]

nstd = np.sqrt(chi2.ppf(CONF_MAT_D, 2))


for i, cluster in enumerate(esbm.manager.values()):


    mu = cluster.mu
    sigma = np.linalg.pinv(cluster.inv_sigma)
    D = cluster.D

    plot_cov_ellipse(sigma, mu[0, :], nstd, ax=ax, **ellkargs)

    ax.scatter(data.iloc[IdCurto == i + 1, 1], data.iloc[IdCurto == i + 1, 2], s=12,
                c=colors[i+1], marker='o', edgecolors='k', lw=0.5 )


    ax.scatter(D[:, 0], D[:, 1], s=17, zorder=20**2, marker='s', c=colors[i + 1],  edgecolors='k', lw=0.2)
    ax.scatter(mu[0, 0], mu[0, 1], color='tab:red', s=17, zorder=10**2, marker='s')

# %% Execução eSBM


import pandas as pd
import numpy as np

np.random.seed(10)

data = pd.read_csv('data/ruspini.csv', sep=';')
# data = pd.read_csv('data/entrada.csv', sep=';')

# data = (data - data.min()) / (data.max() - data.min())

# data_norm.iloc[:, 1:39].plot(legend=None)

# ----------------------

CONF_CLUSTER = 0.7
CONF_ADD_POINT = 0.1
CONF_RESIDUE = 0.44
CONF_MAT_D = 0.6
W = 10

config = {
            'data_path': 'data',
            'sep':  ';',
            'FROM_SCRATCH': 'TRUE',
            'W': W,
            'GAMMA_STATS': CONF_CLUSTER,
            'GAMMA_POINT': CONF_ADD_POINT,
            'GAMMA_CLASS' : CONF_MAT_D,
            'PR_ERROR': CONF_RESIDUE,
            'TAU': 0.0001
        }

from algorithms.esbm import eSBM
from algorithms.ctools.log import Log

from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt

log = Log('{}/{}.log'.format(config['data_path'], 'run'), 'run', True, True)

esbm = eSBM(log, config)

output = []
q = []

for row in data.iterrows():

    row = row[1]

    timestamp = row['Timestamp']
    x = row.values[1:3].astype(float)

    info, quality = esbm.process_input(x[None, :], timestamp)

    output.append(info['IdCurto'])
    q.append(quality)


# ----------------------------
# %% Animação base 2d
import matplotlib as mpl
from matplotlib.patches import Ellipse
import pandas as pd
import numpy as np
from scipy.stats import chi2
from scipy import stats

np.random.seed(1011)

sigma = np.array([[10, 7], [7, 15]])
mu = np.array([10, 10])

g1 = stats.multivariate_normal.rvs(mean=mu, cov=sigma, size=35)
g2 = stats.multivariate_normal.rvs(mean=mu * 5.5, cov=sigma * -1.2, size=35)
g3 = stats.multivariate_normal.rvs(mean=mu * 2.5, cov=sigma * -0.8, size=35)


data = pd.read_csv('data/ruspini.csv', sep=';')

# data = pd.DataFrame(columns=['Timestamp', 'x', 'y'])
# d = np.vstack((g1, g2, g3))

# data.x = d[:, 0]
# data.y = d[:, 1]

# data = pd.read_csv('data/entrada.csv', sep=';')

# data = (data - data.min()) / (data.max() - data.min())

# data_norm.iloc[:, 1:39].plot(legend=None)

# ----------------------

CONF_CLUSTER = 0.999
CONF_ADD_POINT = 0.2
CONF_RESIDUE = 0.75
CONF_MAT_D = 0.999
W = 10

config = {
            'data_path': 'data',
            'sep':  ';',
            'FROM_SCRATCH': 'true',
            'W': W,
            'GAMMA_STATS': CONF_CLUSTER,
            'GAMMA_POINT': CONF_ADD_POINT,
            'GAMMA_CLASS' : CONF_MAT_D,
            'GAMMA_RESIDUE': CONF_RESIDUE,
            'TAU': 1e-5
        }

from algorithms.esbm import eSBM
from algorithms.ctools.log import Log

from matplotlib.animation import FuncAnimation
from matplotlib import animation
from matplotlib import pyplot as plt

log = Log('{}/{}.log'.format(config['data_path'], 'run'), 'run', True, True)

esbm = eSBM(log, config)

idx  = np.random.choice(np.arange(0, 148), 148, replace=False)
colors = np.array(list(mpl.colors.cnames.keys()))[idx.tolist()]
unknow_color = 'k'

markers = mpl.markers.MarkerStyle.filled_markers

fig= plt.figure()
ax = plt.subplot(111)

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')

ax.set_xlim(data.x.min() - 20, data.x.max() + 40)
ax.set_ylim(data.y.min() - 20, data.y.max() + 30)

# ax.set_xlim(data.x.min() - 0.1, data.x.max() + 0.1)
# ax.set_ylim(data.y.min() - 0.1, data.y.max() + 0.1)

fig.add_subplot(ax)

pargs = {'s': 11, 'color': unknow_color, 'marker': 'o', 'edgecolors': 'k', 'lw' : 0.5}
dargs = {'s': 14, 'marker': 's', 'edgecolors': 'k', 'lw' : 0.5}
muargs = {'s': 11 ** 2, 'marker': '+', 'color': 'b', 'edgecolors': 'b', 'lw' : 1.2}

ellkargs_c = {'edgecolor': 'b', 'linewidth': 0.6, 'fill': False, 'linestyle': '--', 'facecolor': 'tab:blue', 'alpha': 1}
ellkargs_s = {'edgecolor': 'r', 'linewidth': 1, 'fill': False, 'linestyle': 'dotted', 'facecolor': 'tab:blue', 'alpha': 1}

clusters = list(esbm.manager.values())

nstd_class = np.sqrt(chi2.ppf(CONF_MAT_D, 2))
nstd_stats = np.sqrt(chi2.ppf(CONF_RESIDUE, 2))


points = []
unknow_points = []
frames = []
ecclip1 = []
ecclip2 = []
mul = []

out =[]

ests_p = []
ests_a = []
cp = None
ell_conf = None

def func(frame):

    global frames
    global points
    global unknow_points
    global mul
    global ecclip1, ecclip2
    global ests_p, ests_a
    global cp, ell_conf, out

    print(frame)
    try:

        row = data.iloc[frame, :]

        timestamp = row['Timestamp']
        x = row.values[1:3].astype(float)

        info, quality = esbm.process_input(x[None, :], timestamp)

        id_ = info['IdCurto']

        ax.set_title('$t=%d$' % (frame + 1))

        ests, c_max= esbm.ests

        p = data.iloc[frame, 1:3]

        [p.remove() for p in ests_p]
        [p.remove() for p in ests_a]
        [p.remove() for p in ecclip1]

        ests_p = []
        ests_a = []
        ecclip1 = []

        if cp is not None:
            cp.remove()

        cp = ax.scatter(p.x, p.y, s=30, color='r', marker='8')

        for c in esbm.manager.values():
            sigma = np.linalg.pinv(c.inv_sigma)
            ecclip1.append(plot_cov_ellipse(sigma, c.mu[0, :], nstd_class, ax, **ellkargs_c))

        for e in ests:

            cl = esbm.manager[e[0]]

            sigma = np.linalg.pinv(cl.inv_sigma)

            ests_a.append(plot_cov_ellipse(sigma, e[1][-1][0, :], nstd_stats, ax, **ellkargs_s))
            ests_p.append(ax.scatter(e[1][-1][0, 0], e[1][-1][0, 1], zorder=100**5, lw=0.5, edgecolor='k',
                                     s=60, marker='*', color=colors[cl.info['IdCurto'] - 1]))

        if quality == 64:
            unknow_points.append(ax.scatter(p.x, p.y, **pargs))
            frames.append(frame)

        else:

                c = esbm.manager[info['IdLongo']]

                [p.remove() for p in unknow_points]

                # [u.remove() for u in points]
                [g.remove() for g in mul]

                unknow_points = []
                points = []
                mul = []

                ax.scatter(data.iloc[frames, 1], data.iloc[frames, 2], color=colors[id_ - 1], s=9, marker='o', lw=0.5, edgecolor='k')
                ax.scatter(p.x, p.y, color=colors[id_ - 1], s=11, marker='o', lw=0.5, edgecolor='k')

                points.append(ax.scatter(c.D[:, 0], c.D[:, 1], color=colors[id_ - 1], **dargs))
                mul = [ax.scatter(c.mu[0, 0], c.mu[0, 1], **muargs) for c in esbm.manager.values()]

                frames = []


        # if frame + 1 == data.shape[0]:

        #     for e in esbm.manager.values():

        #         ax.text(10 + e.info['IdCurto'] ** 3, 10, '$\hat{x}_{t, %d}$' % e.info['IdCurto'], fontdict={'fontsize': 9, 'fontweight': 'normal'})

    except Exception as e:
            print(e)

    # plt.savefig('data/figs/fig_%d.svg' % frame)

f = np.arange(0, data.shape[0])

Writer = animation.writers['ffmpeg']
writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)

funcanim = FuncAnimation(fig, func, init_func=lambda : False,  frames=f, interval=30,
                         repeat=False, cache_frame_data=False, save_count=1)

funcanim.save('esbm_ruspini.mp4', writer)
plt.show()

# %% Animação 2d eSBM Plus

import numpy as np
import matplotlib as mpl

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from matplotlib.patches import Ellipse

import pandas as pd

from scipy.stats import chi2
from scipy import stats

from fdiidf.helpers import data as data_m
from fdiidf.evolving.clustering import esbm as m_esbm

np.random.seed(1011)

## Build data

seed = 98
np.random.seed(seed)

mu_1, mu_2 = 1, 30
sigma_1, sigma_2 = 2, 0.5

num_samples = 3000

changes = {'incip': [
                        {'add': 50, 'where': (1000, 1300)},
                        {'add': 0, 'where': (1300, 1450)},
                        {'add': -50, 'where': (1450, 1500)}
                ],
            'sudden': [
                        {'add': -25, 'where': (2000, 2200)}
                ]
            }

x_1, x_2 = data_m.build_2d_gauss_data(mu_1, mu_2, sigma_1, sigma_2,
                                samples=num_samples, changes=changes, w=50, lags=0)

data = np.zeros(shape=(num_samples, 2), dtype=float)
data[:, 0] = x_1
data[:, 1] = x_2

# %%

config = {
    'dim': 2,
    'k': 10,
    'gamma_class': 0.99,
    'gamma_point': 0.2,
    'gamma_res': 0.5,
    'tau': 0.00001

}

esbm = m_esbm.eSBMPlus(**config)
output = []

for x in data:

    x = x.reshape(1, 2)

    q, cstar = esbm.process_input(x)

    output.append(cstar)

fig, axes = plt.subplots(2, 1, sharex=True)

axes[0].plot(data)
axes[1].plot(output)

# %%

from fdiidf.evolving.clustering import esbm
from fdiidf.diagnosis import contribution

esbm_ins = esbm.eSBMPlus
esbm_config = {
    'dim': 2,
    'k': 10,
    'gamma_class': 0.99,
    'gamma_point': 0.2,
    'gamma_res': 0.5,
    'tau': 0.00001

}

grbc_ins = contribution.GRBC
grbc_config = {
    'w_train': 33,
    'w_diag': 5,
    'n_pc': 0.95,
    'conf_level': 0.99,
    'index': 'combined',
    'lags': 0
}


## Monitoring

from fdiidf.monitoring import Monitor

monitor = Monitor(clustering_method=(esbm_ins, esbm_config),
                  diagnosis_method=(grbc_ins, grbc_config))

output = []

for i, x in enumerate(data):

    x = x.reshape(1, 2)
    print('Sample: ', i + 1)

    with np.errstate(all='raise'):
        res = monitor.process_input(x)

    output.append(res)

# %%


## Buil animation

config = {
    'dim': 2,
    'k': 10,
    'gamma_class': 0.999,
    'gamma_point': 0.2,
    'gamma_res': 0.5,
    'tau': 0.00001

}

esbm = m_esbm.eSBMPlus(**config)

idx  = np.random.choice(np.arange(0, 148), 148, replace=False)
colors = np.array(list(mpl.colors.cnames.keys()))[idx.tolist()]
unknow_color = 'k'

markers = mpl.markers.MarkerStyle.filled_markers

fig= plt.figure()
ax = plt.subplot(111)

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')

# ax.set_xlim(data.x.min() - 20, data.x.max() + 40)
# ax.set_ylim(data.y.min() - 20, data.y.max() + 30)

x_max = np.max(data[:, 0])
x_min = np.min(data[:, 0])

y_max = np.max(data[:, 1])
y_min = np.min(data[:, 1])

ax.set_xlim(x_min - 5, x_max + 5)
ax.set_ylim(y_min - 5, y_max + 5)

fig.add_subplot(ax)

pargs = {'s': 11, 'color': unknow_color, 'marker': 'o', 'edgecolors': 'k', 'lw' : 0.5}
dargs = {'s': 14, 'marker': 's', 'edgecolors': 'k', 'lw' : 0.5}
muargs = {'s': 11 ** 2, 'marker': '+', 'color': 'b', 'edgecolors': 'b', 'lw' : 1.2}

ellkargs_c = {'edgecolor': 'b', 'linewidth': 0.6, 'fill': False, 'linestyle': '--', 'facecolor': 'tab:blue', 'alpha': 1}
ellkargs_s = {'edgecolor': 'r', 'linewidth': 1, 'fill': False, 'linestyle': 'dotted', 'facecolor': 'tab:blue', 'alpha': 1}

nstd_class = chi2.ppf(config['gamma_class'], 2)
nstd_stats = chi2.ppf(config['gamma_res'], 2)


points = []
unknow_points = []
frames = []
ecclip1 = []
ecclip2 = []
mul = []

out =[]

ests_p = []
ests_a = []
cp = None
ell_conf = None

def func(frame):

    global frames
    global points
    global unknow_points
    global mul
    global ecclip1, ecclip2
    global ests_p, ests_a
    global cp, ell_conf, out

    print(frame)
    try:

        x = data[frame].reshape(1, 2)

        quality, cstar = esbm.process_input(x)

        id_ =  cstar

        ax.set_title('$t=%d, c=%d$' % (frame + 1, len(esbm.clusters)))

        ests = esbm.estimations

        [p.remove() for p in ests_p]
        [p.remove() for p in ests_a]
        [p.remove() for p in ecclip1]

        ests_p = []
        ests_a = []
        ecclip1 = []

        if cp is not None:
            cp.remove()

        cp = ax.scatter(x[0, 0], x[0, 1], s=30, color='r', marker='8')

        for c_id, xhat in esbm.estimations:

            # print(x, '\n', xhat)
            c = esbm.clusters[c_id]

            sigma = np.linalg.pinv(c.inv_cov)
            ecclip1.append(plot_cov_ellipse(sigma, c.mean[0, :], nstd_class, ax, **ellkargs_c))

            ests_a.append(plot_cov_ellipse(sigma, xhat[0], nstd_stats, ax, **ellkargs_s))
            ests_p.append(ax.scatter(xhat[0, 0], xhat[0, 1], zorder=100**5, lw=0.5, edgecolor='k',
                                     s=60, marker='*', color=colors[cstar - 1]))

        if not quality:
            unknow_points.append(ax.scatter(x[0, 0], x[0, 1], **pargs))
            frames.append(frame)

        else:
                if cstar in esbm.clusters:
                    c = esbm.clusters[cstar]

                    [p.remove() for p in unknow_points]

                    # [u.remove() for u in points]
                    [g.remove() for g in mul]

                    unknow_points = []
                    points = []
                    mul = []

                    ax.scatter(data[frames, 0], data[frames, 1], color=colors[id_ - 1], s=9, marker='o', lw=0.5,
                               edgecolor='k')
                    ax.scatter(x[0, 0], x[0, 1], color=colors[id_ - 1], s=11, marker='o', lw=0.5, edgecolor='k')

                    points.append(ax.scatter(c.D[:, 0], c.D[:, 1], color=colors[id_ - 1], **dargs))
                    mul = [ax.scatter(c.mean[0, 0], c.mean[0, 1], **muargs) for c in esbm.clusters.values()]

                    frames = []


        # if frame + 1 == data.shape[0]:

        #     for e in esbm.manager.values():

        #         ax.text(10 + e.info['IdCurto'] ** 3, 10, '$\hat{x}_{t, %d}$' % e.info['IdCurto'], fontdict={'fontsize': 9, 'fontweight': 'normal'})

    except Exception as e:

        import traceback
        print(traceback.format_exc())

    # plt.savefig('data/figs/fig_%d.svg' % frame)

f = np.arange(0, data.shape[0])

# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)

funcanim = FuncAnimation(fig, func, init_func=lambda : False,  frames=f, interval=200,
                         repeat=False, cache_frame_data=False, save_count=1)

# funcanim.save('esbm_ruspini.mp4', writer)
plt.show()



# %%

import pandas as pd
import numpy as np

tags = ["CP_30100C_T75_PT_3427=NUM,NORM_MAX_MIN,0.0,250.0",
"CP_30100C_T75_PDIT_3793=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PDIT_3769=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PDIT_3765=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PDIT_3768=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PDIT_3777=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PDIT_3776=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PIT_3804=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PIT_3803=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PIT_3778=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PIT_3779=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PIT_3767=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PDIT_3802=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PDIT_3801=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_VXI_3702=NUM,NORM_MAX_MIN,0.0,200.0",
"CP_30100C_T75_VXI_3704=NUM,NORM_MAX_MIN,0.0,200.0",
"CP_30100C_T75_VXI_3719=NUM,NORM_MAX_MIN,0.0,200.0",
"CP_30100C_T75_VXI_3717=NUM,NORM_MAX_MIN,0.0,200.0",
"CP_30100C_T75_VYI_3702=NUM,NORM_MAX_MIN,0.0,200.0",
"CP_30100C_T75_VYI_3704=NUM,NORM_MAX_MIN,0.0,200.0",
"CP_30100C_T75_VYI_3717=NUM,NORM_MAX_MIN,0.0,200.0",
"CP_30100C_T75_VYI_3719=NUM,NORM_MAX_MIN,0.0,200.0",
"CP_30100C_T75_VZI_3700A=NUM,NORM_MAX_MIN,-1.0,2.0",
"CP_30100C_T75_VZI_3700B=NUM,NORM_MAX_MIN,-1.0,2.0",
"CP_30100C_T75_VZI_3715A=NUM,NORM_MAX_MIN,-1.0,2.0",
"CP_30100C_T75_VZI_3715B=NUM,NORM_MAX_MIN,-1.0,2.0",
"CP_30100C_T75_FIT_3796=NUM,NORM_MAX_MIN,0.0,250.0",
"CP_30100C_T75_FIT_3795=NUM,NORM_MAX_MIN,0.0,250.0",
"CP_30100C_T75_PT_3324=NUM,NORM_MAX_MIN,0.0,100.0",
"CP_30100C_T75_PT_3327=NUM,NORM_MAX_MIN,0.0,250.0",
"CP_30100C_T75_PT_3227=NUM,NORM_MAX_MIN,0.0,100.0",
"CP_30100C_T75_PT_3224=NUM,NORM_MAX_MIN,0.0,100.0",
"CP_30100C_T75_PT_3127=NUM,NORM_MAX_MIN,0.0,100.0",
"CP_30100C_T75_TT_3329=NUM,NORM_MAX_MIN,0.0,200.0", # Este 100 -> 200 (max)
"CP_30100C_T75_TT_3321=NUM,NORM_MAX_MIN,0.0,100.0",
"CP_30100C_T75_TT_3121=NUM,NORM_MAX_MIN,0.0,100.0",
"CP_30100C_T75_TT_3129=NUM,NORM_MAX_MIN,0.0,200.0",
"CP_30100C_T75_TT_3221=NUM,NORM_MAX_MIN,0.0,100.0"]

tags_dict = {tag.split('=')[0] : (float(tag.split(',')[-2]), float(tag.split(',')[-1])) for tag in tags}


data = pd.read_csv('data/entrada.csv', sep=';')

data_norm = pd.DataFrame(columns=data.columns)

for tag, (min_, max_) in tags_dict.items():

    values  = data[tag]

    data_norm[tag] = (values - min_) / (max_ - min_)

# data = pd.read_csv('data/entrada_all.csv', sep=';')

values = data.iloc[:, 1:39]

# data_norm = data.copy()
# data_norm.iloc[:, 1:39] = (values - values.min()) / (values.max() - values.min())


# data_norm.iloc[:, 1:39].plot(legend=None)

# ----------------------
CONF_CLUSTER = 0.99
CONF_ADD_POINT = 0.1
CONF_RESIDUE = 0.4
CONF_MAT_D = 0.99
W = 40

config = {
            'data_path': 'data',
            'sep':  ';',
            'FROM_SCRATCH': 'false',
            'W': W,
            'GAMMA_STATS': CONF_CLUSTER,
            'GAMMA_POINT': CONF_ADD_POINT,
            'GAMMA_CLASS' : CONF_MAT_D,
            'GAMMA_RESIDUE': CONF_RESIDUE,
            'TAU': 0.00001
        }

from algorithms.esbm import eSBM
from algorithms.ctools.log import Log

from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt

log = Log('{}/{}.log'.format(config['data_path'], 'run'), 'run', True, True)

esbm = eSBM(log, config)

# ----------------------




fig, ax = plt.subplots(3, 1, sharex=True)

ax[0].plot(data_norm.iloc[:, 1:39])
vline = ax[0].axvline(0, data_norm.iloc[:, 1:39].min().min(), data.iloc[:, 1:39].max().max(), lw=1.5, color='c')

# line_var = ax[0].plot([], color='g')[0]
line_idcurto = ax[1].plot([], color='b')[0]
line_quality = ax[2].plot([], color='k')[0]



ax[2].set_ylim(60, 200)

vars_v = []
idcurto_v = []
quality_v = []

idx_var = 0
x = []

w = 1000

# ax[0].plot(data_norm.iloc[:, 1:39].iloc[:w, idx_var])

# ax[0].set_title('Sinal')
# ax[1].set_title('IdCurto')
# ax[2].set_title('Qualidade')

def func(frame):

    row = data.iloc[frame, :]

    timestamp = row['Timestamp']
    point = row.values[1:39].astype(float)

    info, quality = esbm.process_input(point[None, :], timestamp)

    # print(quality)

    x.append(frame)
    vars_v.append(point[idx_var])
    idcurto_v.append(info['IdCurto'])
    quality_v.append(quality)

    # line_var.set_data(x, np.array(vars_v))
    line_idcurto.set_data(x, np.array(idcurto_v))
    line_quality.set_data(x, np.array(quality_v))


    x_min = 0

    if frame > w:
        x_min = 0 if frame - w < 0 else frame - w


    vline.set_xdata(frame)

    [a.set_xlim(x_min, frame + w) for a in ax]
    # axv.set_xlim(x_min, frame + w)

    ax[1].set_ylim(0, max(idcurto_v) + 1)
    # axv.set_title(frame)

    return line_quality, line_idcurto, vline


funcanim = FuncAnimation(fig, func, frames=list(range(0, data_norm.shape[0])), interval=5)

plt.tight_layout(pad=0.5, h_pad=0, w_pad=0.1)
plt.show()


# %%


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


tags = ["CP_30100C_T75_PT_3427=NUM,NORM_MAX_MIN,0.0,250.0",
"CP_30100C_T75_PDIT_3793=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PDIT_3769=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PDIT_3765=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PDIT_3768=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PDIT_3777=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PDIT_3776=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PIT_3804=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PIT_3803=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PIT_3778=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PIT_3779=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PIT_3767=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PDIT_3802=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PDIT_3801=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_VXI_3702=NUM,NORM_MAX_MIN,0.0,200.0",
"CP_30100C_T75_VXI_3704=NUM,NORM_MAX_MIN,0.0,200.0",
"CP_30100C_T75_VXI_3719=NUM,NORM_MAX_MIN,0.0,200.0",
"CP_30100C_T75_VXI_3717=NUM,NORM_MAX_MIN,0.0,200.0",
"CP_30100C_T75_VYI_3702=NUM,NORM_MAX_MIN,0.0,200.0",
"CP_30100C_T75_VYI_3704=NUM,NORM_MAX_MIN,0.0,200.0",
"CP_30100C_T75_VYI_3717=NUM,NORM_MAX_MIN,0.0,200.0",
"CP_30100C_T75_VYI_3719=NUM,NORM_MAX_MIN,0.0,200.0",
"CP_30100C_T75_VZI_3700A=NUM,NORM_MAX_MIN,-1.0,2.0",
"CP_30100C_T75_VZI_3700B=NUM,NORM_MAX_MIN,-1.0,2.0",
"CP_30100C_T75_VZI_3715A=NUM,NORM_MAX_MIN,-1.0,2.0",
"CP_30100C_T75_VZI_3715B=NUM,NORM_MAX_MIN,-1.0,2.0",
"CP_30100C_T75_FIT_3796=NUM,NORM_MAX_MIN,0.0,250.0",
"CP_30100C_T75_FIT_3795=NUM,NORM_MAX_MIN,0.0,250.0",
"CP_30100C_T75_PT_3324=NUM,NORM_MAX_MIN,0.0,100.0",
"CP_30100C_T75_PT_3327=NUM,NORM_MAX_MIN,0.0,250.0",
"CP_30100C_T75_PT_3227=NUM,NORM_MAX_MIN,0.0,100.0",
"CP_30100C_T75_PT_3224=NUM,NORM_MAX_MIN,0.0,100.0",
"CP_30100C_T75_PT_3127=NUM,NORM_MAX_MIN,0.0,100.0",
"CP_30100C_T75_TT_3329=NUM,NORM_MAX_MIN,0.0,200.0", # Este 100 -> 200 (max)
"CP_30100C_T75_TT_3321=NUM,NORM_MAX_MIN,0.0,100.0",
"CP_30100C_T75_TT_3121=NUM,NORM_MAX_MIN,0.0,100.0",
"CP_30100C_T75_TT_3129=NUM,NORM_MAX_MIN,0.0,200.0",
"CP_30100C_T75_TT_3221=NUM,NORM_MAX_MIN,0.0,100.0"]

tags_dict = {tag.split('=')[0] : (float(tag.split(',')[-2]), float(tag.split(',')[-1])) for tag in tags}


data = pd.read_csv('data/entrada.csv', sep=';')

data_norm = pd.DataFrame(columns=data.columns)

for tag, (min_, max_) in tags_dict.items():

    values  = data[tag]

    data_norm[tag] = (values - min_) / (max_ - min_)

v0 = 0.1
v1 = 1
v2 = 20
v3 = 85

data = data.iloc[:, 1:39]
data_norm = data_norm.iloc[:, 1:39]

fig = plt.figure()


ax4 = plt.subplot(515,)
ax3 = plt.subplot(514, sharex=ax4)
ax2 = plt.subplot(513, sharex=ax3)
ax1 = plt.subplot(512, sharex=ax2)
ax0 = plt.subplot(511, sharex=ax1)



idx0 = np.nonzero(data.mean() <= v0)[0]

ax0.plot(data_norm.iloc[:, idx0])

idx1 = np.nonzero(data.mean() > v0)[0]
idx1 = np.nonzero(data.iloc[:, idx1].mean() <= v1)[0]

ax1.plot(data_norm.iloc[:, idx1])

idx2 = np.nonzero(data.mean() > v1)[0]
idx2 = np.nonzero(data.iloc[:, idx2].mean() <= v2)[0]

ax2.plot(data_norm.iloc[:, idx2])

idx3 = np.nonzero(data.mean() > v2)[0]
idx3 = np.nonzero(data.iloc[:, idx3].mean() <= v3)[0]
ax3.plot(data_norm.iloc[:, idx3])

ax4.plot(data_norm.iloc[:, np.nonzero(data.mean() > v3)[0]])

fig.add_subplot(ax0)
fig.add_subplot(ax1)
fig.add_subplot(ax2)
fig.add_subplot(ax3)
fig.add_subplot(ax4)

tickslabels = ax0.get_xticklabels()

ax0.set_xticklabels([])

# ax4.set_xticklabels(tickslabels)

plt.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.01)

plt.show()


# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


tags = ["CP_30100C_T75_PT_3427=NUM,NORM_MAX_MIN,0.0,250.0",
"CP_30100C_T75_PDIT_3793=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PDIT_3769=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PDIT_3765=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PDIT_3768=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PDIT_3777=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PDIT_3776=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PIT_3804=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PIT_3803=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PIT_3778=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PIT_3779=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PIT_3767=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PDIT_3802=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_PDIT_3801=NUM,NORM_MAX_MIN,0.0,3.0",
"CP_30100C_T75_VXI_3702=NUM,NORM_MAX_MIN,0.0,200.0",
"CP_30100C_T75_VXI_3704=NUM,NORM_MAX_MIN,0.0,200.0",
"CP_30100C_T75_VXI_3719=NUM,NORM_MAX_MIN,0.0,200.0",
"CP_30100C_T75_VXI_3717=NUM,NORM_MAX_MIN,0.0,200.0",
"CP_30100C_T75_VYI_3702=NUM,NORM_MAX_MIN,0.0,200.0",
"CP_30100C_T75_VYI_3704=NUM,NORM_MAX_MIN,0.0,200.0",
"CP_30100C_T75_VYI_3717=NUM,NORM_MAX_MIN,0.0,200.0",
"CP_30100C_T75_VYI_3719=NUM,NORM_MAX_MIN,0.0,200.0",
"CP_30100C_T75_VZI_3700A=NUM,NORM_MAX_MIN,-1.0,2.0",
"CP_30100C_T75_VZI_3700B=NUM,NORM_MAX_MIN,-1.0,2.0",
"CP_30100C_T75_VZI_3715A=NUM,NORM_MAX_MIN,-1.0,2.0",
"CP_30100C_T75_VZI_3715B=NUM,NORM_MAX_MIN,-1.0,2.0",
"CP_30100C_T75_FIT_3796=NUM,NORM_MAX_MIN,0.0,250.0",
"CP_30100C_T75_FIT_3795=NUM,NORM_MAX_MIN,0.0,250.0",
"CP_30100C_T75_PT_3324=NUM,NORM_MAX_MIN,0.0,100.0",
"CP_30100C_T75_PT_3327=NUM,NORM_MAX_MIN,0.0,250.0",
"CP_30100C_T75_PT_3227=NUM,NORM_MAX_MIN,0.0,100.0",
"CP_30100C_T75_PT_3224=NUM,NORM_MAX_MIN,0.0,100.0",
"CP_30100C_T75_PT_3127=NUM,NORM_MAX_MIN,0.0,100.0",
"CP_30100C_T75_TT_3329=NUM,NORM_MAX_MIN,0.0,200.0", # Este 100 -> 200 (max)
"CP_30100C_T75_TT_3321=NUM,NORM_MAX_MIN,0.0,100.0",
"CP_30100C_T75_TT_3121=NUM,NORM_MAX_MIN,0.0,100.0",
"CP_30100C_T75_TT_3129=NUM,NORM_MAX_MIN,0.0,200.0",
"CP_30100C_T75_TT_3221=NUM,NORM_MAX_MIN,0.0,100.0"]

tags_dict = {tag.split('=')[0] : (float(tag.split(',')[-2]), float(tag.split(',')[-1])) for tag in tags}


labels = np.array(['x_{%d}' % i for i in range(1, 39)])

data = pd.read_csv('data/pr/saida.csv', sep=';')

data_norm = pd.DataFrame(columns=labels)

for i, (tag, (min_, max_)) in enumerate(tags_dict.items()):

    values = data[tag]

    data_norm['x_{%d}' % (i + 1)] = (values - min_) / (max_ - min_)

v0 = 0.1
v1 = 1
v2 = 20
v3 = 85

# data_norm = data_norm.iloc[:, 1:39]

fig, axes = plt.subplots()

import seaborn as sns

np.random.seed(10001)

colors = list(sns.palettes.xkcd_rgb.values())

cm = plt.get_cmap('Accent')

NUM_COLORS = 38
# colors = [cm((i + 0)/NUM_COLORS) for i in range(NUM_COLORS)]

colors_ = [matplotlib.colors.hex2color(c) for c in np.random.choice(colors, NUM_COLORS, replace=False)]

print('UNIQUE:', np.unique(colors_).shape)

axes.set_prop_cycle('color', colors_)

axes.set_ylabel("Signal", fontsize=9)
axes.set_xlabel('Time $(t)$', fontsize=9)
# axes.set_prop_cycle('color', colors)

# plt.ylabel('$aaa$')

# idx0 = np.nonzero(data_norm.mean() <= v0)[0]

# ax0.set_prop_cycle('color', colors[:idx0.shape[0]])


lines = axes.plot(data_norm, lw=0.8)

ticks = np.round(np.arange(0, 1.2, 0.2), decimals=1)
axes.set_yticks(ticks)
axes.set_yticklabels(ticks)
axes.set_ylim(-0.05, 1.05)

p = np.array([490, 1339, 1495, 3191, 4324, 8343])
axes.vlines(p-1, -0.05, 1.05, lw=1.1, linestyle='dotted')
axes.set_xticks(p-1)
axes.set_xticklabels(p, rotation=45, fontdict={'fontsize': 8})
axes.legend(lines, [('$%s$' % l) for l in labels], ncol=9, fontsize=7, framealpha=0.5, columnspacing=0.5, handlelength=0.7, labelspacing=0.25)


plt.tight_layout(pad=0.1, h_pad=0.5, w_pad=0.01)

# %% Resultado todo período
p_created = np.array([40, 525, 1378, 1490, 3169, 3232, 4336, 8382, 9027])
p_detected = np.array(p_created[1:]) - 40

fig = plt.figure(200)
gs = plt.GridSpec(4, 1)

ax1 = fig.add_subplot(gs[:3, 0])
ax2 = fig.add_subplot(gs[3, 0])

import seaborn as sns

np.random.seed(10001)

colors = list(sns.palettes.xkcd_rgb.values())

cm = plt.get_cmap('Accent')

NUM_COLORS = 38
# colors = [cm((i + 0)/NUM_COLORS) for i in range(NUM_COLORS)]

colors_ = [matplotlib.colors.hex2color(c) for c in np.random.choice(colors, NUM_COLORS, replace=False)]

print('UNIQUE:', np.unique(colors_).shape)

ax1.set_prop_cycle('color', colors_)


lines = ax1.plot(data_norm, lw=0.8)
# ax1.vlines(p_detected, -0.04, 0.9, lw=1.1, linestyle='--', color='r')
ax2.plot(data.IdCurto, '.')
ax2.plot(p_detected - 1, data.IdCurto[p_detected-1], linestyle='None', marker=(7, 2, 0), lw=7, color='r')
ax2.plot(p_created - 1, data.IdCurto[p_created-1], linestyle='None', marker=(7, 2, 0), lw=4, color='k')

ax2.set_yticks(np.unique(data.IdCurto))
ax2.set_yticklabels(np.unique(data.IdCurto), fontsize=8)

ax2.set_xticks(p_detected - 1)

ax1.set_xticks(p_detected - 1)
ax1.set_yticklabels(np.round(ax1.get_yticks(), decimals=1), fontsize=8)
ax1.set_ylim(-0.05, 1.05)

ax1.set_xticklabels([])
ax2.set_xticklabels(p_detected, fontsize=8, rotation=45)
ax2.set_ylim(0, data.IdCurto.max() + 1)

ax1.set_ylabel('Signal', fontsize=9)
ax2.set_ylabel('$c$', fontsize=9, rotation=0)
ax2.set_xlabel('Time $(t)$', fontsize=9)

ax1.legend(lines, [('$%s$' % l) for l in labels], ncol=9, fontsize=7, framealpha=0.5, columnspacing=0.5, handlelength=0.7, labelspacing=0.25)


plt.tight_layout(pad=0.1, h_pad=0.5, w_pad=0.01)
plt.show()


# %% Resultado mudança 1

t_min = 480
t_max = 531

p = [490, 1339, 1495, 3191, 4324, 8343]
p_created = [40, 525, 1378, 1490, 3169, 3232, 4336, 8382, 9027]
p_detected = np.array(p_created[1:]) - 40

fig = plt.figure(200)
gs = plt.GridSpec(4, 1)

ax1 = fig.add_subplot(gs[:3, 0])
ax2 = fig.add_subplot(gs[3, 0])

import seaborn as sns

np.random.seed(10001)

colors = list(sns.palettes.xkcd_rgb.values())

cm = plt.get_cmap('Accent')

NUM_COLORS = 38
# colors = [cm((i + 0)/NUM_COLORS) for i in range(NUM_COLORS)]

colors_ = [matplotlib.colors.hex2color(c) for c in np.random.choice(colors, NUM_COLORS, replace=False)]

print('UNIQUE:', np.unique(colors_).shape)

ax1.set_prop_cycle('color', colors_)


lines = ax1.plot(data_norm.iloc[t_min:t_max, :], lw=0.8)
# ax1.vlines(p_detected, -0.04, 0.9, lw=1.1, linestyle='--', color='r')
ax2.plot(data.IdCurto[t_min:t_max], '.')
ax2.plot(p_detected[0] - 1, data.IdCurto[t_min:t_max][p_detected[0]-1], linestyle='None', marker=(7, 2, 0), lw=7, color='r')
ax2.plot(p_created[1] - 1, data.IdCurto[p_created[1]-1], linestyle='None', marker=(7, 2, 0), lw=4, color='k')

ax2.set_yticks(np.unique(data.IdCurto)[:2])
ax2.set_yticklabels(np.unique(data.IdCurto)[:2])

ax2.set_ylim(0, 3)

ax1.set_ylim(-0.05, 1.05)
ax1.set_xticklabels([])


ticks = np.arange(t_min, t_max, 5)
ax2.set_xticks(ticks - 1)
ax1.set_xticks(ticks - 1)
ax2.set_xticklabels(ticks)

ax1.set_ylabel('Signal', fontsize=9)
ax2.set_ylabel('$c$', fontsize=9, rotation=0)
ax2.set_xlabel('Time $(t)$', fontsize=9)
ax1.legend(lines, [('$%s$' % l) for l in labels], ncol=9, fontsize=7, framealpha=0.5, columnspacing=0.5, handlelength=0.7, labelspacing=0.25)


ax1.vlines([p_detected[0] - 1, p_created[1] - 1], -0.05, 1.05, lw=1.1, linestyle='dotted', color='k')

plt.tight_layout(pad=0.1, h_pad=0.5, w_pad=0.01)
plt.show()


# %% Exemplo região residual


from scipy.spatial import distance

import matplotlib


np.random.seed(10000)

mu1, mu2 = np.array([[10, 5]]), np.array([[10, 17]])
sigma1, sigma2 = np.array([[3, 0.7], [0.7, 2]]), np.array([[3, -0.3], [-0.3, 6]])

sr1 = generate_gaussian(mu1, sigma1, sample_num=50)
sr2 = generate_gaussian(mu2, sigma2, sample_num=50)

nstd_80 = np.sqrt(chi2.ppf(0.8, 2))
nstd_99 = np.sqrt(chi2.ppf(0.99, 2))
nstd_999 = np.sqrt(chi2.ppf(0.999, 2))

fig, ax = plt.subplots()

ax.set_xlim(3, 16)
ax.set_ylim(-1, 12)

dist = distance.cdist(sr1, mu1, metric='mahalanobis', VI=np.linalg.pinv(sigma1))

idx = np.argmax(dist)

max_x = sr1[idx]


# ax.scatter(sr1[:, 0], sr1[:, 1], s=10, zorder=10**2, marker='o', color='b')


CONF_CLUSTER = 0.99
CONF_ADD_POINT = 0.2
CONF_RESIDUE = 0.3
CONF_MAT_D = 0.99
W = 40

config = {
            'data_path': 'data',
            'sep':  ';',
            'FROM_SCRATCH': 'TRUE',
            'W': W,
            'GAMMA_STATS': CONF_CLUSTER,
            'GAMMA_POINT': CONF_ADD_POINT,
            'GAMMA_CLASS' : CONF_MAT_D,
            'PR_ERROR': CONF_RESIDUE,
            'TAU': 0.0001
        }

from algorithms.esbm import eSBM
from algorithms.ctools.log import Log

from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt

log = Log('{}/{}.log'.format(config['data_path'], 'run'), 'run', True, True)

esbm = eSBM(log, config)

output = []
q = []

for x in sr1:


    info, quality = esbm.process_input(x[None, :], timestamp=None)

    output.append(info['IdCurto'])
    q.append(quality)



# ax.scatter(sr2[:, 0], sr2[:, 1], s=4, zorder=10**2)

ellkargs = {'edgecolor': 'b', 'linewidth': 0.5, 'fill': True, 'linestyle': '-', 'facecolor': 'tab:blue', 'alpha': 0.2}




esbm.process_input(max_x[None, :])

ests, c_max = esbm.ests
est = ests[0][-1][1]

D = esbm.manager[c_max].D
mu = esbm.manager[c_max].mu
sigma = np.linalg.pinv(esbm.manager[c_max].inv_sigma)


plot_cov_ellipse(sigma, mu[0, :], np.sqrt(chi2.ppf(CONF_MAT_D, 2)), ax=ax, **ellkargs)

levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

colormap = plt.cm.nipy_spectral

colors = [colormap(i) for i in np.linspace(0.2, 1, 10)]

matplotlib.lines.lineStyles

art = []
for i, c in enumerate(levels[::-1]):

    art.append(plot_cov_ellipse(sigma, est[0, :], np.sqrt(chi2.ppf(c, 2)), ax=ax,
                                                                        **{'edgecolor': colors[i], 'linewidth': 1.5,
                                                                          'fill': False, 'linestyle': '-',
                                                                          'facecolor': colors[i],
                                                                         'alpha': 1, 'label': i}))

ax.set_xlim(0, 18)
ax.set_ylim(0, 15)

lg = plt.legend(art[::-1], levels, ncol=2, markerscale=0.25, title='$\gamma_{res}$', **{'fontsize': 9})

ax.text(8, 12, '$\hat{x}_t$', fontsize=9)
ax.text(6, 12, '$x_t$', fontsize=9)
ax.text(3, 12, 'class region', fontsize=9)

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
# [a.set_facecolor('b') for a in lg.get_patches()]
#

# cs = matplotlib.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap(c))
# mappable = matplotlib.contour.ContourLabeler(ax=ax, levels=levels[::-1], cmap=c)

# cb = plt.colorbar(cs, spacing='proportional')
# cb.set_ticks(levels[::-1])
# cb.set_label('  $\quad \gamma_{res}$', rotation=0)

ax.scatter(max_x[0], max_x[1], s=6**2, zorder=10**2, marker='o', color='r')
ax.scatter(mu[0, 0], mu[0, 1], color='tab:red', s=17, zorder=10**2, marker='D')
ax.scatter(D[:, 0], D[:, 1], color='b', marker='s', s=7**2)

ax.scatter(est[0, 0], est[0, 1], color='r', marker='*', s=6**2, zorder=10*100)

#%%

import numpy as np; import pandas as pd; import matplotlib; import matplotlib.pyplot as plt; import seaborn as sns


# m = 3
data = pd.read_csv('data/saida_at_m3.csv', sep=';')


fig = plt.figure(200)
gs = plt.GridSpec(4, 1)

ax1 = fig.add_subplot(gs[:3, 0])
ax2 = fig.add_subplot(gs[3, 0])



data.iloc[:, 1:39].plot(legend=None, ax=ax1, lw=1)
data.IdCurto.plot(ax=ax2, lw=1.2)

ax1.set_xticklabels([])
ax1.set_ylabel('Amplitude')
ax2.set_yticks(data.IdCurto.unique())
ax2.set_ylim(0, data.IdCurto.max() + 1)
ax2.set_ylabel('Índice cluster', fontsize=10)
ax2.set_xlabel('Amostra $(t)$')

plt.tight_layout(h_pad=0.01, w_pad=0.01)
plt.show()

# m=2

data = pd.read_csv('data/saida_at_m2.csv', sep=';')


fig = plt.figure(201)
gs = plt.GridSpec(4, 1)

ax1 = fig.add_subplot(gs[:3, 0])
ax2 = fig.add_subplot(gs[3, 0])


data.iloc[:, 1:39].plot(legend=None, ax=ax1, lw=1)
data.IdCurto.plot(ax=ax2, lw=1.2)

ax1.set_xticklabels([])
ax1.set_ylabel('Amplitude')
ax2.set_yticks(data.IdCurto.unique())
ax2.set_ylim(0, data.IdCurto.max() + 1)
ax2.set_ylabel('Índice cluster', fontsize=10)
ax2.set_xlabel('Amostra $(t)$')

plt.tight_layout(h_pad=0.01, w_pad=0.01)
plt.show()

# m=1.5

data = pd.read_csv('data/saida_at_m1_5.csv', sep=';')


fig = plt.figure(202)
gs = plt.GridSpec(4, 1)

ax1 = fig.add_subplot(gs[:3, 0])
ax2 = fig.add_subplot(gs[3, 0])


data.iloc[:, 1:39].plot(legend=None, ax=ax1, lw=1)
data.IdCurto.plot(ax=ax2, lw=1.2)

ax1.set_xticklabels([])
ax1.set_ylabel('Amplitude')
ax2.set_ylim(0, data.IdCurto.max() + 1)
ax2.set_ylabel('Índice cluster', fontsize=10)
ax2.set_xlabel('Amostra $(t)$')

plt.tight_layout(h_pad=0.01, w_pad=0.01)
plt.show()

# Segunda base de dados m=3

data = pd.read_csv('data/saida_at_b2_m3.csv', sep=';')


fig = plt.figure(2010)
gs = plt.GridSpec(4, 1)

ax1 = fig.add_subplot(gs[:3, 0])
ax2 = fig.add_subplot(gs[3, 0])


data.iloc[:, 1:39].plot(legend=None, ax=ax1, lw=1)
data.IdCurto.plot(ax=ax2, lw=1.2)

ax1.set_xticklabels([])
ax1.set_ylabel('Amplitude')
ax2.set_ylim(0, data.IdCurto.max() + 1)
ax2.set_ylabel('Índice cluster', fontsize=10)
ax2.set_xlabel('Amostra $(t)$')

plt.tight_layout(h_pad=0.01, w_pad=0.01)
plt.show()


# Segunda base de dados m=3, w=10

data = pd.read_csv('data/saida_at_b2_m3_w10.csv', sep=';')


fig = plt.figure(2011)
gs = plt.GridSpec(4, 1)

ax1 = fig.add_subplot(gs[:3, 0])
ax2 = fig.add_subplot(gs[3, 0])


data.iloc[:, 1:39].plot(legend=None, ax=ax1, lw=1)
data.IdCurto.plot(ax=ax2, lw=1.2)

ax1.set_xticklabels([])
ax1.set_ylabel('Amplitude')
ax2.set_ylim(0, data.IdCurto.max() + 1)
ax2.set_ylabel('Índice cluster', fontsize=10)
ax2.set_xlabel('Amostra $(t)$')

plt.tight_layout(h_pad=0.01, w_pad=0.01)
plt.show()


# AutoCloud, eSBM, OEC

data = pd.read_csv('data/saida_at_m2.csv', sep=';')


fig = plt.figure(203)
gs = plt.GridSpec(4, 1)

ax1 = fig.add_subplot(gs[:2, 0])
ax2 = fig.add_subplot(gs[2:, 0])



data.iloc[:, 1].plot(legend=None, ax=ax1, lw=1, label=data.columns[1])
lat= data.IdCurto.plot(ax=ax2, lw=1.5, label='AutoCloud', color='b', ls='-')

ax1.set_xticklabels([])
ax1.set_ylabel('Amplitude')
ax1.legend(fontsize=8, handlelength=1.8)
ax2.set_ylabel('Índice cluster', fontsize=10)
ax2.set_xlabel('Amostra $(t)$')


data = pd.read_csv('data/saida_esbm.csv', sep=';')
le = data.IdCurto.plot(label='eSBM', lw=1.5, color='g', ax=ax2, ls='dashdot')

data = pd.read_csv('data/saida_oec.csv', sep=';')
lo = data.IdCurto.plot(label='OEC', lw=1.5, color='k', ax=ax2, ls='--')

ax2.legend(fontsize=8, handlelength=1.8)

plt.tight_layout(h_pad=0.01, w_pad=0.01)
plt.show()

# w=1.5 w, 10

data = pd.read_csv('data/saida_at_w10.csv', sep=';')


fig = plt.figure(204)
gs = plt.GridSpec(4, 1)

ax1 = fig.add_subplot(gs[:3, 0])
ax2 = fig.add_subplot(gs[3, 0])


data.iloc[:, 1:39].plot(legend=None, ax=ax1, lw=1)
data.IdCurto.plot(ax=ax2, lw=1.2)

ax1.set_xticklabels([])
ax1.set_ylabel('Amplitude')
ax2.set_ylim(0, data.IdCurto.max() + 1)
ax2.set_ylabel('Índice cluster', fontsize=10)
ax2.set_xlabel('Amostra $(t)$')

plt.tight_layout(h_pad=0.01, w_pad=0.01)
plt.show()



# w=1.5 w, 15

data = pd.read_csv('data/saida_at_w15.csv', sep=';')


fig = plt.figure(205)
gs = plt.GridSpec(4, 1)

ax1 = fig.add_subplot(gs[:3, 0])
ax2 = fig.add_subplot(gs[3, 0])


data.iloc[:, 1:39].plot(legend=None, ax=ax1, lw=1)
data.IdCurto.plot(ax=ax2, lw=1.2)

ax1.set_xticklabels([])
ax1.set_ylabel('Amplitude')
ax2.set_ylim(0, data.IdCurto.max() + 1)
ax2.set_ylabel('Índice cluster', fontsize=10)
ax2.set_xlabel('Amostra $(t)$')

plt.tight_layout(h_pad=0.01, w_pad=0.01)
plt.show()
