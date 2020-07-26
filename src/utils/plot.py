import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt


def ellipse_data(semimaj=1, semimin=1, phi=0, x_cent=0, y_cent=0, theta_num=1e3, ax=None, plot_kwargs=None, cov=None, mass_level=0.9):
    # Get Ellipse Properties from cov matrix
    eig_vec, eig_val, u = np.linalg.svd(cov)
    # Make sure 0th eigenvector has positive x-coordinate
    if eig_vec[0][0] < 0:
        eig_vec[0] *= -1
    semimaj = np.sqrt(eig_val[0])
    semimin = np.sqrt(eig_val[1])
    distances = np.linspace(0,20,20001)
    chi2_cdf = chi2.cdf(distances,df=2)
    multiplier = np.sqrt(distances[np.where(np.abs(chi2_cdf-mass_level)==np.abs(chi2_cdf-mass_level).min())[0][0]])
    semimaj *= multiplier
    semimin *= multiplier
    phi = np.arccos(np.dot(eig_vec[0],np.array([1,0])))
    if eig_vec[0][1] < 0 and phi > 0:
        phi *= -1

    # Generate data for ellipse structure
    theta = np.linspace(0, 2*np.pi, theta_num)
    r = 1 / np.sqrt((np.cos(theta))**2 + (np.sin(theta))**2)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    data = np.array([x,y])
    S = np.array([[semimaj, 0], [0, semimin]])
    R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    T = np.dot(R,S)
    data = np.dot(T, data)
    data[0] += x_cent
    data[1] += y_cent

    return data


def plot_clusters(model, x):
    clusters = model.predict(x)
    for k in range(model.k):
        plt.scatter(x[clusters == k][:, 0], x[clusters == k][:, 1], c='C' + str(k), alpha=.5)
        plt.scatter(model.means[k][0], model.means[k][1], color="C" + str(k), marker='X', edgecolor="black", s=300)
    plt.axis('scaled')


def plot_ellipses(model):
    if len(model.covariances.shape) != 3:
        covariances = np.tile(np.identity(model.means.shape[1]), (model.k, 1, 1))
        covariances = covariances * model.covariances[..., np.newaxis, np.newaxis]
    else:
        covariances = model.covariances
    for k in range(model.k):
        ellipse = ellipse_data(x_cent=model.means[k, 0], y_cent=model.means[k, 1], cov=covariances[k, :, :])
        plt.plot(ellipse[0], ellipse[1], c='C' + str(k))
    plt.axis('scaled')


def plot_contours(model, x):
    max_x, max_y = x.max(axis=0)
    min_x, min_y = x.min(axis=0)

    n_pts_by_axes = 100
    xs = np.linspace(min_x, max_x, n_pts_by_axes)
    ys = np.linspace(min_y, max_y, n_pts_by_axes)

    X, Y = np.meshgrid(xs, ys)
    Z = model.predict(np.c_[np.ravel(X), np.ravel(Y)])
    plt.contour(X, Y, Z.reshape(n_pts_by_axes, n_pts_by_axes), colors='black')