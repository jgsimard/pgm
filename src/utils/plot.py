import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import chi2


def ellipse_data(semimaj=1, semimin=1, phi=0, x_cent=0, y_cent=0, theta_num=1e3, ax=None, plot_kwargs=None, cov=None,
                 mass_level=0.9):
    # Get Ellipse Properties from cov matrix
    eig_vec, eig_val, u = np.linalg.svd(cov)
    # Make sure 0th eigenvector has positive x-coordinate
    if eig_vec[0][0] < 0:
        eig_vec[0] *= -1
    semimaj = np.sqrt(eig_val[0])
    semimin = np.sqrt(eig_val[1])
    distances = np.linspace(0, 20, 20001)
    chi2_cdf = chi2.cdf(distances, df=2)
    multiplier = np.sqrt(
        distances[np.where(np.abs(chi2_cdf - mass_level) == np.abs(chi2_cdf - mass_level).min())[0][0]])
    semimaj *= multiplier
    semimin *= multiplier
    phi = np.arccos(np.dot(eig_vec[0], np.array([1, 0])))
    if eig_vec[0][1] < 0 and phi > 0:
        phi *= -1

    # Generate data for ellipse structure
    theta = np.linspace(0, 2 * np.pi, int(theta_num))
    r = 1 / np.sqrt((np.cos(theta)) ** 2 + (np.sin(theta)) ** 2)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    data = np.array([x, y])
    S = np.array([[semimaj, 0], [0, semimin]])
    R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    T = np.dot(R, S)
    data = np.dot(T, data)
    data[0] += x_cent
    data[1] += y_cent

    return data


def plot_info():
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title("EM Gaussian Dataset")
    plt.axis('scaled')


def plot_dataset(x):
    x = x.numpy()
    plt.scatter(x[:, 0], x[:, 1], c='black', alpha=.5)
    plot_info()


def plot_clusters(model, x, means=True):
    clusters = model.predict(x).numpy()
    x = x.numpy()
    for k in range(model.k):
        plt.scatter(x[clusters == k][:, 0], x[clusters == k][:, 1], c='C' + str(k), alpha=.5)
        if means:
            plt.scatter(model.means[k][0].detach().numpy(),
                        model.means[k][1].detach().numpy(),
                        color="C" + str(k), marker='X', edgecolor="black", s=300)
    plot_info()


def plot_ellipses(model):
    if len(model.covariances.shape) != 3:
        covariances = np.tile(np.identity(model.means.shape[1]), (model.k, 1, 1))
        covariances = covariances * model.covariances.detach().numpy()[..., np.newaxis, np.newaxis]
    else:
        covariances = model.covariances.detach().numpy()
    for k in range(model.k):
        ellipse = ellipse_data(x_cent=model.means[k, 0].detach().numpy(),
                               y_cent=model.means[k, 1].detach().numpy(),
                               cov=covariances[k, :, :])
        plt.plot(ellipse[0], ellipse[1], c='C' + str(k))
    plot_info()


def plot_contours(model, x):
    x = x.numpy()
    max_x, max_y = x.max(axis=0)
    min_x, min_y = x.min(axis=0)

    n_pts_by_axes = 100
    xs = np.linspace(min_x - 1, max_x + 1, n_pts_by_axes)
    ys = np.linspace(min_y - 1, max_y + 1, n_pts_by_axes)

    X, Y = np.meshgrid(xs, ys)
    Z = model.predict(torch.from_numpy(np.c_[np.ravel(X), np.ravel(Y)])).numpy()
    plt.contour(X, Y, Z.reshape(X.shape), colors='black')


def plot_clusters_contours_ellipses(model, x):
    plot_clusters(model, x)
    plot_contours(model, x)
    plot_ellipses(model)
