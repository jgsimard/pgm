from src.kmeans import KMeans
import torch
from src.utils.time import timeit
from torch.distributions.multivariate_normal import MultivariateNormal as mvn
from src.model import HiddenVariableModel


class GaussianMixtureModel(HiddenVariableModel):
    def __init__(self, k, d=2, covariance='full', threshold=.01, max_iter=20, seed=0):
        super(GaussianMixtureModel, self).__init__(seed)
        self.k = k
        self.d = d
        self.covariance = covariance

        self.tau_log = None

        self.pi = None
        self.means = None
        self.covariances = None

        self.threshold = threshold
        self.max_iter = max_iter

    def initialize(self, data):
        n, d = data.shape

        kmeans = KMeans(k=self.k)
        kmeans.train(data)

        self.means = kmeans.means

        clusters = kmeans.predict(data)
        self.pi = torch.stack([(clusters == k).sum() for k in range(self.k)]).float()/n

        if self.covariance == 'isotropic':
            self.covariances = 10 * torch.ones(self.k).double()
        else:
            self.covariances = 10 * torch.eye(d).repeat(self.k, 1, 1).view(self.k, d, d).double()

    def _log_likelihood(self, x):
        if self.covariance == 'isotropic':
            r = [mvn(mean, cov * torch.eye(self.d).double()).log_prob(x) for mean, cov in zip(self.means, self.covariances)]
        else:
            r = [mvn(mean, cov).log_prob(x) for mean, cov in zip(self.means, self.covariances)]
        return torch.stack(r, dim=1)

    def expectation(self, x):
        tau = self._log_likelihood(x) + torch.log(self.pi.view(1, -1))
        self.tau_log = tau - torch.logsumexp(tau, dim=1, keepdim=True)

    def maximization(self, x):
        n, d = x.shape
        tau = torch.exp(self.tau_log)
        normalization = tau.sum(dim=0)

        self.pi = normalization / n
        self.means = torch.einsum('ni,nk->ki', x, tau) / normalization.view(-1, 1)

        delta = x.unsqueeze(1) - self.means
        if self.covariance == 'isotropic':
            self.covariances = torch.einsum('nki,nki,nk->k', delta, delta, tau) / d / normalization
        else:
            self.covariances = torch.einsum('nki,nkj,nk->kij', delta, delta, tau) / normalization.view(-1, 1, 1)

    @timeit
    def train(self, x, show=False):
        for i in range(self.max_iter):
            pi, mus, sigmas = self.pi, self.means, self.covariances

            self.expectation(x)
            self.maximization(x)

            if max(torch.norm(pi - self.pi),
                   torch.norm(mus - self.means),
                   torch.norm(sigmas - self.covariances)) < self.threshold:
                break

    def predict(self, x):
        return torch.argmax(self._log_likelihood(x), dim=1)

    def mean_marginal_negative_log_likelihood(self, x):
        return -torch.logsumexp(self._log_likelihood(x) + torch.log(self.pi.view(1, -1)), dim=1).mean()

    def mean_complete_negative_log_likelihood(self, x):
        self.expectation(x)
        return -(torch.exp(self.tau_log) * (self._log_likelihood(x) + torch.log(self.pi))).sum(dim=1).mean()


if __name__ == '__main__':
    from datasets.em_gaussian import EMGaussianDataset
    from src.utils.plot import plot_clusters_contours_ellipses
    import matplotlib.pyplot as plt

    dataset = EMGaussianDataset("../datasets/data/EMGaussian")
    x = dataset[0]

    def test_gmm(gmm):
        gmm.initialize(x)
        gmm.train(x)
        plot_clusters_contours_ellipses(gmm, x)
        plt.show()
        print(gmm.mean_marginal_negative_log_likelihood(x))
        print(gmm.mean_complete_negative_log_likelihood(x))

    gmm_isotropic = GaussianMixtureModel(4, covariance="isotropic")
    test_gmm(gmm_isotropic)
    gmm_full = GaussianMixtureModel(4)
    test_gmm(gmm_full)