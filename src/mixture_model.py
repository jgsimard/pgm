import torch

from src.abstract_models import HiddenVariableModel, GenerativeModel
from src.distributions import GaussianDistribution
from src.utils.registery import get_distribution_initialization


class MixtureModel(HiddenVariableModel, GenerativeModel):
    def __init__(self, k, d=2, distribution_type=GaussianDistribution,  threshold=.01, max_iter=20, seed=0):
        super(MixtureModel, self).__init__(threshold, max_iter, seed)
        self.k = k
        self.d = d
        self.distribution_type = distribution_type
        self.distribution = distribution_type(k,d)
        self.tau_log = None
        self.pi = torch.ones(self.k) / self.k # uniform prior

    def initialize(self, data):
        initializer = get_distribution_initialization(self.distribution.__class__)
        initializer(self, data)
        self.pi = torch.ones(self.k) / self.k  # uniform prior

    def parameters(self):
        return self.pi, *self.distribution.parameters()

    def expectation(self, x):
        tau = self.distribution.log_prob(x) + torch.log(self.pi.view(1, -1))
        self.tau_log = tau - torch.logsumexp(tau, dim=1, keepdim=True)

    def maximization(self, x):
        tau = torch.exp(self.tau_log)
        self.pi = tau.mean(dim=0)
        self.distribution.maximization(x, tau)

    def predict(self, x):
        return torch.argmax(self.distribution.log_prob(x), dim=1)

    def complete_log_likelihood(self, x):
        self.expectation(x)
        return (torch.exp(self.tau_log) * (self.distribution.log_prob(x) + torch.log(self.pi))).sum(dim=1).sum()

    def marginal_log_likelihood(self, x):
        return torch.logsumexp(self.distribution.log_prob(x) + torch.log(self.pi.view(1, -1)), dim=1).sum()

    def sample(self, n=1):
        out = torch.empty((n, self.d))
        for i in range(n):
            z_i = torch.multinomial(self.pi, 1)
            x_i = self.distribution.sample(z_i)
            out[i] = x_i
        return out


if __name__ == '__main__':
    from datasets.em_gaussian import EMGaussianDataset
    from src.utils.plot import plot_clusters_contours_ellipses
    import matplotlib.pyplot as plt
    from functools import partial

    x = EMGaussianDataset("../datasets/data/EMGaussian")[0]

    def _gmm(gmm):
        gmm.initialize(x)
        gmm.train(x)
        gmm.distribution.predict = lambda x: gmm.predict(x)
        plot_clusters_contours_ellipses(gmm.distribution, x)
        plt.show()
        print(gmm.distribution.covariance_type)
        print(f"normalized_negative_marginal_log_likelihood={gmm.normalized_negative_marginal_log_likelihood(x)}")
        print(f"normalized_negative_complete_log_likelihood={gmm.normalized_negative_complete_log_likelihood(x)}")

    gmm_isotropic = MixtureModel(4, distribution_type= partial(GaussianDistribution, covariance_type='isotropic'))
    _gmm(gmm_isotropic)
    gmm_full = MixtureModel(4, distribution_type=GaussianDistribution)
    _gmm(gmm_full)