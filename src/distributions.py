import torch
from torch.distributions.multivariate_normal import MultivariateNormal as mvn


class Distribution:
    def __init__(self, k=1, d=1):
        self.k = k
        self.d = d

    def log_prob(self, x):
        raise NotImplementedError()

    def maximization(self, *args, **kwargs):
        raise NotImplementedError()

    def sample(self, *args, **kwargs):
        raise NotImplementedError()

    def parameters(self):
        raise NotImplementedError()


class GaussianDistribution(Distribution):
    def __init__(self, k, d, covariance_type='full'):
        super(GaussianDistribution, self).__init__(k, d)
        self.means = torch.zeros((k, d))

        if covariance_type not in ['isotropic', 'full']:
            raise NotImplementedError(f"Gaussian type {covariance_type} is not implemented")
        self.covariance_type = covariance_type

        if self.covariance_type == 'isotropic':
            self.covariances = torch.ones(self.k).double()
        elif self.covariance_type == 'full':
            self.covariances = torch.eye(d).repeat(self.k, 1, 1).view(self.k, d, d).double()

    def log_prob(self, x):
        if self.covariance_type == 'isotropic':
            r = [mvn(mean, cov * torch.eye(self.d).double()).log_prob(x) for mean, cov in
                 zip(self.means, self.covariances)]
        elif self.covariance_type == 'full':
            r = [mvn(mean, cov).log_prob(x) for mean, cov in zip(self.means, self.covariances)]
        return torch.stack(r, dim=1)

    def maximization(self, x, weights):
        normalization = weights.sum(dim=0)
        self.means = torch.einsum('ni,nk->ki', x, weights) / normalization.view(-1, 1)
        delta = x.unsqueeze(1) - self.means

        if self.covariance_type == 'isotropic':
            self.covariances = torch.einsum('nki,nki,nk->k', delta, delta, weights) / self.d / normalization
        elif self.covariance_type == 'full':
            self.covariances = torch.einsum('nki,nkj,nk->kij', delta, delta, weights) / normalization.view(-1, 1, 1)

    def sample(self, z):
        mean = self.means[z]
        covariance = self.covariances[z]
        if self.covariance_type == 'isotropic':
            covariance = covariance * torch.eye(self.d)
        return torch.distributions.MultivariateNormal(mean, covariance).sample()

    def parameters(self):
        return self.means, self.covariances
