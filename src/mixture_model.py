import torch

from src.abstract_models import HiddenVariableModel, GenerativeModel


class MixtureModel(HiddenVariableModel, GenerativeModel):
    def __init__(self, k, d=2, threshold=.01, max_iter=20, seed=0):
        super(MixtureModel, self).__init__(threshold, max_iter, seed)
        self.k = k
        self.d = d
        self.distribution = None
        self.tau_log = None
        self.pi = None

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
