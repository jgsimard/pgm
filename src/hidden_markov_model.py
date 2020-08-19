import torch

from src.abstract_models import HiddenVariableModel, GenerativeModel, SequenceModel
from src.distributions import GaussianDistribution
from src.mixture_model import MixtureModel


class HiddenMarkovModel(HiddenVariableModel, GenerativeModel, SequenceModel):
    def __init__(self, k=1, d=2, distribution_type=GaussianDistribution, threshold=1e-3, max_iter=10, seed=0,
                 sequence_length=10):
        super(HiddenMarkovModel, self).__init__(threshold=threshold, max_iter=max_iter, seed=seed, sequence_length=sequence_length)
        self.k = k
        self.d = d
        self.alpha_log = None
        self.beta_log = None
        self.gamma_log = None
        self.emissions_log = None
        self.q_log = None
        self.pi_log = None
        self.transition_matrix_log = None
        self.distribution_type = distribution_type
        self.distribution = distribution_type(k, d)

        self._marginal_log_likelihood = None

    def initialize(self, x):
        # initialize the emission distribution parameters by a mixture model
        mm = MixtureModel(self.k, self.d, distribution_type=self.distribution_type)
        mm.initialize(x)
        mm.train(x)
        self.distribution = mm.distribution
        # uniform prior on the hmm parameters
        self.transition_matrix_log = torch.log(torch.ones((self.k, self.k)) / self.k)
        self.pi_log = torch.log(torch.ones(self.k) / self.k)

    def parameters(self):
        return torch.exp(self.transition_matrix_log), torch.exp(self.pi_log), *self.distribution.parameters()

    def forward_backward_log(self, x):
        T, _ = x.shape

        self.emissions_log = self.distribution.log_prob(x)

        # Forward propagation
        self.alpha_log = torch.empty([T, self.k])
        self.alpha_log[0] = self.pi_log + self.emissions_log[0]
        for t in range(1, T):
            self.alpha_log[t] = self.emissions_log[t] + torch.logsumexp(
                self.transition_matrix_log + self.alpha_log[t - 1], dim=1)

        # Backward propagation
        self.beta_log = torch.empty([T, self.k])
        self.beta_log[-1] = 0
        for t in reversed(range(T - 1)):
            self.beta_log[t] = torch.logsumexp(
                self.transition_matrix_log + self.emissions_log[t + 1] + self.beta_log[t + 1], dim=1)

        # gamma
        gamma_log = self.alpha_log + self.beta_log
        self._marginal_log_likelihood = torch.logsumexp(gamma_log, dim=1, keepdim=True)
        self.gamma_log = (gamma_log - self._marginal_log_likelihood).double()

    def expectation(self, x):
        T, _ = x.shape
        self.forward_backward_log(x)
        self.q_log = self.alpha_log[:T - 1].unsqueeze(1) \
                     + self.gamma_log[1:].unsqueeze(2) \
                     + self.transition_matrix_log \
                     + self.emissions_log[1:].unsqueeze(2) \
                     - self.alpha_log[1:].unsqueeze(2)

    def maximization(self, x):
        self.pi_log = self.gamma_log[0]
        self.transition_matrix_log = torch.logsumexp(self.q_log, dim=0) - torch.logsumexp(self.gamma_log, dim=0)
        self.distribution.maximization(x, torch.exp(self.gamma_log))

    def predict(self, x):
        # AKA VITERBI
        T, _ = x.shape

        self.emissions_log = self.distribution.log_prob(x)

        forward_probs = torch.empty((T, self.k))
        forward_index = torch.empty((T, self.k))

        forward_probs[0] = self.emissions_log[0] + self.pi_log
        for t in range(1, T):
            aux = self.transition_matrix_log + forward_probs[t - 1]
            forward_probs[t] = self.emissions_log[t] + torch.max(aux, dim=1)[0]
            forward_index[t] = torch.argmax(aux, dim=1)

        backtrack_index = torch.empty(T)
        backtrack_index[-1] = torch.argmax(forward_probs[-1])
        for t in reversed(range(T - 1)):
            backtrack_index[t] = forward_index[t + 1, int(backtrack_index[t + 1])]

        return backtrack_index

    def complete_log_likelihood(self, x):
        self.expectation(x)
        gamma = torch.exp(self.gamma_log)
        likelihood = torch.sum(gamma * self.distribution.log_prob(x))
        likelihood += torch.sum(torch.exp(self.q_log) * self.transition_matrix_log)
        likelihood += torch.sum(gamma[0] * self.pi_log)
        return likelihood

    def marginal_log_likelihood(self, x):
        self.forward_backward_log(x)
        return self._marginal_log_likelihood.mean()

    def sample(self, n):
        out = torch.empty((n, self.sequence_length, self.distribution.d))
        for i in range(n):
            z = torch.multinomial(torch.exp(self.pi_log), 1)
            for t in range(self.sequence_length):
                out[i, t] = self.distribution.sample(z)
                z = torch.multinomial(torch.exp(self.transition_matrix_log)[:, z].view(-1), 1)
        return out


if __name__ == '__main__':
    from datasets.em_gaussian import EMGaussianDataset
    from src.utils.plot import plot_clusters, plot_ellipses
    import matplotlib.pyplot as plt

    x = EMGaussianDataset("../datasets/data/EMGaussian")[0]

    hmm = HiddenMarkovModel(k=4, d=2, max_iter=10, threshold=1e-3, sequence_length=200, seed=100)
    hmm.initialize(x)
    hmm.train(x, likelihood=False)
    print("hmm.pi\n", torch.exp(hmm.pi_log))
    print("hmm.distribution.means\n", hmm.distribution.means)
    print("hmm.distribution.covariances\n", hmm.distribution.covariances)
    print("hmm.transition_matrix\n", torch.exp(hmm.transition_matrix_log))

    print(hmm.complete_log_likelihood(x))
    print(hmm.marginal_log_likelihood(x))
    print(hmm.normalized_negative_complete_log_likelihood(x))
    print(hmm.normalized_negative_marginal_log_likelihood(x))

    plot_ellipses(hmm.distribution)
    hmm.distribution.predict = lambda x: hmm.predict(x)
    plot_clusters(hmm.distribution, x)
    # plot_clusters(hmm.emission_distribution, hmm.sample(1)[0], False)
    plt.show()
    # print(hmm.sample(1)[0].shape)
