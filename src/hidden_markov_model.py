import torch
from torch.distributions.multivariate_normal import MultivariateNormal as mvn
from src.model import HiddenVariableModel, GenerativeModel, SequenceModel
from src.gaussian_mixture_model import GaussianMixtureModel
from src.utils.time import timeit


class EmissionDistribution:
    def __init__(self, k, dim=1):
        self.k = k
        self.dim = dim

    def initialize(self, *args, **kwargs):
        raise NotImplementedError()

    def log_prob(self, x):
        raise NotImplementedError()

    def maximization(self, *args, **kwargs):
        raise NotImplementedError()

    def sample(self, *args, **kwargs):
        raise NotImplementedError()


class GaussianEmissionDistribution(EmissionDistribution):
    def __init__(self, k):
        super(GaussianEmissionDistribution, self).__init__(k)
        self.means = None
        self.covariances = None

    def initialize(self, x):
        gmm = GaussianMixtureModel(self.k)
        gmm.initialize(x)
        gmm.train(x)
        self.means = gmm.means
        self.covariances = gmm.covariances
        self.dim = self.means.shape[1]

    def log_prob(self, x):
        return torch.stack([mvn(mean, cov).log_prob(x) for mean, cov in zip(self.means, self.covariances)], dim=1)

    def maximization(self, x, gamma):
        normalization = gamma.sum(dim=0)
        self.means = torch.einsum('ni,nk->ki', x, gamma) / normalization.view(-1, 1)
        delta = x.unsqueeze(1) - self.means
        self.covariances = torch.einsum('nki,nkj,nk->kij', delta, delta, gamma) / normalization.view(-1, 1, 1)

    def sample(self, z):
        return torch.distributions.MultivariateNormal(self.means[z], self.covariances[z]).sample()


class HiddenMarkovModel(HiddenVariableModel, GenerativeModel, SequenceModel):
    def __init__(self, k, emission_distribution_type=GaussianEmissionDistribution, threshold=1e-3, max_iter=10, seed=0, sequence_length=10):
        super(HiddenMarkovModel, self).__init__(seed=seed, sequence_length=sequence_length)
        self.k = k
        self.alpha_log = None
        self.beta_log = None
        self.gamma_log = None
        self.emissions_log = None
        self.q_log = None
        self.pi_log = None
        self.transition_matrix_log = None
        self.emission_distribution = emission_distribution_type(k)
        self.threshold = threshold
        self.max_iter = max_iter

        self._marginal_log_likelihood = torch.Tensor(0)

    def initialize(self, x):
        self.emission_distribution.initialize(x)
        self.transition_matrix_log = torch.log(torch.ones((self.k, self.k)) / self.k)
        self.pi_log = torch.log(torch.ones(self.k) / self.k)

    def forward_backward_log(self, x):
        T, _ = x.shape

        self.emissions_log = self.emission_distribution.log_prob(x)

        # Forward propagation
        self.alpha_log = torch.empty([T, self.k])
        self.alpha_log[0] = self.pi_log + self.emissions_log[0]
        for t in range(1, T):
            self.alpha_log[t] = self.emissions_log[t] + torch.logsumexp(self.transition_matrix_log + self.alpha_log[t - 1], dim=1)

        # Backward propagation
        self.beta_log = torch.empty([T, self.k])
        self.beta_log[-1] = 0
        for t in reversed(range(T-1)):
            self.beta_log[t] = torch.logsumexp(self.transition_matrix_log + self.emissions_log[t + 1] + self.beta_log[t + 1], dim=1)

        # gamma
        gamma_log = self.alpha_log + self.beta_log
        self._marginal_log_likelihood = torch.logsumexp(gamma_log, dim=1, keepdim=True)
        self.gamma_log = (gamma_log - self._marginal_log_likelihood).double()

    def expectation(self, x):
        T,_ = x.shape
        self.forward_backward_log(x)
        self.q_log = self.alpha_log[:T - 1].unsqueeze(1) \
                     + self.gamma_log[1:].unsqueeze(2) \
                     + self.transition_matrix_log \
                     + self.emissions_log[1:].unsqueeze(2) \
                     - self.alpha_log[1:].unsqueeze(2)

    def maximization(self, x):
        self.pi_log = self.gamma_log[0]
        self.transition_matrix_log = torch.logsumexp(self.q_log, dim=0) - torch.logsumexp(self.gamma_log, dim=0)
        self.emission_distribution.maximization(x, torch.exp(self.gamma_log))

    @timeit
    def train(self, x_train, x_test=None, likelihood=False):
        self.initialize(x_train)
        likelihoods_train, likelihoods_test = [], []

        def compute_likelihoods():
            if likelihood:
                likelihoods_train.append(self.normalized_negative_complete_log_likelihood(x_train))
                if x_test is not None:
                    likelihoods_test.append(self.normalized_negative_complete_log_likelihood(x_test))

        compute_likelihoods()
        for i in range(self.max_iter):
            old = self._marginal_log_likelihood.mean()
            self.expectation(x_train)
            self.maximization(x_train)
            compute_likelihoods()
            if old is not None:
                if torch.abs(old - self._marginal_log_likelihood.mean()) < self.threshold:
                    break
        return likelihoods_train, likelihoods_test

    def predict(self, x):
        # AKA VITERBI
        T, _ = x.shape

        self.emissions_log = self.emission_distribution.log_prob(x)

        forward_probs = torch.empty((T, self.k))
        forward_index = torch.empty((T, self.k))

        forward_probs[0] = self.emissions_log[0] + self.pi_log
        for t in range(1, T):
            aux = self.transition_matrix_log + forward_probs[t - 1]
            forward_probs[t] = self.emissions_log[t] + torch.max(aux, dim=1)[0]
            forward_index[t] = torch.argmax(aux, dim=1)

        backtrack_index = torch.empty(T)
        backtrack_index[-1] = torch.argmax(forward_probs[-1])
        for t in reversed(range(T-1)):
            backtrack_index[t] = forward_index[t + 1, int(backtrack_index[t + 1])]

        return backtrack_index

    def complete_log_likelihood(self, x):
        self.expectation(x)
        gamma = torch.exp(self.gamma_log)
        likelihood = torch.sum(gamma * self.emission_distribution.log_prob(x))
        likelihood += torch.sum(torch.exp(self.q_log) * self.transition_matrix_log)
        likelihood += torch.sum(gamma[0] * self.pi_log)
        return likelihood

    def marginal_log_likelihood(self, x):
        return self._marginal_log_likelihood.mean()

    def sample(self, n):
        out = torch.empty((n, self.sequence_length, self.emission_distribution.dim))
        for i in range(n):
            z = torch.multinomial(torch.exp(self.pi_log), 1)
            for t in range(self.sequence_length):
                out[i, t] = self.emission_distribution.sample(z)
                z = torch.multinomial(torch.exp(self.transition_matrix_log)[:,z].view(-1), 1)
        return out


if __name__ == '__main__':
    from datasets.em_gaussian import EMGaussianDataset
    from src.utils.plot import plot_contours, plot_clusters, plot_ellipses
    import matplotlib.pyplot as plt

    x = EMGaussianDataset("../datasets/data/EMGaussian")[0]

    hmm = HiddenMarkovModel(4, max_iter=10, threshold=1e-3, sequence_length=200)
    hmm.train(x, likelihood=False)
    print("hmm.pi\n", torch.exp(hmm.pi_log))
    print("hmm.emission_distribution.means\n", hmm.emission_distribution.means)
    print("hmm.emission_distribution.covariances\n", hmm.emission_distribution.covariances)
    print("hmm.transition_matrix\n", torch.exp(hmm.transition_matrix_log))

    print(hmm.complete_log_likelihood(x))
    print(hmm.marginal_log_likelihood(x))
    print(hmm.normalized_negative_complete_log_likelihood(x))
    print(hmm.normalized_negative_marginal_log_likelihood(x))

    # plot_ellipses(hmm.emission_distribution)
    hmm.emission_distribution.predict = lambda x: hmm.predict(x)
    plot_clusters(hmm.emission_distribution, hmm.sample(1)[0], False)
    plt.show()
    print(hmm.sample(1)[0].shape)
