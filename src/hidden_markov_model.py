import torch
from torch.distributions.multivariate_normal import MultivariateNormal as mvn
from src.model import HiddenVariableModel
from src.gaussian_mixture_model import GaussianMixtureModel
from src.utils.time import timeit


class EmissionDistribution:
    def __init__(self, k, homework_init=False):
        self.k = k

    def initialize(self, x):
        raise NotImplementedError()

    def pdf(self, x):
        raise NotImplementedError()

    def maximization(self, x, gamma):
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

    def pdf(self, x):
        return torch.stack([mvn(mean, cov).log_prob(x) for mean, cov in zip(self.means, self.covariances)], dim=1)

    def maximization(self, x, gamma):
        normalization = gamma.sum(dim=0)
        self.means = torch.einsum('ni,nj->ji', x, gamma) / normalization.view(-1, 1)
        delta = x.unsqueeze(1) - self.means
        self.covariances = torch.einsum('nki,nkj,nk->kij', delta, delta, gamma) / normalization.view(-1, 1, 1)


class HiddenMarkovModel(HiddenVariableModel):
    def __init__(self, k, emission_distribution_type=GaussianEmissionDistribution, threshold=1e-3, max_iter=10):
        super(HiddenMarkovModel, self).__init__()
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
        self.mean_marginal_negative_log_likelihood_ = None

    def initialize(self, x):
        self.emission_distribution.initialize(x)
        self.transition_matrix_log = torch.log(torch.ones((self.k, self.k)) / self.k)
        self.pi_log = torch.log(torch.ones(self.k) / self.k)

    def forward_backward_log(self, x):
        T, _ = x.shape

        self.emissions_log = self.emission_distribution.pdf(x)

        # Forward propagation
        self.alpha_log = torch.zeros([T, self.k])
        self.alpha_log[0] = self.pi_log + self.emissions_log[0]
        for t in range(1, T):
            self.alpha_log[t] = self.emissions_log[t] + torch.logsumexp(self.transition_matrix_log + self.alpha_log[t - 1], dim=1)

        # Backward propagation
        self.beta_log = torch.zeros([T, self.k])
        self.beta_log[-1] = 0
        for t in reversed(range(T-1)):
            self.beta_log[t] = torch.logsumexp(self.transition_matrix_log + self.emissions_log[t + 1] + self.beta_log[t + 1], dim=1)

        # gamma
        gamma_log = self.alpha_log + self.beta_log
        # normalize gamma_log
        likelihood = torch.logsumexp(gamma_log, dim=1, keepdim=True)
        self.mean_marginal_negative_log_likelihood_ = -torch.mean(likelihood) / T
        self.gamma_log = (gamma_log - likelihood).double()

    def expectation(self, x):
        T,_ = x.shape
        self.forward_backward_log(x)
        self.q_log = self.alpha_log[:T - 1].unsqueeze(1) \
                     + self.gamma_log[1:].unsqueeze(2) \
                     + self.transition_matrix_log \
                     + self.emissions_log[1:].unsqueeze(2) \
                     - self.alpha_log[1:].unsqueeze(2)

    def maximization(self, x):
        gamma = torch.exp(self.gamma_log)
        self.pi_log = self.gamma_log[0]
        self.transition_matrix_log = torch.logsumexp(self.q_log, dim=0) - torch.logsumexp(self.gamma_log, dim=0)
        self.emission_distribution.maximization(x, gamma)

    @timeit
    def train(self, x_train, x_test=None, likelihood=False):
        self.initialize(x_train)
        likelihoods_train, likelihoods_test = [], []

        def compute_likelihoods():
            if likelihood:
                likelihoods_train.append(self.mean_complete_negative_log_likelihood(x_train))
                if x_test is not None:
                    likelihoods_test.append(self.mean_complete_negative_log_likelihood(x_test))

        compute_likelihoods()
        for i in range(self.max_iter):
            old = self.mean_marginal_negative_log_likelihood_
            self.expectation(x_train)
            self.maximization(x_train)
            compute_likelihoods()
            if old is not None:
                if torch.abs(old - self.mean_marginal_negative_log_likelihood_) < self.threshold:
                    break
        return likelihoods_train, likelihoods_test

    def viterbi(self, x):
        T, d = x.shape

        self.emissions_log = self.emission_distribution.pdf(x)

        forward_probs = torch.zeros((T, self.k))
        forward_index = torch.zeros((T, self.k))

        forward_probs[0] = self.emissions_log[0] + self.pi_log
        for t in range(1, T):
            aux = self.transition_matrix_log + forward_probs[t - 1]
            forward_probs[t] = self.emissions_log[t] + torch.max(aux, dim=1)[0]
            forward_index[t] = torch.argmax(aux, dim=1)

        backtrack_index = torch.zeros(T)
        backtrack_index[-1] = torch.argmax(forward_probs[-1])
        for t in reversed(range(T-1)):
            backtrack_index[t] = forward_index[t + 1, int(backtrack_index[t + 1])]

        return backtrack_index

    def mean_complete_negative_log_likelihood(self, x):
        self.expectation(x)
        gamma = torch.exp(self.gamma_log)
        likelihood = torch.sum(gamma * self.emission_distribution.pdf(x))
        likelihood += torch.sum(torch.exp(self.q_log) * self.transition_matrix_log)
        likelihood += torch.sum(gamma[0] * self.pi_log)
        return -likelihood / x.shape[0]

    def mean_marginal_negative_log_likelihood(self, x):
        indexes = self.viterbi(x).long().view(-1,1)
        likelihoods = torch.gather(self.emissions_log, 1,indexes)
        return -likelihoods.mean()


if __name__ == '__main__':
    from datasets.em_gaussian import EMGaussianDataset
    from src.utils.plot import plot_contours, plot_clusters, plot_ellipses
    import matplotlib.pyplot as plt

    dataset = EMGaussianDataset("../datasets/data/EMGaussian")

    gmm_isotropic = GaussianMixtureModel(4, covariance="isotropic")
    x = dataset[0]

    hmm = HiddenMarkovModel(4, max_iter=100, threshold=1e-3)
    likelihoods_train, likelihoods_test = hmm.train(x, likelihood=True)
    # print("hmm.pi\n", hmm.pi_log.exp_())
    # print("hmm.emission_distribution.means\n", hmm.emission_distribution.means)
    # print("hmm.emission_distribution.covariances\n", hmm.emission_distribution.covariances)
    print("hmm.transition_matrix\n", torch.exp(hmm.transition_matrix_log))
    print(torch.exp(hmm.transition_matrix_log).sum(dim=0))
    print(f"hmm.mean_marginal_negative_log_likelihood = {hmm.mean_marginal_negative_log_likelihood_}")
    print(likelihoods_train)
    print(hmm.mean_complete_negative_log_likelihood(x))
    print(hmm.mean_marginal_negative_log_likelihood(x))
    # plot_ellipses(hmm.emission_distribution)
    # hmm.emission_distribution.predict = lambda _: hmm.viterbi(x)
    # plot_clusters(hmm.emission_distribution, x)
    # plt.show()