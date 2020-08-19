from abc import ABC

import torch

from src.utils.time import timeit


class Model(ABC):
    def __init__(self, threshold=1e-3, max_iter=10, seed=0):
        self.seed = seed
        torch.manual_seed(self.seed)
        self.threshold = threshold
        self.max_iter = max_iter

    def initialize(self, data):
        pass

    def train(self, *args, **kwargs):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()

    def parameters(self):
        raise NotImplementedError()


class GenerativeModel(Model, ABC):
    def sample(self, n):
        raise NotImplementedError()


class SequenceModel(Model, ABC):
    def __init__(self, threshold=1e-3, max_iter=10, seed=0, sequence_length=1):
        super(SequenceModel, self).__init__(threshold, max_iter, seed)
        self.sequence_length = sequence_length


class HiddenVariableModel(Model, ABC):
    def expectation(self, *args, **kwargs):
        raise NotImplementedError()

    def maximization(self, *args, **kwargs):
        raise NotImplementedError()

    def complete_log_likelihood(self, x):
        raise NotImplementedError()

    def marginal_log_likelihood(self, x):
        raise NotImplementedError()

    def normalized_negative_complete_log_likelihood(self, x):
        return - self.complete_log_likelihood(x) / x.shape[0]

    def normalized_negative_marginal_log_likelihood(self, x):
        return - self.marginal_log_likelihood(x) / x.shape[0]

    # @timeit
    def train(self, x_train, x_test=None, likelihood=False):
        likelihoods_train, likelihoods_test = [], []

        def compute_likelihoods():
            if likelihood:
                likelihoods_train.append(self.normalized_negative_complete_log_likelihood(x_train))
                if x_test is not None:
                    likelihoods_test.append(self.normalized_negative_complete_log_likelihood(x_test))

        compute_likelihoods()
        for i in range(self.max_iter):
            old_parameters = self.parameters()

            self.expectation(x_train)
            self.maximization(x_train)
            compute_likelihoods()
            if max([torch.norm(param - old_param) for param, old_param in zip(self.parameters(), old_parameters)]) < self.threshold:
                break
        return likelihoods_train, likelihoods_test
