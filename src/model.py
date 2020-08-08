from abc import ABC

import torch


class Model(ABC):
    def __init__(self, seed=0):
        self.seed = seed
        torch.manual_seed(seed)

    def initialize(self, data):
        pass

    def train(self, *args, **kwargs):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()


class GenerativeModel(Model, ABC):
    def sample(self, n):
        raise NotImplementedError()


class SequenceModel(Model, ABC):
    def __init__(self, seed=0, sequence_length=1):
        super(SequenceModel, self).__init__(seed)
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
