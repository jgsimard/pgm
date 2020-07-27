import torch


class Model(object):
    def __init__(self, seed=0):
        self.seed = seed
        torch.manual_seed(seed)

    def initialize(self, data):
        pass

    def train(self, *args, **kwargs):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()


class HiddenVariableModel(Model):
    def __init__(self, seed=0):
        super(HiddenVariableModel, self).__init__(seed)

    def expectation(self, *args, **kwargs):
        raise NotImplementedError()

    def maximization(self, *args, **kwargs):
        raise NotImplementedError()

    def mean_complete_negative_log_likelihood(self, x):
        raise NotImplementedError()

    def mean_marginal_negative_log_likelihood(self, x):
        raise NotImplementedError()