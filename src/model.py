import torch


class Model(object):
    def __init__(self):
        pass

    def  initialize(self, data):
        pass

    def train(self, *args, **kwargs):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()


class HiddenVariableModel(Model):
    def __init__(self):
        super(HiddenVariableModel, self).__init__()

    def expectation(self, *args, **kwargs):
        raise NotImplementedError()

    def maximization(self, *args, **kwargs):
        raise NotImplementedError()

    def mean_complete_negative_log_likelihood(self, x):
        raise NotImplementedError()

    def mean_marginal_negative_log_likelihood(self, x):
        raise NotImplementedError()