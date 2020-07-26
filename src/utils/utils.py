import numpy as np

# USEFUL FUNCTIONS
def augment_features(x):
    return np.c_[np.ones((x.shape[0], 1)), x]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def log_sum_exp(a, axis=None, keepdims=np._NoValue):
    b = np.max(a, axis=axis, keepdims=keepdims)
    return b + np.log(np.sum(np.exp(a - b), axis=axis, keepdims=keepdims))