from src.distributions import GaussianDistribution
from src.initializers import gaussian_mixture_model_initializer


def get_distribution_initialization(dist):
    if dist == GaussianDistribution:
        return gaussian_mixture_model_initializer
