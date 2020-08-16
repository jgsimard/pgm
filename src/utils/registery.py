from src.distributions import GaussianDistribution
from src.gaussian_mixture_model import GaussianMixtureModel


def get_mixture_model(dist):
    if dist == GaussianDistribution:
        return GaussianMixtureModel
