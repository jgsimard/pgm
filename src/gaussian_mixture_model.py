import torch

from src.distributions import GaussianDistribution
from src.kmeans import KMeans
from src.mixture_model import MixtureModel


class GaussianMixtureModel(MixtureModel):
    def __init__(self, k, d=2, covariance_type='full', threshold=.01, max_iter=20, seed=0):
        super(GaussianMixtureModel, self).__init__(k, d, threshold, max_iter, seed)
        self.distribution = GaussianDistribution(k, d, covariance_type)

    def initialize(self, data):
        n, d = data.shape

        kmeans = KMeans(k=self.k)
        kmeans.train(data)

        clusters = kmeans.predict(data)
        self.pi = torch.stack([(clusters == k).sum() for k in range(self.k)]).float() / n

        # gaussian dist parameters init
        self.distribution.means = kmeans.means
        self.distribution.covariances *= 1000  # high cov is better for initialization


if __name__ == '__main__':
    from datasets.em_gaussian import EMGaussianDataset

    dataset = EMGaussianDataset("../datasets/data/EMGaussian")
    x = dataset[0]


    def test_gmm(gmm):
        gmm.initialize(x)
        gmm.train(x)
        # plot_clusters_contours_ellipses(gmm, x)
        # plt.show()
        print(gmm.normalized_negative_marginal_log_likelihood(x))
        print(gmm.normalized_negative_complete_log_likelihood(x))


    gmm_isotropic = GaussianMixtureModel(4, covariance_type="isotropic")
    test_gmm(gmm_isotropic)
    gmm_full = GaussianMixtureModel(4)
    test_gmm(gmm_full)

    # samples = gmm_isotropic.sample(10000)
    # plot_dataset(samples)
    # plt.show()
