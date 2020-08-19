import torch

from src.kmeans import KMeans


def gaussian_mixture_model_initializer(mm, data):
    n, d = data.shape

    kmeans = KMeans(k=mm.k)
    kmeans.initialize(data)
    kmeans.train(data)

    clusters = kmeans.predict(data)
    mm.pi = torch.stack([(clusters == k).sum() for k in range(mm.k)]).float() / n

    # gaussian dist parameters init
    mm.distribution.means = kmeans.means
    mm.distribution.covariances *= 10  # high cov is better for initialization