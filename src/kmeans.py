import torch
from src.utils.time import timeit


class KMeans:
    def __init__(self, k, max_iter=10):
        self.k = k
        self.means = None
        self.max_iter = max_iter

    @timeit
    def train(self, x):
        self.means = self.initialization_plus_plus(x)
        for i in range(self.max_iter):
            # E-step
            labels = self.predict(x)
            # M-step
            self.means = torch.stack([torch.mean(x[labels == j], dim=0) for j in range(self.k)])

    def initialization_plus_plus(self, data):
        n, d = data.shape
        means = torch.zeros((self.k, d))
        means[0, :] = data[torch.randint(n,(1,))]
        distances = torch.norm(data - means[0, :], dim=1)
        for i in range(1, self.k):
            distances = torch.min(torch.norm(data - means[i], dim=1), distances)
            means[i, :] = data[torch.multinomial(distances, 1)]
        return means

    def predict(self, x):
        return torch.argmin(torch.norm(x.unsqueeze(1) - self.means, dim=-1), dim=1)

    def loss(self, x):
        return torch.mean(torch.min(torch.norm(x.unsqueeze(1) - self.means, dim=-1) ** 2, dim=1)[0])


if __name__ == '__main__':
    from datasets.em_gaussian import EMGaussianDataset
    from src.utils.plot import plot_contours, plot_clusters
    import matplotlib.pyplot as plt

    dataset = EMGaussianDataset("../datasets/data/EMGaussian")
    x = dataset[0]

    kmeans = KMeans(4)
    kmeans.train(x)
    plot_clusters(kmeans, x)
    plot_contours(kmeans, x)
    plt.show()
