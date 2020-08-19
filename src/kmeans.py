import torch

from src.abstract_models import HiddenVariableModel


class KMeans(HiddenVariableModel):
    def __init__(self, k, threshold=1e-3, max_iter=10, seed=0):
        super(KMeans, self).__init__(threshold, max_iter, seed)
        self.k = k
        self.labels = None
        self.means = None

    def initialize(self, data):
        # initialization ++
        n, d = data.shape
        means = torch.zeros((self.k, d))
        means[0, :] = data[torch.randint(n, (1,))]
        distances = torch.norm(data - means[0, :], dim=1)
        for i in range(1, self.k):
            distances = torch.min(torch.norm(data - means[i], dim=1), distances)
            means[i, :] = data[torch.multinomial(distances, 1)]
        self.means = means

    def expectation(self, x):
        self.labels = torch.argmin(torch.norm(x.unsqueeze(1) - self.means, dim=-1), dim=1)

    def maximization(self, x):
        self.means = torch.stack([torch.mean(x[self.labels == j], dim=0) for j in range(self.k)])

    def train(self, x_train, x_test=None, likelihood=False):
        old_loss = self.loss(x_train)
        for i in range(self.max_iter):
            self.expectation(x_train)
            self.maximization(x_train)
            new_loss = self.loss(x_train)
            if torch.abs(new_loss - old_loss) < self.threshold:
                break
            old_loss = new_loss

    def predict(self, x):
        self.expectation(x)
        return self.labels

    def loss(self, x):
        return torch.mean(torch.min(torch.norm(x.unsqueeze(1) - self.means, dim=-1) ** 2, dim=1)[0])


if __name__ == '__main__':
    from datasets.em_gaussian import EMGaussianDataset
    from src.utils.plot import plot_contours, plot_clusters
    import matplotlib.pyplot as plt

    x = EMGaussianDataset("../datasets/data/EMGaussian")[0]

    kmeans = KMeans(4)
    kmeans.initialize(x)
    kmeans.train(x)
    plot_clusters(kmeans, x)
    plot_contours(kmeans, x)
    plt.show()
