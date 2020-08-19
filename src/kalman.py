import torch
from  torch.distributions.multivariate_normal import MultivariateNormal as mvn

from src.abstract_models import HiddenVariableModel, GenerativeModel, SequenceModel
# import genbmm # use this when doing log space computation


class Kalman(HiddenVariableModel, GenerativeModel, SequenceModel):
    def __init__(self, dim_x=1, dim_y=2, threshold=1e-3, max_iter=10, seed=0, sequence_length=10):
        super(Kalman, self).__init__(threshold, max_iter, seed, sequence_length)
        self.dim_x = dim_x
        self.dim_y = dim_y

        self.A = torch.empty((self.dim_x, self.dim_x))
        self.Q = torch.empty((self.dim_x, self.dim_x))
        self.C = torch.empty((self.dim_y, self.dim_x))
        self.R = torch.empty((self.dim_y, self.dim_x))

        self.initial_mean = None
        self.initial_cov = None

        self.x_pred = None
        self.cov_pred = None
        self.x_update = None
        self.cov_update = None
        self.x_smooth = None
        self.cov_smooth = None

    def initialize(self, x):
        pass

    def parameters(self):
        pass

    def _predict(self, x, cov):
        x_pred = self.A @ x
        cov_pred = self.A @ cov @ self.A.t() + self.Q
        return x_pred, cov_pred

    def _update(self, x_pred, cov_pred, y):
        kalman_gain = cov_pred @ self.C.t() @ (self.C @ cov_pred @ self.C.t() + self.R).inverse()
        x_update = x_pred + kalman_gain @ (y - self.C @ x_pred)
        cov_update = cov_pred - kalman_gain @ self.C @ cov_pred
        return x_update, cov_update

    def expectation(self, y: torch.Tensor):
        T, _ = y.shape
        self.x_pred = torch.empty((T,self.dim_x))
        self.cov_pred = torch.empty((T, self.dim_x, self.dim_x))
        self.x_update = torch.empty((T, self.dim_x))
        self.cov_update = torch.empty((T, self.dim_x, self.dim_x))
        self.x_smooth = torch.empty((T, self.dim_x))
        self.cov_smooth = torch.empty((T, self.dim_x, self.dim_x))

        # filter
        x = self.initial_mean
        cov = self.initial_cov
        for t in range(1, T):
            x_pred_t, cov_pred_t = self._predict(x, cov)
            x_update_t, cov_update_t = self._update(x_pred_t, cov_pred_t, y[t])
            x, cov = x_update_t, cov_update_t

            self.x_pred[t] = x_pred_t
            self.cov_pred[t] = cov_pred_t
            self.x_update[t] = x_update_t
            self.cov_update[t] = cov_update_t
        print(self.x_update)
        # smooth
        for t in reversed(range(T - 1)):
            L = self.cov_update[t] @ self.A.t() @ self.cov_pred[t+1].inverse()
            self.x_smooth[t] = self.x_update[t] + L @ (self.x_smooth[t+1] - self.x_pred[t+1])
            self.cov_smooth[t] = self.cov_update[t] + L @ (self.cov_smooth[t+1] - self.cov_pred[t+1]) @ L.t()

    def maximization(self, y):
        T, _ = y.shape
        self.C = torch.einsum('ti,tj->ij', y, self.x_smooth) @ self.cov_smooth.sum(0).inverse()
        self.R = ().sum(0)/T

    def predict(self, x):
        pass

    def complete_log_likelihood(self, x):
        pass

    def marginal_log_likelihood(self, x):
        pass

    def sample(self, n):
        def sample_mvn(mean, cov, dim):
            if dim > 1:
                return mvn(mean, cov).sample()
            return torch.normal(mean, cov)
        out = torch.empty((n, self.sequence_length, self.dim_y))
        out_hidden = torch.empty((n, self.sequence_length, self.dim_x))
        for i in range(n):
            x = sample_mvn(self.initial_mean, self.initial_cov, self.dim_x)
            for t in range(self.sequence_length):
                out_hidden[i,t] = x
                out[i, t] = sample_mvn(self.C @ x, self.R, self.dim_y)
                x = sample_mvn(self.A @ x, self.Q, self.dim_x)
        return out, out_hidden


if __name__ == '__main__':
    kalman = Kalman(dim_x=2, dim_y=2)
    ''' state : constant speed, obs: pos
    x = [pos,
         speed]
    A = [[1,1],
         [0,1]]
    C = [1, 0]
    '''
    kalman.A = torch.Tensor([[1, 1],
                             [0, 1]])
    kalman.Q = torch.eye(2) * 0.00001
    kalman.C = torch.Tensor([[1, 0],
                             [0,1]])
    kalman.R = torch.eye(2) * 0.00001
    kalman.initial_mean = torch.Tensor([0, 1])
    kalman.initial_cov = torch.eye(2) * 0.0001

    y, x = kalman.sample(1)
    print(y)
    lltrain, lltest = kalman.train(y[0])

