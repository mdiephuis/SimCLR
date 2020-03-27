import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
import torch.nn.functional as F


class MiEstimator(nn.Module):
    def __init__(self, size1, size2, d):
        super(MiEstimator, self).__init__()
        self.size1 = size1
        self.size2 = size2
        self.d = d

        self.network = nn.Sequential(
            nn.Linear(self.size1 + self.size2, self.d),
            nn.ReLU(True),
            nn.Linear(self.d, self.d),
            nn.ReLU(True),
            nn.Linear(self.d, 1)
        )

    def forward(self, x1, x2):
        # Gradient for JSD mutual information estimation and EB-based estimation
        pos = self.network(torch.cat([x1, x2], -1))  # Positive Samples
        neg = self.network(torch.cat([torch.roll(x1, 1, 0), x2], -1))
        grad = -F.softplus(-pos).mean() - F.softplus(neg).mean()
        out = pos.mean() - neg.exp().mean() + 1
        return grad, out


class MNIST_Encoder(nn.Module):
    def __init__(self, input_dim, z_dim):
        super(MNIST_Encoder, self).__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim

        # Vanilla MLP
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, z_dim * 2),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        params = self.network(x)

        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        sigma = F.softplus(sigma) + 1e-7

        return Independent(Normal(loc=mu, scale=sigma), 1)


