from torchvision.datasets import utils
import torch.utils.data as data_utils
import torch
import os
import numpy as np
from torch import nn
from torch.functional import F
from torch.optim import Adam
import math


class Encoder(nn.Module):
    def __init__(self, dim_hidden):
        super(Encoder, self).__init__()
        self.dim_hidden = dim_hidden
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 256, (5, 5)),
            nn.ELU()
        )
        self.linear_mean = nn.Linear(256, self.dim_hidden, bias=True)
        self.linear_log_var = nn.Linear(256, self.dim_hidden, bias=True)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 256)
        mu = self.linear_mean(x)
        log_var = self.linear_log_var(x)
        return mu, log_var


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode='bilinear', align_corners=True):
        super(Interpolate, self).__init__()
        self.scale_factor=scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)


class Decoder(nn.Module):
    def __init__(self, dim_hidden=100):
        super(Decoder, self).__init__()
        self.dim_hidden = dim_hidden
        self.linear_in = nn.Sequential(
            nn.Linear(dim_hidden, 256, bias=True),
            nn.ELU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=(5, 5), padding=(4, 4)),
            nn.ELU(),
            Interpolate(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(2, 2)),
            nn.ELU(),
            Interpolate(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, kernel_size=(3, 3), padding=(2, 2)),
            nn.ELU(),
            nn.Conv2d(16, 1, kernel_size=(3, 3), padding=(2, 2))
        )

    def forward(self, x):
        x = self.linear_in(x)
        x = x.view(-1, 256, 1, 1)
        x = self.conv(x)
        return x


class VAE(nn.Module):
    def __init__(self, dim_hidden):
        super(VAE, self).__init__()
        self.dim_hidden = dim_hidden
        self.encoder = Encoder(self.dim_hidden)
        self.decoder = Decoder(self.dim_hidden)
        self.epsilon = 10e-7

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var) + self.epsilon
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD