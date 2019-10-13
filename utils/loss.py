import numpy as np
import torch
import torch.nn as nn


# %% Loss
def MSE(x_test, x_recons, axis=None):
    return np.mean((x_test - x_recons) ** 2, axis)


def PSNR(x_test, x_recons, axis=None, max_value=255):
    mse = MSE(x_test, x_recons, axis)
    return 10 * np.log10(max_value ** 2 / mse)


def MSE_torch(x_test, x_recons, axis=None):
    assert type(axis) in [int, tuple]
    return torch.mean((x_test - x_recons) ** 2, dim=axis)


def PSNR_torch(x_test, x_recons, axis=-1):
    mse = MSE_torch(x_test, x_recons, axis)
    return 10 * torch.log10(255 ** 2 / mse)


# %% Gaussian emission

class Loss():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""
        log_term = torch.sum(2 * torch.log(std_2) - 2 * torch.log(std_1))
        mean_term = torch.sum((mean_1 - mean_2).pow(2) / std_2.pow(2))
        trace_term = torch.sum(std_1.pow(2) / std_2.pow(2))

        return 0.5 * (log_term + mean_term + trace_term - 1)

    def nll_bernoulli(self, theta, x):
        return - torch.sum(x * torch.log(theta) + (1 - x) * torch.log(1 - theta))

    def nll_gauss(self, mean, std, x):
        var = std ** 2
        # Sum all dimensions and all time instants.
        cte = np.log(2 * np.pi) * torch.ones(x.shape, dtype=torch.float)
        cte = cte.to(self.device)

        # Sum all dimensions and all time instants.
        reconstruction = cte - 0.5 * torch.log(var) - 0.5 * torch.div((mean - x) ** 2, var)
        return -torch.sum(reconstruction)

    def kld_cat(self, prob_1, prob_2):
        """Using std to compute KLD"""
        logs = torch.log(prob_1 / prob_2)
        kl = torch.sum(prob_1 * logs)
        return kl


class WassersteinLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logit_real, logit_fake):
        return - logit_real.mean() + logit_fake.mean()


class HingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logit_real, logit_fake):
        return nn.ReLU()(1.0 - logit_real).mean() + nn.ReLU()(1.0 + logit_fake).mean()
