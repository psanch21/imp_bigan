from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.autograd import Variable

from utils.aux import cuda


# %% GAN
class GAN(ABC):
    def __init__(self, z_dim):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.G = None
        self.D = None
        self.z_dim = z_dim
        super().__init__()

    @abstractmethod
    def get_complete_name(self):
        pass

    def preprocess_batch(self, x):
        data = Variable(x).float()
        data = cuda(data)
        return data

    def sample_z(self, bs):
        z = torch.randn((bs, self.z_dim))
        z = cuda(z)
        z = Variable(z)
        return z

    def sample(self, bs):
        x, z = self.sample4(bs)
        return x, z

    def sample2(self, z, tensor=False):
        z = torch.tensor(z)
        z = Variable(z).detach()
        z = cuda(z)
        x = self.G(z)
        if not tensor:
            return x.data.cpu().numpy()
        else:
            return x

    def sample3(self, z):
        z = torch.tensor(z, dtype=torch.float)
        z = Variable(z).detach()
        z = cuda(z)
        x = self.G(z)
        x = x.data.cpu().numpy()
        return x

    def sample4(self, bs):
        # It’s more efficient than any other autograd setting - it will use
        # the absolute minimal amount of memory to evaluate the model. volatile
        # also determines that requires_grad is False.
        n_bs = bs // 64 + 1
        x_list = list()
        z_list = list()
        for i in range(n_bs):
            z = Variable(torch.randn(64, self.z_dim)).detach()
            z = cuda(z)
            x = self.G(z)
            x_list.extend(x.cpu().data.numpy())
            z_list.extend(z.cpu().data.numpy())
        return np.array(x_list[:bs]), np.array(z_list[:bs])

    def cuda(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def parameters(self):
        pass

    def load_state_dict(self, model_dict):
        pass

    def load(self, ckpt_file, model_dict):
        pass
