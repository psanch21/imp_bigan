from abc import ABC, abstractmethod

import numpy as np
import torch
# Variable is like Placeholder, it indicates where to stop backpropagation
from torch.autograd import Variable

import utils.constants as cte
from utils.aux import create_dir
from utils.aux import cuda
from utils.fid.fid_object import FID
from utils.fid.fid_object_fmnist import FID_FM


class BaseTrainer(ABC):
    def __init__(self, model, dataset, scaler, args, clip=5, save_every=10, n_critic=5, batch_size=128):
        self.model = model
        self.model.cuda()
        self.__args = args
        self.dataset = dataset
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.init_epoch = 1

        self.clip = clip
        self.save_every = save_every

        self.n_critic = n_critic
        self.batch_size = batch_size

        if dataset != cte.FMNIST:
            self.fid = FID(dims=2048)
        else:
            self.fid = FID_FM()
        self.best_fid = 5000

        self.scaler = scaler

        self.ckpt_name = None
        self.ckpt_file = None

        self.save_dir = None

        self.samples_dir = None

        super().__init__()

    @abstractmethod
    def get_complete_name(self):
        pass

    @abstractmethod
    def train_epoch(self, train_loader):
        pass

    @abstractmethod
    def train(self, train_set, valid_set, n_epochs):
        pass

    def init_folders(self):
        create_dir(self.save_dir)
        create_dir(self.samples_dir)

    def get_params(self):
        return {'batch_size': self.batch_size,
                'ckpt_file': self.ckpt_file,
                'clip': self.clip,
                'best_fid': self.best_fid,
                'save_dir': self.save_dir,
                'samples_dir': self.samples_dir,
                'n_critic': self.n_critic}

    def get_state_dict(self, epoch):
        state_dict = {'epoch': epoch,
                      'model': self.model.get_state_dict(),
                      'params': self.get_params(),
                      'dataset': self.dataset,
                      'args': self.__args}
        return state_dict

    def save(self, dict_params, epoch=-1):
        if epoch == -1:
            torch.save(dict_params, self.ckpt_file)
            print('Saved model to ' + self.ckpt_file)
        else:
            ckpt_file = self.ckpt_file.split('/')[-1].split('.')[0] + '_{}'.format(epoch) + '.pth'
            # torch.save(dict_params, os.path.join(self.save_dir, ckpt_file))
            print('NOT Saved model to ' + ckpt_file)

    def load_checkpoint(self, model_dict):
        self.model.load_checkpoint(model_dict['model'])
        self.init_epoch = model_dict['epoch'] + 1

    def preprocess_batch(self, x):
        data = Variable(x).float()
        data = cuda(data)
        return data

    def compute_fid(self):
        self.model.eval()

        x_gener, z = self.model.sample(5000)
        x_gener = self.process_output(x_gener)
        fid_score = self.fid.compute2(x_gener)

        self.model.train()
        return fid_score

    def process_output(self, x):
        dims = x.shape
        x = self.scaler.inverse_transform(x.reshape(dims[0], -1)).reshape(dims).astype(int)
        x = np.transpose(x, [0, 2, 3, 1])
        return x

    def save_images(self, z):
        self.model.eval()
        x_g = self.model.sample2(z, tensor=True)

        x_gener = self.process_output(x_g.data.cpu().numpy())

        h, w, _ = x_gener.shape[1:]
        self.model.train()
        return (x_g + 1.) / 2.
