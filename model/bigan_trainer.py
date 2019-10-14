import itertools
import os
import time

import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils.constants as cte
import utils.log_likelihood as ll_fn
import utils.loss as loss_fn
from model.base.trainer import BaseTrainer
from utils.aux import cuda
from utils.data.functions.aux import get_numpy_from_dataset
from utils.data.functions.custom_dataset import Dataset
from utils.fid.fid_object import FID
from utils.fid.fid_object_fmnist import FID_FM


class SNBiGANTrainer(BaseTrainer):

    def __init__(self, model, dataset, scaler, args, batch_size=128, clip=5, save_every=10,
                 ckpt_folder='exper', l_rate=2e-4,
                 n_critic=5, probs=None, l_recons=0, l_norm=0, l_perc=0.0, l_dis=0, beta1=0.0, beta2=0.9):
        super().__init__(model, dataset, scaler, args, clip, save_every, n_critic, batch_size)
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

        self.scaler = scaler

        self.l_rate = l_rate

        # Initialize optimizers

        self.optimizer_d = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                   model.D.parameters()), lr=l_rate, betas=(beta1, beta2))

        self.optimizer_g = torch.optim.Adam(itertools.chain(model.G.parameters(), model.E.parameters()),
                                            lr=l_rate, betas=(beta1, beta2))

        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_d, gamma=0.99)
        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_g, gamma=0.99)

        # Initialize FID

        if self.dataset != cte.FMNIST:
            self.fid = FID(dims=2048)
        else:
            self.fid = FID_FM()
        self.best_fid = 5000

        # Initialize regularization hyperparameters
        if probs is not None:
            assert (
                    l_perc > 0 or l_dis > 0), 'ERROR: Necessary parameters to perform non-uniform sampling: probs, l_perc, l_dis'
        self.set_nonuni_params(probs, l_perc, l_dis)
        self.l_recons_current = l_recons
        self.l_recons = l_recons
        self.l_norm = l_norm if model.get_name() in cte.TRBiGAN_LIST else 0.0
        self.adap_l_norm = True if model.get_name() in cte.TRBiGAN_LIST else False

        # Initialize files and folders
        self.ckpt_name = self.get_complete_name()
        self.ckpt_file = os.path.join(ckpt_folder, self.ckpt_name + '.pth')

        self.save_dir = os.path.join(ckpt_folder, self.ckpt_name)
        self.writer = SummaryWriter(log_dir=self.save_dir)
        self.samples_dir = os.path.join(self.save_dir, 'samples_train')
        self.init_folders()
        self.bigan_type = model.bigan_type

        self.data_batch = None
        self.d_batch_on = False

    def get_complete_name(self):
        r = '{}'.format(self.l_recons).replace('.', '')
        n = '1' if self.l_norm > 0 else '0'
        p = '{}'.format(self.l_perc).replace('.', '')
        probs_str = '{}p{}d'.format(p, self.l_dis)
        return '{}_{}_{}r_{}n_{}'.format(self.model.get_complete_name(), self.dataset, r, n, probs_str)

    def get_params(self):
        base_params = super().get_params()
        self_params = {'l_rate': self.l_rate,
                       'l_recons': self.l_recons,
                       'l_norm': self.l_norm,
                       'probs': self.probs}
        return {**base_params, **self_params}

    def get_state_dict(self, epoch):
        base_state_dict = super().get_state_dict(epoch)
        self_state_dict = {'optimizer_g': self.optimizer_g.state_dict(),
                           'optimizer_d': self.optimizer_d.state_dict(),
                           'params': self.get_params()}
        return {**base_state_dict, **self_state_dict}

    def set_nonuni_params(self, probs, l_perc, l_dis):
        self.probs = probs
        self.l_perc = l_perc
        self.l_dis = l_dis
        if probs is not None:
            print('Ratio Prmax/Prmin: {}'.format(np.max(probs) / np.min(probs)))

    def load_checkpoint(self, model_dict):
        super().load_checkpoint(model_dict)
        self.optimizer_g.load_state_dict(model_dict['optimizer_g'])
        self.optimizer_d.load_state_dict(model_dict['optimizer_d'])
        self.model.cuda()

    def get_data_batch(self, data_curr):
        if not self.d_batch_on:
            return data_curr

        if self.bigan_type in [cte.EPMDGAN] and self.data_batch is None:
            self.data_batch = cuda(data_curr)
            return data_curr

        data = torch.cat((cuda(data_curr), self.data_batch), 0)

        self.model.eval()
        data_recons = self.model.G(self.model.E(data))
        psnr = loss_fn.PSNR_torch(data, data_recons, axis=(-3, -2, -1))
        idx_sorted = psnr.argsort(descending=False)  # From smaller to larger
        self.model.train()
        half_batch_size = self.batch_size // 3
        if np.random.uniform() > self.l_perc:
            self.data_batch = cuda(data_curr)
        else:
            self.data_batch = data[idx_sorted[:half_batch_size]]
        self.model.train()
        return data[idx_sorted[:self.batch_size]]  # Keep the images with the worst reconstruction

    def train_epoch(self, train_loader):
        train_iter = iter(train_loader)

        for i in tqdm(range(len(train_iter))):  # for i in tqdm(range(5)):

            data = self.get_data_batch(next(train_iter))
            imgs = self.preprocess_batch(data)
            self.optimizer_d.zero_grad()  # Set gradients to zero otherwise they accumulate
            self.optimizer_g.zero_grad()  # Set gradients to zero otherwise they accumulate

            d_loss = self.model.train_D(imgs)
            d_loss.backward()  # This accumulates gradients
            self.optimizer_d.step()

            if (i + 1) % self.n_critic == 0:
                self.optimizer_g.zero_grad()  # Set gradients to zero otherwise they accumulate
                self.optimizer_d.zero_grad()  # Set gradients to zero otherwise they accumulate

                g_loss, z_norm, ge_loss = self.model.train_GE(imgs, self.l_recons_current, self.l_norm)

                ge_loss.backward()  # This accumulates gradients
                self.optimizer_g.step()

        d_loss = float(d_loss.data.cpu().numpy())
        g_loss = float(g_loss)
        ge_loss = float(ge_loss.data.cpu().numpy())
        z_norm = z_norm.data.cpu().numpy()

        return d_loss, (g_loss, ge_loss), z_norm

    def update_scheduler(self, epoch):
        if epoch > 10:
            self.scheduler_d.step()
            self.scheduler_g.step()

    def train(self, train_set, valid_set, n_epochs):
        if self.init_epoch >= (n_epochs + 1): return
        train_loader = self.get_data_loader(train_set, istrain=True)
        train_loader_sorted = torch.utils.data.DataLoader(Dataset(train_set), batch_size=self.batch_size, shuffle=False,
                                                          pin_memory=True)
        self.model.train()
        x_valid_norm = get_numpy_from_dataset(valid_set, n_samples=5000)
        # x_train_norm = get_numpy_from_dataset(train_set)
        x_valid = self.process_output(x_valid_norm[:5000])
        self.fid.save_stats_real(x_valid)
        z = self.model.sample_z(64)
        start = time.time()
        # fid_score = self.compute_fid()
        # print('Current FID: {} Best FID: {}'.format(fid_score, self.best_fid))
        for epoch in range(self.init_epoch, n_epochs + 1):
            end = time.time()
            print('{} min Epoch {}/{} l_norm:{:.2f}'.format((end - start) // 60, epoch, n_epochs, self.l_norm))

            last_epoch = epoch == n_epochs
            if epoch > 10:
                if self.l_dis == 0:
                    self.d_batch_on = False  # self.d_batch_on = True

                elif self.l_dis > 0 and self.bigan_type in [cte.EPMDGAN]:
                    self.probs = self.get_probs_psnr(train_loader_sorted)
                    train_loader = self.get_data_loader(train_set, istrain=True)

            d_loss, (g_loss, ge_loss), z_norm = self.train_epoch(train_loader)
            z_n_mean = np.mean(z_norm)
            z_n_var = np.var(z_norm)
            # self.update_l_recons(epoch)
            if self.adap_l_norm and epoch > 200 and epoch % 5 == 0:
                x_train_norm_i = get_numpy_from_dataset(train_set, n_samples=10000, shuffle=True)
                self.update_l_norm(epoch, x_train_norm_i, init_epoch=200)
            self.update_scheduler(epoch)
            # saving model
            if epoch % 100 == 0:
                self.save(self.get_state_dict(epoch), epoch=epoch)
            if epoch % self.save_every == 0 or last_epoch:
                imgs = self.save_images(z)
                grid = torchvision.utils.make_grid(imgs)
                self.writer.add_image('generated', grid, epoch, dataformats='CHW')
                self.save(self.get_state_dict(epoch))
                fid_score = self.compute_fid()
                print('Current FID: {} Best FID: {}'.format(fid_score, self.best_fid))
                self.writer.add_scalar('Evaluation/fid', fid_score, epoch)

                if fid_score <= self.best_fid:
                    self.best_fid = fid_score
                    print('Updating best FID: {}'.format(self.best_fid))

            print('\nEpoch: {}/{}  | d_loss: {:.2f} | g_loss: {:.2f} | z_norm: {:.2f}'.format(
                epoch, n_epochs, d_loss, g_loss, z_n_mean))

            self.writer.add_scalar('Loss/discriminator', d_loss, epoch)
            self.writer.add_scalar('Loss/generator', g_loss, epoch)
            self.writer.add_scalar('Loss/generator_encoder', ge_loss, epoch)
            self.writer.add_histogram('z_encoded_norm', z_norm, epoch)
            self.writer.add_scalar('z_encoded/mean', z_n_mean, epoch)
            self.writer.add_scalar('z_encoded/var', z_n_var, epoch)
        self.writer.close()

    def update_l_recons(self, epoch, init_epoch=10):
        print('l_c: {} l_f: {}'.format(self.l_recons_current, self.l_recons))
        if self.l_recons > 0 and epoch > init_epoch:
            inc = epoch / 200
            self.l_recons_current = min(self.l_recons, self.l_recons * inc)

    def update_l_norm(self, epoch, x_valid, init_epoch=200):

        self.model.eval()

        z_e = self.model.get_z(x_valid)
        z_dim = z_e.shape[-1]

        z_norm = np.linalg.norm(z_e, axis=-1)

        z_norm_mean = np.mean(z_norm)
        z_norm_std = np.std(z_norm)

        z_norm_p = np.linalg.norm(np.random.normal(0, 1, [1000, z_dim]), axis=1)
        z_norm_p_mean = np.sqrt(z_dim)
        z_norm_p_std = np.std(z_norm_p)

        print('Mean norm: {:.2f} Std norm: {:.2f}'.format(z_norm_mean, z_norm_std))
        print('Mean norm p: {:.2f} Std norm p: {:.2f}'.format(z_norm_p_mean, z_norm_p_std))
        cond_std = z_norm_std - 1.01 * z_norm_p_std > 0
        if cond_std:
            inc = max(0.01, init_epoch / epoch / 10)
            new_l2 = min(self.l_norm + inc, 1.0)
            print('Update l_norm: {:.2f} --> {:.2f}'.format(self.l_norm, new_l2))
            self.l_norm = new_l2

        cond_std = z_norm_std - 0.98 * z_norm_p_std < 0

        if cond_std:
            inc = max(0.01, init_epoch / epoch / 10)
            new_l2 = max(self.l_norm - inc, 0.001)
            print('Update l_norm: {:.2f} --> {:.2f}'.format(self.l_norm, new_l2))
            self.l_norm = new_l2

        self.model.train()

        return

    def get_data_loader(self, x, istrain):
        shuffle = False
        if istrain and self.l_perc > 0 and self.probs is not None:
            print('Non-uniform mini-batch sampling ON!: l_perc: {}'.format(self.l_perc))
            data = Dataset(x, probs=self.probs, perc=self.l_perc)
        else:
            shuffle = True
            data = Dataset(x)

        num_workers = 4 if self.dataset in [cte.CELEBA, cte.LSUN] else 2
        data_loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=shuffle, pin_memory=True,
                                                  num_workers=num_workers)

        return data_loader

    def get_probs_psnr(self, train_loader_sorted):
        self.model.eval()
        # z_list, x_recons_norm = self.model.recons(x_train_norm)
        # psnr = loss_fn.PSNR(x_train_norm, x_recons_norm, axis=(-3, -2, -1))
        psnr = self.model.psnr(train_loader_sorted)
        self.model.train()

        idx_sorted = np.argsort(psnr)[::-1]  # From larger to smaller
        probs_ordered = ll_fn.get_probs(len(idx_sorted), self.l_dis)  # From smaller to larger
        probs = np.ones(len(probs_ordered))

        count = 0
        # The one with the larger likelihood is assigned the smaller probability
        for i in idx_sorted:
            probs[i] = probs_ordered[count]
            count = count + 1

        assert np.sum(probs == 1) == 0

        return probs
