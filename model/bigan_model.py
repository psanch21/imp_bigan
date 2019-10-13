import os
import shutil

import numpy as np
import probvis.aux as pva
import probvis.general as pvg
import probvis.images as pvi
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import utils.constants as cte
import utils.data.functions.aux as data_fn
import utils.loss as loss
import utils.tensorboard as tbu
import utils.tensorboard as utb
from model.base.model import GAN
from model.networks.discriminator import DiscriminatorJoin, DiscriminatorJoinDeep
from model.networks.generator import Generator, GeneratorDeep
from model.networks.inference import Inference, InferenceDeep
from utils.aux import cuda, create_dir
from utils.loss import WassersteinLoss, HingeLoss


# %% Main Module


class BiGAN(GAN):
    def __init__(self, z_dim=256, out_size=32, image_channels=3, loss_type=cte.HINGE, bigan_type=cte.BiGAN,
                 deep=False):
        super().__init__(z_dim)
        self.name = bigan_type
        self.z_dim = z_dim
        self.image_channels = image_channels
        self.image_size = out_size
        self.loss_type = loss_type
        self.bigan_type = bigan_type
        self.type_recons_loss = cte.L2
        self.deep = deep

    def init(self):
        if self.deep:
            self.G = GeneratorDeep(self.z_dim, n_channels=self.image_channels, image_size=self.image_size)
            self.D = DiscriminatorJoinDeep(n_channels=self.image_channels, image_size=self.image_size, z_dim=self.z_dim)
            self.E = InferenceDeep(out_size=self.image_size, n_channels=self.image_channels, z_dim=self.z_dim)
        else:
            self.G = Generator(self.z_dim, n_channels=self.image_channels, image_size=self.image_size)
            self.D = DiscriminatorJoin(n_channels=self.image_channels, image_size=self.image_size, z_dim=self.z_dim)
            self.E = Inference(out_size=self.image_size, n_channels=self.image_channels, z_dim=self.z_dim)
        # self.G.reset_parameters()
        # self.D.reset_parameters()
        self.loss = self.set_loss(self.loss_type)

    def get_name(self):
        return self.name

    def get_complete_name(self):
        d = 'd' if self.deep else 's'
        return '{}_{}z_{}{}'.format(self.name, self.z_dim, self.loss_type, d)

    def set_loss(self, loss_type):
        if loss_type == cte.WASSERSTEIN:
            return WassersteinLoss()
        elif loss_type == cte.HINGE:
            return HingeLoss()
        elif loss_type == cte.BCE:
            return nn.BCEWithLogitsLoss()

    def load_checkpoint_trbigan(self, model_dict):
        print('Loading trbigan...')
        self.G.load_state_dict(model_dict['state_dict_G'])
        self.D.load_state_dict(model_dict['state_dict_D'])
        self.E.load_state_dict(model_dict['state_dict_E'])

    def load_checkpoint(self, model_dict):
        self.set_params(model_dict['params'])
        self.init()
        self.G.load_state_dict(model_dict['state_dict_G'])
        self.D.load_state_dict(model_dict['state_dict_D'])
        self.E.load_state_dict(model_dict['state_dict_E'])

    def set_params(self, params_dict):
        self.loss_type = params_dict['loss_type']
        self.z_dim = params_dict['z_dim']
        self.image_size = params_dict['image_size']
        self.image_channels = params_dict['image_channels']
        self.type_recons_loss = params_dict['type_recons_loss']
        self.bigan_type = params_dict['bigan_type']
        self.name = params_dict['name']

    def get_state_dict(self):
        return {'state_dict_G': self.G.state_dict(),
                'state_dict_D': self.D.state_dict(),
                'state_dict_E': self.E.state_dict(),
                'params': self.get_params()}

    def get_params(self):
        return {'loss_type': self.loss_type,
                'z_dim': self.z_dim,
                'image_size': self.image_size,
                'image_channels': self.image_channels,
                'type_recons_loss': self.type_recons_loss,
                'bigan_type': self.bigan_type,
                'name': self.name}

    def cuda(self):
        if torch.cuda.is_available():
            self.G.cuda()
            self.D.cuda()
            self.E.cuda()
            self.loss.cuda()

    def train(self):
        self.G.train()
        self.D.train()
        self.E.train()

    def eval(self):
        self.G.eval()
        self.D.eval()
        self.E.eval()

    def parameters(self):
        params = list()
        params.extend(self.G.parameters())
        params.extend(self.D.parameters())
        params.extend(self.E.parameters())
        return params

    def compute_d_loss(self, real_logit, real_labels, fake_logit, fake_labels):
        if self.loss_type == cte.BCE:
            d_loss_real = self.loss(real_logit, real_labels)  # -log(D(x))
            d_loss_fake = self.loss(fake_logit, fake_labels)  # -log(1 - D(x))
            d_loss = d_loss_real + d_loss_fake  # - (log(D(x))+ log(1-D(x_fake)))
        else:
            d_loss = self.loss(real_logit, fake_logit)
        return d_loss

    def compute_ge_loss(self, real_logit, fake_logit, fake_labels, real_labels):
        if self.loss_type == cte.WASSERSTEIN or self.loss_type == cte.HINGE:
            g_loss = -self.loss(real_logit, fake_logit)
        else:
            g_loss_real = self.loss(real_logit, fake_labels)  # -log(1 - D(x))
            g_loss_fake = self.loss(fake_logit, real_labels)  # -log(D(x_fake))
            g_loss = g_loss_real + g_loss_fake
        return g_loss

    def train_D(self, x):
        bs = x.size(0)

        real_labels = cuda(Variable(torch.ones(bs)))
        fake_labels = cuda(Variable(torch.zeros(bs)))

        z_e = self.E(x)
        real_logit = self.D(x, z_e)

        z = self.sample_z(bs)

        x_fake = self.G(z)
        fake_logit = self.D(x_fake, z)
        d_loss = self.compute_d_loss(real_logit, real_labels, fake_logit, fake_labels)
        return d_loss

    def train_GE(self, x, l_recons=None, l_norm=None):
        """

        :param x:
        :param l_recons:
        :param l_norm:
        :return:
        """
        bs = x.size(0)

        real_labels = cuda(Variable(torch.ones(bs)))
        fake_labels = cuda(Variable(torch.zeros(bs)))

        z_e = self.E(x)
        real_logit = self.D(x, z_e)

        z = self.sample_z(bs)

        x_fake = self.G(z)
        fake_logit = self.D(x_fake, z)
        z_norm = torch.sqrt(torch.sum(z_e ** 2, dim=-1))

        g_loss = self.compute_ge_loss(real_logit, fake_logit, fake_labels, real_labels)
        g_loss_out = g_loss.data.cpu().numpy()
        if self.bigan_type != cte.BiGAN:
            x_recons = self.G(z_e)
            x_recons_loss = self.get_x_recons_loss(x, x_recons)
            g_loss += l_recons * x_recons_loss.mean()

        if self.bigan_type in [cte.PMDGAN, cte.MLPMDGAN, cte.EPMDGAN]:
            z_e_loss = self.get_ze_loss(z_e)
            g_loss += l_norm * z_e_loss.mean()

        return g_loss_out, z_norm, g_loss

    def get_x_recons_loss(self, x, x_recons):
        if self.type_recons_loss == cte.L1:
            return torch.sqrt(torch.mean(torch.abs(x_recons - x), dim=-1).mean(-1).mean(-1))
        elif self.type_recons_loss == cte.L2:
            return torch.sqrt(torch.mean((x_recons - x) ** 2, dim=-1).mean(-1).mean(-1))

    def get_ze_loss(self, z_e):
        # return torch.abs(torch.sqrt(torch.sum(z_e ** 2, dim=-1)) - np.sqrt(self.z_dim))
        norm_loss = (torch.sqrt(torch.sum(z_e ** 2, dim=-1)) - np.sqrt(self.z_dim)) ** 2
        # norm_loss = (torch.std(z_e) - 1) ** 2
        return norm_loss / np.sqrt(self.z_dim)

    def recons(self, x_input):
        n_imgs = x_input.shape[0]
        x_list = list()
        z_list = list()
        n_batches = int(np.ceil(n_imgs / 128))
        for i in range(n_batches):
            x = torch.tensor(x_input[i * 128:(i + 1) * 128]).float()
            x = cuda(x)
            z = self.E(x)
            x_recons = self.G(z)
            x_recons = x_recons.data.cpu().numpy()
            x_list.extend(x_recons)
            z_list.extend(z.data.cpu().numpy())

        z_list = np.array(z_list)
        x_list = np.array(x_list)
        return z_list, x_list

    def psnr(self, loader_sorted):
        psnr_list = list()
        for i, x in enumerate(loader_sorted):
            x = cuda(x)
            z = self.E(x)
            x_recons = self.G(z)
            psnr = loss.PSNR_torch(x, x_recons, axis=(-3, -2, -1))
            psnr = psnr.data.cpu().numpy()
            psnr_list.extend(psnr)

        return np.array(psnr_list)

    def get_z(self, x_input):
        n_imgs = x_input.shape[0]
        z_list = list()
        n_batches = int(np.ceil(n_imgs / 128))
        for i in range(n_batches):
            x = torch.tensor(x_input[i * 128:(i + 1) * 128]).float()
            x = cuda(x)
            z = self.E(x)
            z_list.extend(z.data.cpu().numpy())

        z_list = np.array(z_list)
        return z_list

    # %% Plot methods
    def plot_generation(self, base_dir, scaler, n_col=16, n_row=8, n_bs=3):
        save_dir = create_dir(os.path.join(base_dir, 'gener'))

        x_gener_norm, z = self.sample(512)
        x_gener = data_fn.denorm(x_gener_norm, scaler)

        z_recons, x_recons_norm = self.recons(x_gener_norm)
        x_recons = data_fn.denorm(x_recons_norm, scaler)

        h, w = x_gener.shape[1:3]
        n_imgs = n_col * n_row
        for i in range(n_bs):
            x = x_gener[i * n_imgs:(i + 1) * n_imgs]
            f, _ = pvi.plot_image(pvi.merge_images(x, n_row, n_col, direction=0, dtype=int), title='')

            pva.save_fig(f, os.path.join(save_dir, 'x_gener_{}'.format(i)))

            ze_mean = np.mean(z_recons)
            ze_var = np.var(z_recons)
            # print('E(x) distribution stats: {:.2} +- {:.2}'.format(ze_mean, ze_var))

            # result.plot_real_recons(save_dir, x_gener, x_recons, n_imgs=20, n_bs=2, train_str='tr', shuffle=False)
            # result.plot_hist_ze_with_prior(save_dir, z_train=z_list)

    # %% Plot methods
    def plot_reconstruction(self, base_dir, scaler, x_norm):
        save_dir = create_dir(os.path.join(base_dir, 'recons', 'tb'))
        shutil.rmtree(save_dir)
        writer = SummaryWriter(log_dir=save_dir)

        n_imgs, h, w, c = x_norm.shape
        z_list, x_recons_norm = self.recons(x_norm)
        x_recons = data_fn.denorm(x_recons_norm, scaler)
        x = data_fn.denorm(x_norm, scaler)

        writer.add_embedding(mat=z_list, label_img=torch.from_numpy(x.transpose((0, 3, 1, 2)) / 255.),
                             tag='latent_space')
        ze_mean = np.mean(z_list, axis=0)
        ze_var = np.var(z_list, axis=0)

        z_norm = np.linalg.norm(z_list, axis=-1)
        z_norm_prior = np.linalg.norm(np.random.normal(0, 1, [n_imgs, self.z_dim]), axis=1)

        writer.add_histogram(tag='latent_space/norm', values=z_norm, global_step=1)
        writer.add_histogram(tag='latent_space/norm', values=z_norm_prior, global_step=2)
        for i in range(self.z_dim):
            writer.add_scalar(tag='latent_space/mean', scalar_value=ze_mean[i], global_step=i)
            writer.add_scalar(tag='latent_space/var', scalar_value=ze_var[i], global_step=i)

        n_bs = n_imgs // 64
        for i in range(n_bs):
            # x = np.concatenate([x_data[i * n_imgs:(i + 1) * n_imgs], x_recons[i * n_imgs:(i + 1) * n_imgs]], axis=0)
            x_i = tbu.images_to_tensor_grid(x_recons[i * 64: (i + 1) * 64])

            writer.add_image('x/reconstruction', x_i, i, dataformats='CHW')

            x_i = tbu.images_to_tensor_grid(x[i * 64: (i + 1) * 64])

            writer.add_image('x/original', x_i, i, dataformats='CHW')

        psnr = loss.PSNR(x, x_recons, axis=(1, 2, 3))
        writer.add_histogram(tag='psnr/histogram', values=psnr)
        idx_sorted = np.argsort(psnr)  # From smaller to larger
        # psnr_sorted = psnr[idx_sorted]

        for i in range(len(psnr)):
            writer.add_scalar(tag='psnr/original', scalar_value=psnr[i], global_step=i)
            writer.add_scalar(tag='psnr/idx', scalar_value=idx_sorted[i], global_step=i)
            writer.add_scalar(tag='psnr/sorted', scalar_value=psnr[idx_sorted[i]], global_step=i)

        x_sorted = x[idx_sorted]

        n_bs = n_imgs // 64
        for i in range(n_bs):
            # x = np.concatenate([x_data[i * n_imgs:(i + 1) * n_imgs], x_recons[i * n_imgs:(i + 1) * n_imgs]], axis=0)

            x_i = tbu.images_to_tensor_grid(x_sorted[i * 64: (i + 1) * 64])

            writer.add_image('x/psnr_sorted', x_i, i, dataformats='CHW')

        # Save images

        save_dir = create_dir(os.path.join(base_dir, 'recons', 'images'))

        pvg.mean_var_plot(save_dir, z_list, name='z', xlabel=r'Dimension', ylabel='')
        pvg.multi_hist_plot(save_dir, [z_norm_prior, z_norm], [r'$||z||$', r'$||z_e||$'], name='ze', xlabel=r'Norm',
                            density=True, alpha=0.7, fontsize=32)

        # %% Plot methods

    def plot_reconstruction_fig(self, base_dir, scaler, x_norm, im_idx=[1, 2, 3, 4, 5]):
        n_imgs, c, h, w = x_norm.shape
        z_list, x_recons_norm = self.recons(x_norm)
        x_recons = data_fn.denorm(x_recons_norm, scaler)
        x = data_fn.denorm(x_norm, scaler)

        ze_mean = np.mean(z_list, axis=0)
        ze_var = np.var(z_list, axis=0)

        z_norm = np.linalg.norm(z_list, axis=-1)
        z_norm_prior = np.linalg.norm(np.random.normal(0, 1, [n_imgs, self.z_dim]), axis=1)

        n_bs = n_imgs // 64

        psnr = loss.PSNR(x, x_recons, axis=(1, 2, 3))
        idx_sorted = np.argsort(psnr)  # From smaller to larger
        # psnr_sorted = psnr[idx_sorted]

        x_sorted = x[idx_sorted]

        n_bs = n_imgs // 64

        # Save images
        save_dir = create_dir(os.path.join(base_dir, 'recons', 'images'))

        pvg.mean_var_plot(save_dir, z_list, name='z', xlabel=r'Dimension', ylabel='')
        pvg.var_plot(save_dir, z_list, name='z', xlabel=r'Dimension')

        binwidth = (np.max(z_norm_prior) - np.min(z_norm_prior)) / 100
        x_min = min(np.min(z_norm_prior), np.min(z_norm)) - 1

        x_max = max(np.percentile(z_norm, q=90), np.max(z_norm_prior)) + 1
        pvg.multi_hist_plot(save_dir, [z_norm_prior, z_norm], [r'$||z||$', r'$||z_e||$'], name='ze', xlabel=r'Norm',
                            density=False, alpha=0.7, fontsize=38, binwidth=binwidth, x_lim=[x_min, x_max])
        mean = np.mean(z_list, axis=0)
        var = np.var(z_list, axis=0)

        pvg.multi_hist_plot(save_dir, [mean, var], [r'$\mu$', r'$\sigma^2$'], name='z_mean_var', xlabel=r'value',
                            ylabel=r'$\#$ of dimensions',
                            density=False, alpha=0.7, fontsize=32, color_list=['black', 'green'])

        save_dir = create_dir(os.path.join(base_dir, 'recons', 'images', 'batch'))

        n_imgs = 64
        for i in range(5):
            x_i = x_recons[i * n_imgs:(i + 1) * n_imgs]
            f, _ = pvi.plot_image(pvi.merge_images(x_i, 8, 8, direction=0, dtype=int), title='')

            pva.save_fig(f, os.path.join(save_dir, 'x_{}_recons'.format(i)))

            x_i = x[i * n_imgs:(i + 1) * n_imgs]
            f, _ = pvi.plot_image(pvi.merge_images(x_i, 8, 8, direction=0, dtype=int), title='')

            pva.save_fig(f, os.path.join(save_dir, 'x_{}_real'.format(i)))

        x_i = np.zeros((2 * len(im_idx), h, w, c))
        half = len(im_idx)
        x_i[:half] = x[im_idx]
        x_i[half:] = x_recons[im_idx]

        f, _ = pvi.plot_image(pvi.merge_images(x_i, 2, half, direction=1, dtype=int), title='')

        pva.save_fig(f, os.path.join(save_dir, 'x_rec_real'))

    def plot_loglikelihod(self, base_dir, scaler, x_norm, T=40):
        """

        :param base_dir: i.e. experiments/fm/6/trbiganp_256z_bces_fm_60r_0n_08p16d/test/
        :param scaler:
        :param x_norm:
        :return:
        """
        save_dir = create_dir(os.path.join(base_dir, 'loglikelihood', str(T), 'images'))
        n_imgs, h, w, c = x_norm.shape
        try:
            event_acc = utb.load_event_accumulator(os.path.join(base_dir, 'loglikelihood', str(T)))
            ll_list, sigma_list = utb.load_loglikelihood(event_acc)
        except:
            print('No loglikelihood data at: {} '.format(os.path.join(base_dir, 'loglikelihood', str(T))))
            return

        x_norm = x_norm[:len(ll_list)]
        z_list, x_recons_norm = self.recons(x_norm)
        x_recons = data_fn.denorm(x_recons_norm, scaler)
        x = data_fn.denorm(x_norm, scaler)

        psnr = loss.PSNR(x, x_recons, axis=(1, 2, 3))

        pvg.hist_plot(save_dir, ll_list, density=False, name='ll', alpha=0.8, xlabel=r'$\log_{10} p(G(E(x)))$',
                      fontsize=32, ylabel=r'$\#$ of images', close='all')

        pvg.scater_plot_with_images(save_dir, x_data=psnr, y_data=ll_list, images=x, name='psnr_ll', alpha=0.8,
                                    xlabel=r'PSNR', ylabel=r'$\log_{10} p(G(E(x)))$')

        model = LinearRegression()

        psnr = np.array(psnr).reshape((-1, 1))
        ll_list = np.array(ll_list)

        model.fit(psnr, ll_list)

        r_sq = model.score(psnr, ll_list)
        print('coefficient of determination: {} %'.format(r_sq * 100))
        print('intercept:', model.intercept_)
        print('slope:', model.coef_)
