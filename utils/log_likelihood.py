import math
import os
import time

import numpy as np
from torch.utils.tensorboard import SummaryWriter

import utils.loss as loss
import utils.tensorboard as utb


def get_probs(length, exp):
    probs = (np.arange(1, length + 1) / 100) ** exp
    last_x = int(0.9 * length)
    probs[last_x:] = probs[last_x]
    return probs / probs.sum()


def get_probs2(length, dist, prop):
    probs = (np.arange(1, length + 1) / 100) ** dist
    last_x = int(0.999 * length)
    probs[last_x:] = probs[last_x]
    return (1 - prop) / length + prop * probs / probs.sum()


def get_probs_from_ll(base_dir, T, l_dis):
    event_acc = utb.load_event_accumulator(os.path.join(base_dir, 'train', 'loglikelihood', str(T)))
    ll_list, sigma_list = utb.load_loglikelihood(event_acc)

    idx_sorted = np.argsort(ll_list)[::-1]  # From larger to smaller
    probs_ordered = get_probs(len(idx_sorted), l_dis)  # From smaller to larger
    probs = np.ones(len(probs_ordered))

    count = 0
    # The one with the larger likelihood is assigned the smaller probability
    for i in idx_sorted:
        probs[i] = probs_ordered[count]
        count = count + 1

    assert np.sum(probs == 1) == 0

    return probs


def log_sum_exp_trick(value_list):
    max_value = np.max(value_list)
    ds = value_list - max_value
    sumOfExp = np.exp(ds).sum()
    return max_value + np.log(sumOfExp)


SIGMA_LIST = np.linspace(0.01, 0.3, 200)


class ApproxLL:
    def __init__(self, model, scaler, folder, N=128, N_max=10000, T=40, sigma_list=None):
        self.folder = os.path.join(folder, 'loglikelihood', str(T))
        self.model = model
        self.scaler = scaler
        self.bs = 256
        self.k = 5

        self.sigma_list = sigma_list if sigma_list else SIGMA_LIST
        n_iter = math.ceil(N / self.bs)
        self.N = n_iter * self.bs

        self.n_sigma = len(self.sigma_list)
        n_iter = math.ceil(N_max / self.bs)
        self.N_max = n_iter * self.bs
        self.T = T

        self.writer = SummaryWriter(log_dir=self.folder)

    def process_output(self, x):
        assert x.shape[1] <= 3, 'Error: Data should be in CHW format'
        dims = x.shape
        x = self.scaler.inverse_transform(x.reshape(dims[0], -1)).reshape(dims).astype(int)
        x = np.transpose(x, [0, 2, 3, 1])  # CHW --> HWC
        return x

    def compute_ll(self, x_recons, z_infer):
        n_imgs, z_dim = z_infer.shape
        for i in range(len(self.sigma_list)):
            self.writer.add_scalar('sigma_list', self.sigma_list[i], i)
        print('Starting analysis non-isotropic | n_imgs: {}'.format(n_imgs))
        if self.N > self.N_max:
            print('Error: N > N_max {} > {}'.format(self.N, self.N_max))

        init_time = time.time()

        # ll_list, sigma_i_list, ll_evolution_list = self.load_data(n_imgs)
        event_acc = utb.load_event_accumulator(self.folder)
        ll, _ = utb.load_loglikelihood(event_acc)

        init_img = max(0, len(ll) - 1)
        for i in range(init_img, n_imgs):
            # if ll_list[i] < 0:
            #     summary_str = '\n[{}]  LL={} | sigma={}'
            #     print(summary_str.format(i, ll_list[i], sigma_i_list[i]))
            #     continue
            cur_time = time.time()
            ll = self.compute_ll_img(x_recons[i], z_infer[i, :], str(i))
            time_total = time.time() - init_time
            time_epoch = time.time() - cur_time

            min_epochs = int(time_epoch / 60)
            summary_str = '\n[{}]Time: {}:{} | Total time: {} log10 = {}'
            print(summary_str.format(i, min_epochs, int(time_epoch), int(time_total / 60), ll))

        print('Analysis non-isotropic completed | n_imgs: {}'.format(n_imgs))
        self.writer.close()
        return

    def compute_ll_img(self, x_recons, z_c, img_idx):
        z_dim = z_c.shape[-1]
        x_tmp = np.tile(x_recons, [self.bs, 1, 1, 1])
        N_i = self.N
        j = 0
        while j < self.n_sigma:
            sigma = self.sigma_list[j]
            accepted_samples_count = 0
            tries = 0
            while accepted_samples_count == 0:
                psnr_tmp, log_ratio_p_q, log_ratio_1_q = self.get_psnr_ratio(z_c, x_tmp, N_i, sigma, z_dim)
                accepted_samples = psnr_tmp > self.T
                accepted_samples_count = np.sum(accepted_samples)
                assert tries < 5, 'There are not accepted samples in img with id {}'.format(img_idx)
                tries += 1

            N = np.log(len(log_ratio_p_q))
            ll_i = self.get_loglikelihood(log_ratio_p_q[accepted_samples], N)
            # print('IDX {} ll {} sigma {}'.format(img_idx, ll_i, sigma))
            self.writer.add_histogram('log(weights)/{}'.format(img_idx), log_ratio_p_q, j)
            self.writer.add_scalar('loglikelihood/{}'.format(img_idx), ll_i, j)
            self.writer.add_scalar('N_i/{}'.format(img_idx), N_i, j)
            if accepted_samples_count < 0.95 * N_i or self.T == 40:
                j = j + 1
            else:
                self.writer.add_scalar('loglikelihood/{}'.format(img_idx), ll_i, j + 1)
                self.writer.add_scalar('loglikelihood/{}'.format(img_idx), ll_i, j + 2)
                self.writer.add_scalar('N_i/{}'.format(img_idx), N_i, j + 1)
                self.writer.add_scalar('N_i/{}'.format(img_idx), N_i, j + 2)
                j = j + 3

            if accepted_samples_count <= N_i / 10:
                if N_i == self.N_max:
                    break
                n_iter = math.ceil(N_i * self.k / self.bs)
                N_i = min(n_iter * self.bs, self.N_max)
                print(N_i)

        return ll_i / np.log(10)

    def get_psnr_ratio(self, z_c, x_tmp, N_i, sigma, z_dim):
        n_iter = math.ceil(N_i / self.bs)
        psnr_tmp = np.zeros(N_i)
        log_ratio_p_q = np.zeros(N_i)
        log_ratio_1_q = np.zeros(N_i)
        for n in range(n_iter):
            z_c_tile = np.tile(z_c, [self.bs, 1])
            z_tmp = z_c_tile + np.random.normal(0, sigma, [self.bs, z_dim])
            x_gener = self.model.sample3(z_tmp)
            x_gener = self.process_output(x_gener)
            log_ratio_p_q[n * self.bs:(n + 1) * self.bs] = self.log_ratio_p_q(z_tmp, z_c_tile, sigma)
            log_ratio_1_q[n * self.bs:(n + 1) * self.bs] = self.log_ratio_1_q(z_tmp, z_c_tile, sigma)
            psnr_tmp[n * self.bs:(n + 1) * self.bs] = loss.PSNR(x_gener, x_tmp, axis=(1, 2, 3))

        return psnr_tmp, log_ratio_p_q, log_ratio_1_q

    def log_ratio_p_q(self, z, z_c, sigma):
        z_dim = z.shape[-1]
        return 1 / 2 * (np.sum((z - z_c) ** 2, axis=-1) / (sigma ** 2) - np.sum(z ** 2, axis=-1)) + z_dim * np.log(
            sigma)

    def log_ratio_1_q(self, z, z_c, sigma):
        z_dim = z.shape[-1]
        a = 1 / 2 * (np.sum((z - z_c) ** 2, axis=-1) / (sigma ** 2))
        a2 = z_dim * np.log(sigma)
        a3 = z_dim * np.log(2 * np.pi) / 2
        return a + a2 + a3

    def get_loglikelihood(self, log_ratio_p_q_accepted, N):
        """
        Compute the log likelihood using the log_sum_trick
        :param log_ratio_p_q_accepted:
        :param N:
        :return:
        """
        return log_sum_exp_trick(log_ratio_p_q_accepted) - N
