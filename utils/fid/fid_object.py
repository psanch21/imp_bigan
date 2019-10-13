import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm

from utils.fid.inception import InceptionV3


class FID:
    def __init__(self, dims=2048):
        '''
        64: first max pooling features
        192: second max pooling features
        768: pre-aux classifier features
        2048: final average pooling features (this is the default)

        '''
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([block_idx])
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.model.cuda()

        self.model.eval()
        self.bs = 128
        self.m1 = None
        self.s1 = None

        self.dims = dims

    def get_activations(self, images):
        N = images.shape[0]
        n_batches = N // self.bs
        n_used_imgs = n_batches * self.bs

        pred_arr = np.empty((n_used_imgs, self.dims))

        for i in tqdm(range(n_batches)):
            start = i * self.bs
            end = start + self.bs

            batch = torch.from_numpy(images[start:end]).type(torch.FloatTensor)
            if self.cuda:
                batch = batch.cuda()

            pred = self.model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.shape[2] != 1 or pred.shape[3] != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred_arr[start:end] = pred.cpu().data.numpy().reshape(self.bs, -1)

        return pred_arr

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.

        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)

    def calculate_activation_statistics(self, images):
        """Calculation of the statistics used by the FID.
        Params:
        -- files       : List of image files paths
        -- model       : Instance of inception model
        -- batch_size  : The images numpy array is split into batches with
                         batch size batch_size. A reasonable batch size
                         depends on the hardware.
        -- dims        : Dimensionality of features returned by Inception
        -- cuda        : If set to True, use GPU
        -- verbose     : If set to True and parameter out_step is given, the
                         number of calculated batches is reported.
        Returns:
        -- mu    : The mean over samples of the activations of the pool_3 layer of
                   the inception model.
        -- sigma : The covariance matrix of the activations of the pool_3 layer of
                   the inception model.
        """
        act = self.get_activations(images)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    def preprocess(self, x):

        return x.transpose((0, 3, 1, 2)) / 255

    def save_stats_real(self, x_real):
        if x_real.shape[-1] == 1:
            print('Not possible to compute FID with image shape: {}'.format(x_real.shape))
            return
        x_real = self.preprocess(x_real)
        self.m1, self.s1 = self.calculate_activation_statistics(x_real)

    def compute2(self, x_gener):
        if type(self.m1) == type(None):
            return -1
        x_gener = self.preprocess(x_gener)
        m2, s2 = self.calculate_activation_statistics(x_gener)
        fid_value = self.calculate_frechet_distance(self.m1, self.s1, m2, s2)
        print('FID: {}'.format(fid_value))
        return fid_value

    def compute(self, x_real, x_gener_list):
        # Images should be int with shape HMC
        x_real = self.preprocess(x_real)
        m1, s1 = self.calculate_activation_statistics(x_real)

        fids = list()
        for x_gener in x_gener_list:
            x_gener = self.preprocess(x_gener)
            m2, s2 = self.calculate_activation_statistics(x_gener)
            fid_value = self.calculate_frechet_distance(m1, s1, m2, s2)
            print('FID: {}'.format(fid_value))
            fids.append(fid_value)
        return fids

    def compute3(self, x_real_list, x_gener_list):
        # Images should be int with shape HMC
        m1_list = list()
        s1_list = list()
        fid_dict = dict()
        for i, x_real in enumerate(x_real_list):
            x_real = self.preprocess(x_real)
            m1, s1 = self.calculate_activation_statistics(x_real)
            m1_list.append(m1)
            s1_list.append(s1)
            fid_dict[i] = list()

        for j, x_gener in enumerate(x_gener_list):
            x_gener = self.preprocess(x_gener)
            m2, s2 = self.calculate_activation_statistics(x_gener)
            for i in range(len(x_real_list)):
                print('len m1: {} len m2: {}'.format(len(m1_list[i]), len(m2)))
                fid_value = self.calculate_frechet_distance(m1_list[i], s1_list[i], m2, s2)
                print(' real({}) gener({}) || FID: {}'.format(i, j, fid_value))
                fid_dict[i].append(fid_value)
        return fid_dict
