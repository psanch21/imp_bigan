import numpy as np
import torch
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm

import utils.precision_recall.precision_recall as pr_fn
from utils.fid.inception import InceptionV3


class ImprovedPR:
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
        self.f1 = None

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

    def knn_precision_recall_features(self, ref_features, eval_features):
        state = pr_fn.knn_precision_recall_features(ref_features, eval_features, nhood_sizes=[3],
                                                    row_batch_size=25000, col_batch_size=50000, num_gpus=1)
        print(state)
        return state

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
        -- features    : The mean over samples of the activations of the pool_3 layer of
                   the inception model.
        """
        features = self.get_activations(images)

        return features

    def preprocess(self, x):

        return x.transpose((0, 3, 1, 2)) / 255

    def save_stats_real(self, x_real):
        if x_real.shape[-1] == 1:
            print('Not possible to compute Precision-recall with image shape: {}'.format(x_real.shape))
            return
        x_real = self.preprocess(x_real)
        self.f1 = self.calculate_activation_statistics(x_real)

    def compute2(self, x_gener):
        x_gener = self.preprocess(x_gener)
        f2 = self.calculate_activation_statistics(x_gener)
        state = self.knn_precision_recall_features(self.f1, f2)
        return state

    def compute(self, x_real, x_gener_list):
        # Images should be int with shape HMC
        x_real = self.preprocess(x_real)
        f1 = self.calculate_activation_statistics(x_real)

        pr_list = list()
        for x_gener in x_gener_list:
            x_gener = self.preprocess(x_gener)
            f2 = self.calculate_activation_statistics(x_gener)
            state = self.knn_precision_recall_features(f1, f2)
            pr_list.append(state)
        return pr_list

    def compute3(self, x_real_list, x_gener_list):
        # Images should be int with shape HMC
        f1_list = list()
        state_dict = dict()
        for i, x_real in enumerate(x_real_list):
            x_real = self.preprocess(x_real)
            f1 = self.calculate_activation_statistics(x_real)
            f1_list.append(f1)
            state_dict[i] = list()

        for j, x_gener in enumerate(x_gener_list):
            x_gener = self.preprocess(x_gener)
            f2 = self.calculate_activation_statistics(x_gener)
            for i in range(len(x_real_list)):
                state = self.knn_precision_recall_features(f1_list[i], f2)

                state_dict[i].append(state)
        return state_dict
