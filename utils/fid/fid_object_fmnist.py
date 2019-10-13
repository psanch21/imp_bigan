import numpy as np
import torch
from tqdm import tqdm

import utils.fid.fmnist_classifier as fm_clf
from utils.fid.fid_object import FID


class FID_FM(FID):
    def __init__(self):
        '''
        64: first max pooling features
        192: second max pooling features
        768: pre-aux classifier features
        2048: final average pooling features (this is the default)

        '''
        trainer = fm_clf.TrainClf(gpu=0)
        self.model = trainer.train(lr=0.005, n_epochs=30, batch_size=128)
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.model.cuda()

        self.model.eval()
        self.bs = 128
        self.m1 = None
        self.s1 = None

        self.dims = self.model.get_feature_dim()

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

            pred = self.model.extract_features(batch)

            pred_arr[start:end] = pred.cpu().data.numpy().reshape(self.bs, -1)

        return pred_arr

    def save_stats_real(self, x_real):
        x_real = self.preprocess(x_real)
        self.m1, self.s1 = self.calculate_activation_statistics(x_real)
