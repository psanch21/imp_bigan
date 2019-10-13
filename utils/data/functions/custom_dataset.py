import torch
import numpy as np


# %% PyTorch Dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, probs=None, perc=0):
        self.dataset = dataset
        self.n_imgs = dataset.__len__()
        self.probs = probs
        self.perc = perc  # Percentage of samples drawn non-uniformly
        if probs is None:
            assert self.perc == 0
        else:
            assert self.n_imgs == len(self.probs)

        if perc is None:
            ten_percent = int(self.n_imgs * 0.1)
            self.idx_large_probs = np.argsort(self.probs)[::-1][:ten_percent]  # From larger to smaller

    def __len__(self):
        'Denotes the total number of samples'
        return self.dataset.__len__()

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        if self.perc is None:
            sequence = self.__get_lowlikelkihood_item()
        elif self.perc > 0:
            sequence = self.__get_nonuni_item(index)
        else:
            sequence, _ = self.dataset.__getitem__(index)
        return sequence

    def __get_nonuni_item(self, index):
        u = np.random.uniform()
        if (u < self.perc):
            index = np.random.choice(self.n_imgs, 1, p=self.probs)[0]
        sequence, _ = self.dataset.__getitem__(index)
        return sequence

    def __get_lowlikelkihood_item(self):
        index = np.random.choice(len(self.idx_large_probs), 1)[0]

        sequence, _ = self.dataset.__getitem__(self.idx_large_probs[index])
        return sequence
