import os

import numpy as np
import pandas as pd
import torch
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset

import utils.constants as cte
import utils.data.functions.scalers as sca


def determ_split(dataset, lengths):
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    return [Subset(dataset, np.arange(offset - length, offset)) for offset, length in
            zip(_accumulate(lengths), lengths)]


def load(dataset, data_path):
    """
    Load data with the format [-1, C, H, W] normalize between [-1,1]
    :param dataset:
    :param data_path:
    :return:
    """

    assert dataset in cte.DATASET_LIST
    print('Loading {} data from {}'.format(dataset, data_path))
    trainset, validset, testset = None, None, None

    if dataset == cte.CIFAR10:
        from utils.data.cifar10 import load_cifar10
        trainset, validset, testset = load_cifar10(data_path)
    if dataset == cte.STL10:
        from utils.data.stl10 import load_stl10
        trainset, validset, testset = load_stl10(data_path)


    elif dataset == cte.CELEBA_S:
        from utils.data.celebA import load_celebA_small
        trainset, validset, testset = load_celebA_small(data_path)

    elif dataset == cte.CELEBA:
        from utils.data.celebA import load_celebA
        trainset, validset, testset = load_celebA(data_path)

    elif dataset == cte.MNIST:
        from utils.data.mnist import load_mnist
        trainset, validset, testset = load_mnist(data_path)

    elif dataset == cte.FMNIST:
        from utils.data.fminist import load_fmnist
        trainset, validset, testset = load_fmnist(data_path)

    elif dataset == cte.LSUN:
        from utils.data.lsun.lsun import load_lsun
        trainset, validset, testset = load_lsun(data_path)


    elif dataset == cte.MNIST1:
        from utils.data.mnist import load_mnist1
        trainset, validset, testset = load_mnist1(data_path)

    assert trainset is not None, 'Wrong dataset or data_path: {}, {}'.format(dataset, data_path)
    print('train: {}'.format(len(trainset)))
    print('valid: {}'.format(len(validset)))
    print('test: {}'.format(len(testset)))

    return (trainset, validset, testset)


def denorm(x, scaler):
    if (x.shape[1] > 3):
        print('Error: Data should be in CHW format')
    dims = x.shape
    x = scaler.inverse_transform(x.reshape(dims[0], -1)).reshape(dims).astype(int)
    x = np.transpose(x, [0, 2, 3, 1])  # CHW --> HWC
    return x


def get_scaler():
    scaler = sca.MinMaxScaler(feature_range=(-1, 1))
    return scaler


def get_numpy_from_dataset(dataset, n_samples=None, shuffle=False):
    if (n_samples == None):
        n_samples = len(dataset)
    elif n_samples < 0:
        n_samples = min(50000, len(dataset))
    else:
        n_samples = min(n_samples, len(dataset))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=n_samples, shuffle=shuffle, num_workers=0)
    return next(iter(data_loader))[0].data.numpy()


def get_csv(csv_name, cols):
    if (not os.path.exists(csv_name)):
        train_df = pd.DataFrame(columns=cols)
        train_df.to_csv(csv_name, sep=';', index=False)
    else:
        train_df = pd.read_csv(csv_name, sep=';')

    return train_df


def save_row_csv(df, csv_file, row_list, drop_duplicates=False, subset='model_name'):
    df.loc[len(df)] = row_list
    if drop_duplicates:
        df = df.drop_duplicates(subset=subset, keep='last')
        df.to_csv(csv_file, sep=';', index=False)
