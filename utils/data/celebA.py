import os

from torchvision import transforms
from torchvision.datasets import ImageFolder

import utils.data.functions.aux as d_fn


def load_celebA(data_path):
    print('Loading CelebA')
    transform = transforms.Compose([
        transforms.CenterCrop(160),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    totalset = ImageFolder(os.path.join(data_path, 'celebA', 'train'), transform)

    train_size = int(0.9 * len(totalset))
    valid_size = len(totalset) - train_size
    trainset, validset = d_fn.determ_split(totalset, [train_size, valid_size])

    testset = ImageFolder(os.path.join(data_path, 'celebA', 'test'), transform)
    return (trainset, validset, testset)


def load_celebA_small(data_path):
    print('Loading CelebA Small')
    transform = transforms.Compose([
        transforms.CenterCrop(160),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    totalset = ImageFolder(os.path.join(data_path, 'celebA_40k', 'train'), transform)

    train_size = int(0.9 * len(totalset))
    valid_size = len(totalset) - train_size
    trainset, validset = d_fn.determ_split(totalset, [train_size, valid_size])

    testset = ImageFolder(os.path.join(data_path, 'celebA_40k', 'test'), transform)
    return (trainset, validset, testset)
