import os

import torchvision

from torchvision import transforms
import utils.data.functions.aux as d_fn


def load_mnist(data_path):
    print('Loading MNIST')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    totalset = torchvision.datasets.MNIST(root=os.path.join(data_path, 'MNIST'), train=True,
                                          download=True, transform=transform)

    train_size = int(0.9 * len(totalset))
    valid_size = len(totalset) - train_size
    trainset, validset = d_fn.determ_split(totalset, [train_size, valid_size])

    testset = torchvision.datasets.MNIST(root=os.path.join(data_path, 'MNIST'), train=False,
                                         download=True, transform=transform)

    return (trainset, validset, testset)


def load_mnist1(data_path):
    print('Loading MNIST1')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    totalset = torchvision.datasets.MNIST(root=os.path.join(data_path, 'MNIST'), train=True,
                                          download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root=os.path.join(data_path, 'MNIST'), train=False,
                                         download=True, transform=transform)

    if torchvision.__version__ == '0.3.0':
        idx = totalset.targets == 1
        totalset.targets = totalset.targets[idx]
        totalset.data = totalset.data[idx]

        idx = testset.targets == 1
        testset.targets = testset.targets[idx]
        testset.data = testset.data[idx]
    else:
        idx = totalset.train_labels == 1
        totalset.train_labels = totalset.train_labels[idx]
        totalset.train_data = totalset.train_data[idx]

        idx = testset.test_labels == 1
        testset.test_labels = testset.test_labels[idx]
        testset.test_data = testset.test_data[idx]

    train_size = int(0.9 * len(totalset))
    valid_size = len(totalset) - train_size
    trainset, validset = d_fn.determ_split(totalset, [train_size, valid_size])

    return (trainset, validset, testset)
