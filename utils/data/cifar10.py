import os

import torchvision

from torchvision import transforms
import utils.data.functions.aux as d_fn


def load_cifar10(data_path):
    print('Loading CIFAR10')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    totalset = torchvision.datasets.CIFAR10(root=os.path.join(data_path, 'CIFAR10'), train=True,
                                            download=True, transform=transform)

    train_size = int(0.9 * len(totalset))
    valid_size = len(totalset) - train_size
    trainset, validset = d_fn.determ_split(totalset, [train_size, valid_size])

    testset = torchvision.datasets.CIFAR10(root=os.path.join(data_path, 'CIFAR10'), train=False,
                                           download=True, transform=transform)

    return (trainset, validset, testset)
