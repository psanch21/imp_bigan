import os

import torchvision
from torchvision import transforms


def load_stl10(data_path):
    print('Loading STL10')

    trainset = load_stl10_set(root=os.path.join(data_path, 'STL10'), split='unlabeled')
    validset = load_stl10_set(root=os.path.join(data_path, 'STL10'), split='train')
    testset = load_stl10_set(root=os.path.join(data_path, 'STL10'), split='test')

    return (trainset, validset, testset)


def load_stl10_set(root='../Data/STL10', split='test'):
    stl10 = torchvision.datasets.STL10(root=root, split=split, download=True,
                                       transform=transforms.Compose([
                                           transforms.CenterCrop(96),
                                           transforms.Resize(64),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       ]))

    return stl10
