import os

import torchvision

from torchvision import transforms
import utils.data.functions.aux as d_fn

def load_lsun(data_path):
    print('Loading LSUN')
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    totalset = torchvision.datasets.LSUN(root=os.path.join(data_path, 'LSUN'), classes=['bedroom_train'],
                                         transform=transform)

    train_size = int(0.03 * len(totalset))
    valid_size = len(totalset) - train_size
    totalset, _ = d_fn.determ_split(totalset, [train_size, valid_size])

    train_size = int(0.9 * len(totalset))
    valid_size = len(totalset) - train_size
    trainset, validset = d_fn.determ_split(totalset, [train_size, valid_size])

    train_size = int(0.5 * len(validset))
    valid_size = len(validset) - train_size
    validset, testset = d_fn.determ_split(validset, [train_size, valid_size])

    return (trainset, validset, testset)