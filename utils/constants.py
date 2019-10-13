import numpy as np

COLORS = ['darkblue', 'orange', 'darkgreen', 'darkred', 'lightgreen', 'lightred']
MARKERS = ['o', 'v', 's', 'd', 'x', '+']

# X reconstruction error
L1 = 1
L2 = 2
HINGE = 'hin'
BCE = 'bce'
WASSERSTEIN = 'was'

GAN_LOSS_LIST = [HINGE, BCE, WASSERSTEIN]

# %% Datasets
CIFAR10 = 'c10'

MNIST = 'mn'
MNIST1 = 'mn1'
MNIST2 = 'mn2'

FMNIST = 'fm'

CELEBA_S = 'cAs'
CELEBA = 'cA'

STL10 = 'stl'

LSUN = 'ls'

DATASET_LIST = [CIFAR10, CELEBA_S, CELEBA, MNIST, MNIST1, MNIST2, FMNIST, STL10, LSUN]

TEST_STR = 'tst'
TRAIN_STR = 'tr'

FM_IM_TST = [82, 51, 7, 128, 194]
FM_IM_TR = [82, 30, 56, 128, 194]

C10_IM_TST = [1, 9, 37, 82]
C10_IM_TR = [134, 136, 138, 186]

CA_IM_TST = [1, 9, 37, 82]
CA_IM_TR = [134, 136, 138, 186]

LS_IM_TST = [1, 9, 37, 82]
LS_IM_TR = [134, 136, 138, 186]

IM = {FMNIST: [FM_IM_TST, FM_IM_TR], CIFAR10: [C10_IM_TST, C10_IM_TR], CELEBA: [CA_IM_TST, CA_IM_TR],
      LSUN: [LS_IM_TST, LS_IM_TR]}

# %% Evaluation
FID = 'fid'
PR = 'pr'
CLASSES_PROP = 'classes'
RECONS = 'recons'
GENER = 'gener'
TRAINING = 'training'

COMPUTE_LL = 'compute-ll'
COMPUTE_LL_TST = 'compute-ll-tst'
LOGLIKELIHOOD = 'loglikelihood'

EVALUATION_LIST = [FID, ]

# %% GAN Type
BiGAN = 'bigan'  # Standard BiGAN
MDGAN = 'mdgan'  # Regularized BiGAN ( X recons regularization)
PMDGAN = 'pmdgan'  # Typically Regularized BiGAN ( X recons + E(z) regularization)
MLPMDGAN = 'mlpmdgan'  # Non-uniform Typically Regularized BiGAN
EPMDGAN = 'epmdgan'  # RBiGAN with non-uni sampling based on PSNRB

TRBiGAN_LIST = [PMDGAN, MLPMDGAN, EPMDGAN]
BiGAN_LIST = [BiGAN, MDGAN, PMDGAN, MLPMDGAN, EPMDGAN]


# %% Methods about constants

def map_name_to_id(set_names):
    out = list()
    for n in set_names:
        if n == MDGAN:
            out.append(0)
        elif n == PMDGAN:
            out.append(3)
        elif n == MLPMDGAN:
            out.append(4)
        elif n == EPMDGAN:
            out.append(5)

    names_unique = [set_names[i] for i in np.argsort(out)]
    return names_unique


def map_name_to_id2(set_names):
    out = list()
    for n in set_names:
        if n == MDGAN:
            out.append(0)
        elif n == PMDGAN:
            out.append(1)
        elif n == MLPMDGAN:
            out.append(2)
        elif n == EPMDGAN:
            out.append(3)

    return out


def map_name_to_id3(set_names):
    out = list()
    for n in set_names:
        if n == MDGAN:
            out.append(0)
        elif n == PMDGAN:
            out.append(1)
        elif n == MLPMDGAN:
            out.append(2)
        elif n == EPMDGAN:
            out.append(3)

    return [i for i in np.sort(out)]


def get_name(n):
    if n == MDGAN:
        return 'MDGAN'
    elif n == PMDGAN:
        return 'P-MDGAN'
    elif n == MLPMDGAN:
        return 'P-MDGAN\nwith MLeq'
    elif n == EPMDGAN:
        return 'EP-MDGAN'
    else:
        print('Error in get_name')


def get_name2(n):
    if n == MDGAN:
        return 'MDGAN'
    elif n == PMDGAN:
        return 'P-MDGAN'
    elif n == MLPMDGAN:
        return 'P-MDGAN with MLeq'
    elif n == EPMDGAN:
        return 'EP-MDGAN'
    else:
        print('Error in get_name')
