import argparse
import os

import utils.constants as cte


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--l_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Learning rate')
    parser.add_argument('--beta2', type=float, default=0.999, help='Learning rate')
    parser.add_argument('--n_critic', type=int, default=5,
                        help='Ratio between  # updates distriminator and # update generator')

    parser.add_argument('--save_every', type=int, default=10, help='Save model every n epochs')
    parser.add_argument('--print_every', type=int, default=10, help='Print result every n epochs')

    parser.add_argument('--z_dim', type=int, default=256, help='Dimension of z')
    parser.add_argument('--batch_size', type=int, default=128, help='Constraints to z')
    parser.add_argument('--clip', type=int, default=5, help='Set to 1 to restore model')

    parser.add_argument('--dataset', type=str, default='c10', help='Dataset to train the model')
    parser.add_argument('--data_path', type=str, default='../Data', help='Root directory for data')
    parser.add_argument('--ckpt_dir', type=str, default='saves', help='Parent directory to save models')
    parser.add_argument('--ckpt_file', type=str, default=None, help='Checkpoint file to restore model')
    parser.add_argument('--probs_file', type=str, default=None, help='Checkpoint file to restore model')

    parser.add_argument('--loss_type', type=str, default=cte.BCE, help='Root directory for data')

    parser.add_argument('--gpu', type=str, default='-1', help='Select gpu')

    parser.add_argument('--gan_type', type=str, default=cte.BiGAN, help='Select GAN arquitecture')

    # Hyperparameter for X reconstruction regularization
    parser.add_argument('--lr', type=float, default=1, help='Regularization parameter for reconstruction loss')
    # Hyperparameter for E(x) regularization
    parser.add_argument('--ln', type=float, default=0.0,
                        help='Initial Regularization parameter for E(X) regularization ')
    # Hyperparameters for non-uniform sampling
    parser.add_argument('--lp', type=float, default=0.0, help='Percentage of non-uniform sampling')
    parser.add_argument('--ld', type=int, default=0, help='Exponent to compute probabilities for non-uniform sampling')

    args = parser.parse_args()

    return validate_args(args)


def validate_args(args):
    assert args.n_epochs > 0, "Number of epochs in non positive"
    assert args.l_rate > 0
    assert args.beta1 >= 0
    assert args.beta2 >= 0.9

    assert args.n_critic > 0
    assert args.z_dim > 0
    assert args.batch_size > 0
    assert args.clip > 0

    assert args.loss_type in cte.GAN_LOSS_LIST
    assert args.dataset in cte.DATASET_LIST
    assert args.gan_type in cte.BiGAN_LIST

    assert args.lr >= 0
    assert args.ln >= 0
    assert args.lp >= 0
    assert args.ld >= 0
    if args.gan_type in [cte.MLPMDGAN]:
        assert args.ld >= 0
        assert args.probs_file is not None
        assert str(args.ld) in args.probs_file.split('/')[-1]
    assert isinstance(args.ld, int)

    return args


def get_args_evaluate():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--data_path', type=str, default='../Data', help='Parent directory to save models')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset to train the model')
    parser.add_argument('--output_dir', type=str, default=None, help='Parent directory to save models')
    parser.add_argument('--ckpt_name', type=str, default=None, help='Parent directory to save models')
    parser.add_argument('--folder', type=str, default=None, help='Parent directory to save models')

    parser.add_argument('--operation', type=str, default=None, help='Parent directory to save models')

    parser.add_argument('--l_dist', type=int, default=-1, help='Parent directory to save models')
    parser.add_argument('--T', type=int, default=-40, help='PSNR Threshold')
    parser.add_argument('--gpu', type=str, default='-1', help='Select gpu')

    args = parser.parse_args()

    return validate_args_evaluate(args)


def validate_args_evaluate(args):
    assert args.dataset is not None, "Select a datset"
    assert args.folder is not None
    assert args.operation is not None
    assert args.T > 0
    args.output_dir = args.folder
    if cte.NON_UNI_PROBS in args.operation:
        assert args.l_dist > 0

    return args


def get_device(gpu):
    if (int(gpu) == -1):
        device = 'cpu'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        device = 'cuda'
        print('Using GPU: {}'.format(gpu))
    return device
