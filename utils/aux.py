import os

import torch

import utils.constants as cte


def cuda(xs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    if not isinstance(xs, (list, tuple)):
        return xs.to(torch.device(device))
    else:
        return [x.to(torch.device(device)) for x in xs]


def create_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    return folder


def load_model(ckpt_file, gpu, device):
    print('\nCheckpoint file: {}'.format(ckpt_file))
    assert os.path.isfile(ckpt_file) == True, 'Checkpoint file not found: {}'.format(ckpt_file)

    print('Restore from: {}'.format(ckpt_file))
    if int(gpu) == -1:
        train_dict = torch.load(ckpt_file, map_location=device)
    else:
        train_dict = torch.load(ckpt_file)

    model_dict = train_dict['model']
    dataset = train_dict['dataset']
    model_params_dict = model_dict['params']
    print(model_params_dict.keys())
    print('Getting GAN model')
    gan = get_model2(model_params_dict['name'], dataset)

    gan.load_checkpoint(model_dict)

    print('GAN Epochs trained: {}'.format(train_dict['epoch']))

    gan.eval()
    if gpu != -1:
        gan.cuda()

    return gan


def get_model(args):
    if args.dataset in [cte.CELEBA, cte.CELEBA_S, cte.STL10, cte.LSUN]:
        c, h, w = [3, 64, 64]
    if args.dataset == cte.CIFAR10:
        c, h, w = [3, 32, 32]
    if args.dataset in [cte.MNIST, cte.MNIST1, cte.FMNIST]:
        c, h, w = [1, 28, 28]

    deep = True if args.dataset in [cte.STL10] else False
    model = None
    if args.gan_type in cte.BiGAN_LIST:
        from model.bigan_model import BiGAN
        model = BiGAN(z_dim=args.z_dim, out_size=h, image_channels=c, loss_type=args.loss_type,
                      bigan_type=args.gan_type, deep=deep)

    assert model is not None
    model.init()
    return model


def get_model2(gan_type, dataset):
    model = None
    deep = True if dataset in [cte.STL10] else False
    if gan_type in cte.BiGAN_LIST:
        from model.bigan_model import BiGAN
        model = BiGAN(deep=deep)

    assert model is not None
    return model


def get_trainer(args, args_model, scaler, model, probs):
    trainer = None
    if args.gan_type in cte.BiGAN_LIST:
        from model.bigan_trainer import SNBiGANTrainer
        trainer = SNBiGANTrainer(model, dataset=args_model.dataset, scaler=scaler, args=args_model,
                                 batch_size=args.batch_size, clip=args_model.clip, save_every=args_model.save_every,
                                 ckpt_folder=args_model.ckpt_dir, l_rate=args_model.l_rate,
                                 n_critic=args_model.n_critic,
                                 probs=probs, l_recons=args_model.lr, l_norm=args_model.ln, l_perc=args_model.lp,
                                 l_dis=args_model.ld, beta1=args_model.beta1, beta2=args_model.beta2)

    assert trainer is not None
    return trainer


def get_model_trainer(args, scaler, probs):
    model = get_model(args)
    trainer = get_trainer(args, scaler, model, probs)
    return model, trainer
