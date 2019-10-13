import os

import torch

import utils.constants as cte
import utils.data.functions.aux as data_fn
import utils.log_likelihood as ll
from utils.aux import get_model, get_trainer
from utils.data.functions.scalers import MinMaxScaler
from utils.process_args import get_device, get_args

# %% Hyperparameters
args = get_args()

device = get_device(args.gpu)

data = data_fn.load(args.dataset, args.data_path)
train_set, valid_set, test_set = data

scaler = MinMaxScaler(feature_range=(-1, 1))

# %% Get the BiGAN model and trainer
model = get_model(args)

if args.gan_type == cte.MLPMDGAN:
    probs = ll.get_probs_from_ll(base_dir=args.probs_file, T=40, l_dis=args.ld)
else:
    probs = None
trainer = get_trainer(args, args, scaler, model, probs)

# %% Restore model
if os.path.isfile(trainer.ckpt_file):
    print('\nRestoring model: {}'.format(trainer.ckpt_file))
    model_dict = torch.load(trainer.ckpt_file, map_location=device) if int(args.gpu) < 0 else torch.load(
        trainer.ckpt_file)
    trainer.load_checkpoint(model_dict)
    print('Epochs trained: {}\n'.format(model_dict['epoch']))

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Trainable params: {}'.format(total_params))

print('Checkpoint file: {}'.format(trainer.get_complete_name()))

# %% Train the model
print('Training points: {}'.format(len(train_set)))
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
trainer.train(train_set, valid_set, n_epochs=args.n_epochs)
