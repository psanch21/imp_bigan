import os

import torch

import utils.constants as cte
import utils.data.functions.aux as data_fn
import utils.log_likelihood as ll
from utils.aux import get_model, get_trainer
from utils.data.functions.scalers import MinMaxScaler
from utils.fid.fid_object import FID
from utils.fid.fid_object_fmnist import FID_FM
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
model.eval()

# %% Plotting results

# pva.activate_latex_format()

x_train_norm = data_fn.get_numpy_from_dataset(train_set, n_samples=None, shuffle=False)
x_train = data_fn.denorm(x_train_norm, scaler)

x_test_norm = data_fn.get_numpy_from_dataset(test_set, n_samples=None, shuffle=False)
x_test = data_fn.denorm(x_test_norm, scaler)

# %% Generation and reconstruction
model.plot_generation(base_dir=trainer.save_dir, scaler=scaler)

model.plot_reconstruction(base_dir=os.path.join(trainer.save_dir, 'test'), scaler=scaler, x_norm=x_test_norm)

# %% Marginal log-likelihood

z_list, x_recons_test_norm = model.recons(x_test_norm)
x_recons_test = data_fn.denorm(x_recons_test_norm, scaler)

z_list, x_recons_tr_norm = model.recons(x_train_norm)
x_recons_tr = data_fn.denorm(x_recons_tr_norm, scaler)

if args.gan_type in [cte.EPMDGAN, cte.PMDGAN]:
    print('Computing log-likelihood')
    import utils.log_likelihood as ll

    ll_object = ll.ApproxLL(model, scaler, folder=os.path.join(trainer.save_dir, 'test'), N=128, N_max=256, T=40)
    ll_object.compute_ll(x_recons_test, z_list)
    model.plot_loglikelihod(base_dir=os.path.join(trainer.save_dir, 'test'), scaler=scaler, x_norm=x_test_norm)

    ll_object = ll.ApproxLL(model, scaler, folder=os.path.join(trainer.save_dir, 'train'), N=128, N_max=256, T=40)
    ll_object.compute_ll(x_recons_tr, z_list)
    model.plot_loglikelihod(base_dir=os.path.join(trainer.save_dir, 'train'), scaler=scaler, x_norm=x_train_norm)

x_tr = x_train[:1000]
x_tst = x_test[:1000]

n_samples = min(len(x_tr), len(x_tst))
x_gener_norm, z = model.sample(n_samples)
x_gener = data_fn.denorm(x_gener_norm, scaler)

# %% FID score
csv_file = os.path.join('experiments', 'fid_scores.csv')
print('Computing FID: {}'.format(csv_file))

if args.dataset != cte.FMNIST:
    fid_object = FID(dims=2048)
else:
    fid_object = FID_FM()

fid_score_dict = fid_object.compute3([x_tr, x_tst], [x_gener])
fid_score_tr = fid_score_dict[0][0]
fid_score_tst = fid_score_dict[1][0]
row_list = [args.dataset, args.gan_type, args.lr, args.ld, args.lp, fid_score_tst, fid_score_tr, trainer.save_dir]

df = data_fn.get_csv(csv_file,
                     ['dataset', 'model_type', 'lambda_r', 'lambda_d', 'lambda_p', 'test', 'train', 'model_name'])
data_fn.save_row_csv(df, csv_file, row_list, drop_duplicates=True, subset='model_name')

# %% Precision-Recall score
from utils.precision_recall.pr_object import ImprovedPR
from utils.precision_recall.pr_object_fmnist import ImprovedPR_FM

csv_file = os.path.join('experiments', 'pr_scores.csv')
print('Computing PR: {}'.format(csv_file))

if args.dataset != cte.FMNIST:
    pr_object = ImprovedPR(dims=2048)
else:
    pr_object = ImprovedPR_FM()

x_tr = x_train[:1024]
x_tst = x_test[:1024]
x_gen = x_gener[:1024]
pr_dict = pr_object.compute3([x_tr, x_tst], [x_gen])
p_tr, r_tr = pr_dict[0][0]['precision'][0], pr_dict[0][0]['recall'][0]
p_tst, r_tst = pr_dict[1][0]['precision'][0], pr_dict[1][0]['recall'][0]
row_list = [args.dataset, args.gan_type, args.lr, args.ld, args.lp, p_tst, p_tr, r_tst, r_tr, trainer.save_dir]
df = data_fn.get_csv(csv_file,
                     ['dataset', 'model_type', 'lambda_r', 'lambda_d', 'lambda_p', 'test_p', 'train_p', 'test_r',
                      'train_r', 'model_name'])
data_fn.save_row_csv(df, csv_file, row_list, drop_duplicates=True, subset='model_name')
