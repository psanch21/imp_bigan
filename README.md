# Improving BiGAN training with marginal likelihood equalization - Official PyTorch implementation

## Getting started

Clone the package [probvis](https://github.com/psanch21/prob-visualize), which is needed to visualize the results. Then create a conda environment as follows:
```
conda env create -f ./environments/env_macos.yml
```

and you should be ready to to go.

## Usage
### Parameters
- ````ld````: Parameter that controls the shape of the non-uniform distribution, namely $\lambda_{dist}$
- ````lp````: Parameter that controls the percentage of the non-uniform samples per mini-bartch, namely $\lambda_{perc}$
- ````lr````: Parameter that controls reconstruction regularization, namely $\lambda_{cyc}$

### Examples
```
python3 main.py --n_epochs 800 --z_dim 256 --dataset c10 --data_path ../Data --gan_type epmdgan --gpu 0 --ckpt_dir experiments/c10/3  --lr 3 --lp 0.8 --ld 4
```

## BibTex citation

## Acknowledgements

This code uses the folowing repositories:

- Downloading/Loading LSUN data set: [code](https://github.com/fyu/lsun_toolkit/) [paper](https://arxiv.org/pdf/1506.03365.pdf)

- Computing the FID score: [code](https://github.com/mseitzer/pytorch-fid) 

- Regularize GAN with Spectral Normalization: [code](https://github.com/christiancosgrove/pytorch-spectral-normalization-gan) [paper](https://openreview.net/forum?id=B1QRgziT-)

- Computing Precision & Recall: [code](https://github.com/kynkaat/improved-precision-and-recall-metric) [paper](https://arxiv.org/abs/1904.06991)
## Contact

**Pablo Sanchez** - For any questions, comments or help to get it to run, please don't hesitate to mail me: <pablo.sanchez-martin@tuebingen.mpg.de>

