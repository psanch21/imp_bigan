import numpy as np
import torch
import torchvision
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_event_accumulator(folder):
    event_acc = EventAccumulator(folder)
    event_acc.Reload()

    print('Tags in TensorBoard data located in: {}'.format(folder))
    return event_acc


def images_to_tensor_grid(x):
    if np.max(x) > 1: x = x / 255.
    x_i = torch.from_numpy(x.transpose((0, 3, 1, 2)))
    return torchvision.utils.make_grid(x_i)


def load_loglikelihood(event_acc, log10=True):
    loglikelihood_tags = [sca for sca in event_acc.Tags()['scalars'] if 'loglikelihood' in sca]

    ll_list = list()
    sigma_ll = list()
    denom = np.log(10) if log10 else 1.
    if len(loglikelihood_tags) == 0: return ll_list, sigma_ll
    w_times, step_nums, sigma = zip(*event_acc.Scalars('sigma_list'))
    for tag in loglikelihood_tags:
        w_times, step_nums, vals = zip(*event_acc.Scalars(tag))

        ll_list.append(vals[-1] / denom)
        sigma_ll.append(sigma[step_nums[-1]])

    return ll_list, sigma_ll
