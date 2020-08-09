import torch
import torch.nn as nn
import torch.distributed as dist
import os


def type_tdouble(use_cuda=False):
    return torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor


def one_hot(labels, n_class, use_cuda=False):
    # Ensure labels are [N x 1]
    if len(list(labels.size())) == 1:
        labels = labels.unsqueeze(1)
    mask = type_tdouble(use_cuda)(labels.size(0), n_class).fill_(0)
    # scatter dimension, position indices, fill_value
    return mask.scatter_(1, labels, 1)


def init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_process(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29300'
    dist.init_process_group(backend, rank=rank, world_size=size)
