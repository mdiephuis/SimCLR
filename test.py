import argparse
import torch
from torch.optim import SGD
from tensorboardX import SummaryWriter
import os

from models import *
from utils import *
from data import *


# Enable CUDA, set tensor type and device
use_cuda = False
if use_cuda:
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    print('GPU')
else:
    dtype = torch.FloatTensor
    device = torch.device("cpu")


# Get train and test loaders for dataset
in_channels = 3
# Get train and test loaders for dataset
loader = Loader('CIFAR10C', 'data', True, 32, None, None, use_cuda)
train_loader = loader.train_loader
test_loader = loader.test_loader

model = resnet50().type(dtype)

xi, xj, _ = next(iter(test_loader))

print(xi.size())
print(xj.size())

xi = xi.cuda() if use_cuda else xi
xj = xj.cuda() if use_cuda else xj

hi = model(xi)
