import argparse
import torch
from torch.optim import SGD
from tensorboardX import SummaryWriter
import os

from models import *
from utils import *
from data import *


parser = argparse.ArgumentParser(description='SIMCLR')

parser.add_argument('--uid', type=str, default='SimCLR',
                    help='Staging identifier (default: SimCLR)')
parser.add_argument('--dataset-name', type=str, default='CIFAR10C',
                    help='Name of dataset (default: CIFAR10C')
parser.add_argument('--data-dir', type=str, default='data',
                    help='Path to dataset (default: data')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input training batch-size')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of training epochs (default: 15)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 1e-3')
parser.add_argument("--decay-lr", default=0.75, action="store", type=float,
                    help='Learning rate decay (default: 0.75')
parser.add_argument('--log-dir', type=str, default='runs',
                    help='logging directory (default: runs)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables cuda (default: False')

args = parser.parse_args()

# Set cuda
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Set tensorboard
use_tb = args.log_dir is not None
log_dir = args.log_dir

# Logger
if use_tb:
    logger = SummaryWriter(comment='_' + args.uid + '_' + args.dataset_name)

# Enable CUDA, set tensor type and device
if args.cuda:
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    print('GPU')
else:
    dtype = torch.FloatTensor
    device = torch.device("cpu")


if args.dataset_name == 'CIFAR10C':
    in_channels = 3
    loader = Loader('CIFAR10C', 'data', True, 32, None, None, use_cuda)


# model definition
model = resnet50().type(dtype)



# train validate


def train_validate(model, optim, loader, epoch, is_train):
    data_loader = loader.train_loader if is_train else loader.test_loader

    model.train() if is_train else model.eval()

    for batch_idx, (x_i, x_j, _) in enumerate(data_loader):

