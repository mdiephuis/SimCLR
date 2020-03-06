import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from tensorboardX import SummaryWriter

from models import *
from utils import *
from data import *
from loss import *

parser = argparse.ArgumentParser(description='SIMCLR')

parser.add_argument('--uid', type=str, default='SimCLR',
                    help='Staging identifier (default: SimCLR)')
parser.add_argument('--dataset-name', type=str, default='CIFAR10C',
                    help='Name of dataset (default: CIFAR10C')
parser.add_argument('--data-dir', type=str, default='data',
                    help='Path to dataset (default: data')
parser.add_argument('--feature-size', type=int, default=128,
                    help='Feature output size (default: 128')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input training batch-size')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of training epochs (default: 15)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 1e-3')
parser.add_argument("--decay-lr", default=0.75, action="store", type=float,
                    help='Learning rate decay (default: 0.75')
parser.add_argument('--tau', default=0.5, type=float,
                    help='Tau temperature smoothing (default 0.5)')

parser.add_argument('--log-dir', type=str, default='runs',
                    help='logging directory (default: runs)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables cuda (default: False')

args = parser.parse_args()

# Set cuda
use_cuda = not args.no_cuda and torch.cuda.is_available()

if use_cuda:
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda:0")
    print('GPU')
else:
    dtype = torch.FloatTensor
    device = torch.device("cpu")

# Setup tensorboard
use_tb = args.log_dir is not None
log_dir = args.log_dir

# Logger
if use_tb:
    logger = SummaryWriter(comment='_' + args.uid + '_' + args.dataset_name)


if args.dataset_name == 'CIFAR10C':
    in_channels = 3
    # Get train and test loaders for dataset
    train_transforms = cifar_train_transforms()
    test_transforms = cifar_test_transforms()

    loader = Loader('CIFAR10C', '../data', True, args.batch_size, train_transforms, test_transforms, None, use_cuda)
    train_loader = loader.train_loader
    test_loader = loader.test_loader


# train validate
def train_validate(model, loader, optimizer, is_train, use_cuda):

    loss_func = contrastive_loss(tau=args.tau)

    data_loader = loader.train_loader if is_train else loader.test_loader

    model.train() if is_train else model.eval()

    batch_loss = 0
    for batch_idx, (x_i, x_j, _) in enumerate(data_loader):

        x_i = x_i.cuda() if use_cuda else x_i
        x_j = x_j.cuda() if use_cuda else x_j

        f_i = model(x_i)
        f_j = model(x_j)

        print(f_i.size())

        loss = loss_func(f_i, f_j)

        if is_train:
            model.zero_grad()
            loss.backward()
            optimizer.step()

        batch_loss = loss.item()

    return batch_loss / (batch_idx + 1)


def execute_graph(model, loader, optimizer, schedular, epoch, use_cuda):
    t_loss = train_validate(model, loader, optimizer, True, use_cuda)
    v_loss = train_validate(model, loader, optimizer, False, use_cuda)

    schedular.step(v_loss)

    if use_tb:
        logger.add_scalar(log_dir + '/train-loss', t_loss, epoch)
        logger.add_scalar(log_dir + '/valid-loss', v_loss, epoch)

    print('Epoch: {} Train loss {}'.format(epoch, t_loss))
    print('Epoch: {} Valid loss {}'.format(epoch, v_loss))

    return


# model definition
model = resnet50_cifar(args.feature_size).type(dtype)

# init?

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay_lr)
schedular = ExponentialLR(optimizer, gamma=args.decay_lr)


# Main training loop
for epoch in range(args.epochs):
    execute_graph(model, loader, optimizer, schedular, epoch, use_cuda)


# TensorboardX logger
logger.close()

# save model / restart training
