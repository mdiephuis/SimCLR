import argparse
import torch
import torch.optim as optim
import torch.distributions as D
from torchlars import LARS
import numpy as np

from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
import time

from simple_models import *
from models import *
from utils import *
from data import *
from loss import *
from scheduler import *
from visdom_grapher import VisdomGrapher

parser = argparse.ArgumentParser(description='MIB')

parser.add_argument('--uid', type=str, default='MIB',
                    help='Staging identifier (default: MIB)')
parser.add_argument('--dataset-name', type=str, default='MNISTC',
                    help='Name of dataset (default: MNISTC')
parser.add_argument('--data-dir', type=str, default='data',
                    help='Path to dataset (default: data')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input training batch-size')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of training epochs (default: 150)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 1e-3')
parser.add_argument('--log-dir', type=str, default='runs',
                    help='logging directory (default: runs)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables cuda (default: False')
parser.add_argument('--load-model', type=str, default=None,
                    help='Load model to resume training for (default None)')
parser.add_argument('--device-id', type=int, default=0,
                    help='GPU device id (default: 0')
# Visdom
parser.add_argument('--visdom-url', type=str, default=None,
                    help='visdom url, needs http, e.g. http://localhost (default: None)')
parser.add_argument('--visdom-port', type=int, default=8097,
                    help='visdom server port (default: 8097')

args = parser.parse_args()

# Set cuda
use_cuda = not args.no_cuda and torch.cuda.is_available()

if use_cuda:
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda")
    torch.cuda.set_device(args.device_id)
    print('GPU')
else:
    dtype = torch.FloatTensor
    device = torch.device("cpu")

# Setup tensorboard and visdom
use_visdom = args.visdom_url is not None
use_tb = args.log_dir is not None
log_dir = args.log_dir

# Setup asset directories
if not os.path.exists('models'):
    os.makedirs('models')

if not os.path.exists('runs'):
    os.makedirs('runs')

# Logger
if use_tb:
    logger = SummaryWriter(comment='_' + args.uid)

# Visdom init
if use_visdom:
    vis = VisdomGrapher(args.uid + '_' + args.dataset_name, args.visdom_url, args.visdom_port)


if args.dataset_name == 'CIFAR10C':
    in_channels = 3
    # Get train and test loaders for dataset
    train_transforms = cifar_train_transforms()
    test_transforms = cifar_test_transforms()
    target_transforms = None

if args.dataset_name == 'MNISTC':
    train_transforms = mnist_train_transforms()
    test_transforms = mnist_test_transforms()
    target_transforms = None

loader = Loader(args.dataset_name, args.data_dir, True, args.batch_size, train_transforms, test_transforms, target_transforms, use_cuda)
train_loader = loader.train_loader
test_loader = loader.test_loader


# train validate
def train_validate(encoder, mi_estimator, loader, E_optim, MI_optim, beta_scheduler, is_train, epoch, use_cuda):

    data_loader = loader.train_loader if is_train else loader.test_loader

    if is_train:
        encoder.train()
        mi_estimator.train()
        encoder.zero_grad()
        mi_estimator.zero_grad()
    else:
        encoder.eval()
        mi_estimator.eval()

    desc = 'Train' if is_train else 'Validation'

    total_loss = 0.0

    tqdm_bar = tqdm(data_loader)
    for i, (xi, xj, _) in enumerate(tqdm_bar):

        beta = beta_scheduler(i)

        xi = xi.cuda() if use_cuda else xi
        xj = xj.cuda() if use_cuda else xj

        # Encoder forward
        p_xi_vi = encoder(xi)
        p_xj_vj = encoder(xj)

        # Sample
        zi = p_xi_vi.rsample()
        zj = p_xj_vj.rsample()

        # MI gradient
        mi_grad, mi_out = mi_estimator(zi, zj)
        mi_grad *= -1

        mi_estimator.zero_grad()
        mi_grad.backward(retain_graph=True)
        MI_optim.step()

        # Symmetric KL
        kl_1_2 = p_xi_vi.log_prob(zi) - p_xj_vj.log_prob(zi)
        kl_2_1 = p_xj_vj.log_prob(zj) - p_xi_vi.log_prob(zj)
        skl = (kl_1_2 + kl_2_1).mean() / 2.

        loss = mi_grad + beta * skl

        encoder.zero_grad()
        loss.backward()
        E_optim.step()

        total_loss += loss.item()

        tqdm_bar.set_description('{} Epoch: [{}] Loss: {}'.format(desc, epoch, loss.item()))

    return total_loss / (len(data_loader.dataset))


def execute_graph(encoder, mi_estimator, loader, E_optim, MI_optim, beta_scheduler, epoch, use_cuda):
    t_loss = train_validate(encoder, mi_estimator, loader, E_optim, MI_optim, beta_scheduler, True, epoch, use_cuda)
    v_loss = train_validate(encoder, mi_estimator, loader, E_optim, MI_optim, beta_scheduler, False, epoch, use_cuda)

    if use_tb:
        logger.add_scalar(log_dir + '/train-loss', t_loss, epoch)
        logger.add_scalar(log_dir + '/valid-loss', v_loss, epoch)

    if use_visdom:
        # Visdom: update training and validation loss plots
        vis.add_scalar(t_loss, epoch, 'Training loss', idtag='train')
        vis.add_scalar(v_loss, epoch, 'Validation loss', idtag='valid')

    # print('Epoch: {} Train loss {}'.format(epoch, t_loss))
    # print('Epoch: {} Valid loss {}'.format(epoch, v_loss))

    return v_loss


# Simple model for MNISTC testing
encoder = MNIST_Encoder(28 * 28, 64).type(dtype)
mi_estimator = MiEstimator(64, 64, 128).type(dtype)

E_optim = optim.Adam(encoder.parameters(), lr=1e-3)
MI_optim = optim.Adam(mi_estimator.parameters(), lr=1e-3)

# Beta scheduler
beta_start_value = 1e-3
beta_end_value = 1.0
beta_n_iterations = 100000
beta_start_iteration = 50000

beta_scheduler = ExponentialScheduler(start_value=beta_start_value, end_value=beta_end_value,
                                      n_iterations=beta_n_iterations, start_iteration=beta_start_iteration)


# Main training loop
best_loss = np.inf

# Resume training
if args.load_model is not None:
    if os.path.isfile(args.load_model):
        checkpoint = torch.load(args.load_model)
        encoder.load_state_dict(checkpoint['encoder'])
        E_optim.load_state_dict(checkpoint['E_optim'])
        MI_optim.load_state_dict(checkpoint['MI_optim'])
        best_loss = checkpoint['val_loss']
        epoch = checkpoint['epoch']
        print('Loading model: {}. Resuming from epoch: {}'.format(args.load_model, epoch))
    else:
        print('Model: {} not found'.format(args.load_model))

for epoch in range(args.epochs):
    v_loss = execute_graph(encoder, mi_estimator, loader, E_optim, MI_optim, beta_scheduler, epoch, use_cuda)

    if v_loss < best_loss:
        best_loss = v_loss
        print('Writing model checkpoint')
        state = {
            'epoch': epoch,
            'encoder': encoder.state_dict(),
            'E_optim': E_optim.state_dict(),
            'MI_optim': MI_optim.state_dict(),
            'val_loss': v_loss
        }
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M', t)
        file_name = 'models/{}_{}_{}_{:04.4f}.pt'.format(timestamp, args.uid, epoch, v_loss)

        torch.save(state, file_name)


# TensorboardX logger
logger.close()

# save model / restart training
