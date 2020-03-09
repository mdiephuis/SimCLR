import argparse
import torch
import torch.optim as optim
import numpy as np

from tensorboardX import SummaryWriter
from tqdm import tqdm
import os


from models import *
from utils import *
from data import *
from loss import *


parser = argparse.ArgumentParser(description='SIMCLR-CLASSI')

parser.add_argument('--uid', type=str, default='SimCLR-CLASSI',
                    help='Staging identifier (default: SimCLR-CLASSI)')
parser.add_argument('--load-model', type=str, default=None,
                    help='Load model for feature extraction (default None)')

parser.add_argument('--dataset-name', type=str, default='CIFAR10',
                    help='Name of dataset (default: CIFAR10')
parser.add_argument('--data-dir', type=str, default='data',
                    help='Path to dataset (default: data')
parser.add_argument('--feature-size', type=int, default=128,
                    help='Feature output size (default: 128')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input training batch-size')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of training epochs (default: 150)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 1e-3')
parser.add_argument("--decay-lr", default=1e-6, action="store", type=float,
                    help='Learning rate decay (default: 1e-6')
parser.add_argument('--log-dir', type=str, default='runs',
                    help='logging directory (default: runs)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables cuda (default: False')

args = parser.parse_args()

# Set cuda
use_cuda = not args.no_cuda and torch.cuda.is_available()

if use_cuda:
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda")
    torch.cuda.set_device(1)
    print('GPU')
else:
    dtype = torch.FloatTensor
    device = torch.device("cpu")

# Setup tensorboard
use_tb = args.log_dir is not None
log_dir = args.log_dir

# Setup asset directories
if not os.path.exists('models'):
    os.makedirs('models')

if not os.path.exists('runs'):
    os.makedirs('runs')

# Logger
if use_tb:
    logger = SummaryWriter(comment='_' + args.uid + '_' + args.dataset_name)

# Datasets
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])


loader = Loader(args.dataset_name, args.data_dir, True, args.batch_size, train_transforms, test_transforms, None, use_cuda)
train_loader = loader.train_loader
test_loader = loader.test_loader


# train validate
def train_validate(classifier_model, feature_model, loader, optimizer, is_train, epoch, use_cuda):

    loss_func = nn.CrossEntropyLoss()

    data_loader = loader.train_loader if is_train else loader.test_loader

    classifier_model.train() if is_train else classifier_model.eval()
    desc = 'Train' if is_train else 'Validation'

    total_loss = 0
    total_acc = 0

    tqdm_bar = tqdm(data_loader)
    for batch_idx, (x, y) in enumerate(tqdm_bar):
        batch_loss = 0
        batch_acc = 0

        x = x.cuda() if use_cuda else x
        y = y.cuda() if use_cuda else y

        # Get features
        f_x, _ = feature_model(x)
        f_x = f_x.detach()

        # Classify features
        y_hat = classifier_model(f_x)

        loss = loss_func(y_hat, y)

        if is_train:
            classifier_model.zero_grad()
            loss.backward()
            optimizer.step()

        # Reporting
        batch_loss = loss.item() / x.size(0)
        total_loss += loss.item()

        pred = y_hat.max(dim=1)[1]
        correct = pred.eq(y).sum().item()
        correct /= y.size(0)
        batch_acc = (correct * 100)
        total_acc += batch_acc

        tqdm_bar.set_description('{} Epoch: [{}] Batch Loss: {:.4f} Batch Acc: {:.4f}'.format(desc, epoch, batch_loss, batch_acc))

    return total_loss / (batch_idx + 1), total_acc / (batch_idx + 1)


def execute_graph(classifier_model, feature_model, loader, optimizer, epoch, use_cuda):
    t_loss, t_acc = train_validate(classifier_model, feature_model, loader, optimizer, True, epoch, use_cuda)
    v_loss, v_acc = train_validate(classifier_model, feature_model, loader, optimizer, False, epoch, use_cuda)

    if use_tb:
        logger.add_scalar(log_dir + '/train-loss', t_loss, epoch)
        logger.add_scalar(log_dir + '/valid-loss', v_loss, epoch)

        logger.add_scalar(log_dir + '/train-acc', t_acc, epoch)
        logger.add_scalar(log_dir + '/valid-acc', v_acc, epoch)

    # print('Epoch: {} Train loss {}'.format(epoch, t_loss))
    # print('Epoch: {} Valid loss {}'.format(epoch, v_loss))

    return v_loss


#
# Load feature extraction model
feature_model = resnet50_cifar(args.feature_size).type(dtype)
feature_model.eval()

if os.path.isfile(args.load_model):
    checkpoint = torch.load(args.load_model)
    feature_model.load_state_dict(checkpoint['model'])
    epoch = checkpoint['epoch']
    print('Loading model: {}, from epoch: {}'.format(args.load_model, epoch))
else:
    print('Model: {} not found'.format(args.load_model))

#
# Define linear classification model
classifier_model = SimpleNet().type(dtype)
optimizer = optim.Adam(classifier_model.parameters(), lr=args.lr, weight_decay=args.decay_lr)


# Main training loop
best_loss = np.inf

for epoch in range(args.epochs):
    execute_graph(classifier_model, feature_model, loader, optimizer, epoch, use_cuda)

# TensorboardX logger
logger.close()

# save model / restart training
