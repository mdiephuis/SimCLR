import argparse
import torch
import torch.nn as nn
from torch.optim import SGD
from tensorboardX import SummaryWriter
import os

from models import *
from utils import *
from data import *


class contrastive_loss(nn.Module):
    def __init__(self, tau=1, normalize=False):
        super(contrastive_loss, self).__init__()
        self.tau = tau
        self.normalize = normalize

    def forward(self, xi, xj):

        x = torch.cat((xi, xj), dim=0)

        is_cuda = x.is_cuda
        sim_mat = torch.mm(x, x.T)
        if self.normalize:
            sim_mat_denom = torch.mm(torch.norm(x, dim=1).unsqueeze(1), torch.norm(x, dim=1).unsqueeze(1).T)
            sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)

        sim_mat = torch.exp(sim_mat / self.tau)

        # no diag because it's not diffrentiable -> sum - exp(1 / tau)
        # diag_ind = torch.eye(xi.size(0) * 2).bool()
        # diag_ind = diag_ind.cuda() if use_cuda else diag_ind

        # sim_mat = sim_mat.masked_fill_(diag_ind, 0)

        # top
        if self.normalize:
            sim_mat_denom = torch.norm(xi, dim=1) * torch.norm(xj, dim=1)
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / sim_mat_denom / self.tau)
        else:
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / self.tau)

        sim_match = torch.cat((sim_match, sim_match), dim=0)

        norm_sum = torch.exp(torch.ones(x.size(0)) / self.tau)
        norm_sum = norm_sum.cuda() if is_cuda else norm_sum
        loss = torch.mean(-torch.log(sim_match / (torch.sum(sim_mat, dim=-1) - norm_sum)))

        return loss


def contrastive_loss2(out_1, out_2):
    out = torch.cat([out_1, out_2], dim=0)
    batch_size = 32
    temperature = 0.5
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss


# Enable CUDA, set tensor type and device
use_cuda = True
if use_cuda:
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda")
    torch.cuda.set_device(1)
    print('GPU')
else:
    dtype = torch.FloatTensor
    device = torch.device("cpu")


in_channels = 3
# Get train and test loaders for dataset
train_transforms = cifar_train_transforms()
test_transforms = cifar_test_transforms()

loader = Loader('CIFAR10C', '../data', True, 32, train_transforms, test_transforms, None, use_cuda)
train_loader = loader.train_loader
test_loader = loader.test_loader


loss_func = contrastive_loss(0.5)

model = resnet50_cifar(128).type(dtype)
init_weights(model)
model.eval()

for x_i, x_j, _ in train_loader:

    x_i = x_i.cuda() if use_cuda else x_i
    x_j = x_j.cuda() if use_cuda else x_j

    _, z_i = model(x_i)
    _, z_j = model(x_j)

    loss1 = loss_func(z_i, z_j).item()

    loss2 = contrastive_loss2(z_i, z_j).detach().item()

    print('Loss 1 vs Loss 2: {:.4f} - {:.4f}'.format(loss1, loss2))
