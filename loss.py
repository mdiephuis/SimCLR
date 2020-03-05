import torch
import torch.nn as nn


class contrastive_loss(nn.Module):
    def __init__(self, tau=1, use_cuda):
        self.tau = tau
        self.use_cuda = use_cuda

    def forward(self, x):
        b_sz = x.size(0) // 2

        sim_mat_nom = torch.mm(x, x.T)
        sim_mat_denom = torch.mm(torch.norm(x, dim=1).unsqueeze(1), torch.norm(x, dim=1).unsqueeze(1).T)

        sim_mat = sim_mat_nom / sim_mat_denom.clamp(min=1e-16)
        sim_mat = torch.exp(sim_mat / self.tau)

        # getting rid of diag
        diag_ind = torch.eye(b_sz * 2).bool()
        sim_mat = sim_mat.masked_fill_(diag_ind, 0)

        loss_nom = torch.zeros(b_sz).cuda() if self.use_cuda else torch.zeros(b_sz)
        loss_denom = torch.zeros(b_sz).cuda() if self.use_cuda else torch.zeros(b_sz)

        for i in range(b_sz):
            loss_nom[i] = sim_mat[i][i + b_sz]
            loss_denom[i] = torch.sum(sim_mat[i, :])

        loss = torch.mean(-torch.log(loss_nom / loss_denom))

        return loss
