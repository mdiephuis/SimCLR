import torch
import torch.nn as nn


class contrastive_loss(nn.Module):
    def __init__(self, tau=1):
        super(contrastive_loss, self).__init__()
        self.tau = tau

    def forward(self, xi, xj):

        x = torch.cat((xi, xj), dim=0)

        sim_mat_nom = torch.mm(x, x.T)
        sim_mat_denom = torch.mm(torch.norm(x, dim=1).unsqueeze(1), torch.norm(x, dim=1).unsqueeze(1).T)
        sim_mat = sim_mat_nom / sim_mat_denom.clamp(min=1e-16)
        sim_mat = torch.exp(sim_mat / self.tau)

        # getting rid of diag
        diag_ind = torch.eye(xi.size(0) * 2).bool()
        sim_mat = sim_mat.masked_fill_(diag_ind, 0)

        # top
        sim_mat_denom = torch.norm(xi, dim=1) * torch.norm(xj, dim=1)
        sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / sim_mat_denom / self.tau)
        sim_match = torch.cat((sim_match, sim_match), dim=0)

        loss = torch.mean(-torch.log(sim_match / torch.sum(sim_mat, dim=-1)))

        return loss
