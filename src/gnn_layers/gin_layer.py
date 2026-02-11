import torch
import torch.nn as nn


class GINLayer(nn.Module):
    def __init__(self, in_dim, out_dim, eps=0.0, train_eps=True):
        super().__init__()

        if train_eps:
            self.eps = nn.Parameter(torch.tensor(eps))
        else:
            self.register_buffer("eps", torch.tensor(eps))

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x, adj):
        agg = adj @ x
        out = (1 + self.eps) * x + agg
        return self.mlp(out)
