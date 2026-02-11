import torch
import torch.nn as nn
from .aggregation import normalize_adjacency


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x, adj):
        Q = normalize_adjacency(adj)
        x = Q @ x
        x = self.linear(x)
        return x
