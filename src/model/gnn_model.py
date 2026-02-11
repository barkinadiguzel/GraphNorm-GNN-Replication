import torch.nn as nn

from src.gnn_layers.gcn_layer import GCNLayer
from src.normalization.graphnorm import GraphNorm


class GNNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()

        self.linear = nn.Linear(in_dim, hidden_dim)
        self.gcn = GCNLayer(hidden_dim, hidden_dim)
        self.norm = GraphNorm(hidden_dim)
        self.act = nn.ReLU()

    def forward(self, x, adj):
        x = self.linear(x)
        x = self.gcn(x, adj)
        x = self.norm(x)
        x = self.act(x)
        return x
