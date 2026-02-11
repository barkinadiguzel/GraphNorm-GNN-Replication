import torch
import torch.nn as nn


class GraphNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.alpha = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        centered = x - self.alpha * mean

        var = centered.var(dim=1, unbiased=False, keepdim=True)
        std = torch.sqrt(var + self.eps)

        out = centered / std
        out = out * self.weight + self.bias
        return out
