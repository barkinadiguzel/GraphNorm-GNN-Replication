import torch.nn as nn


class InstanceNormGNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.InstanceNorm1d(dim, affine=True)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2)
