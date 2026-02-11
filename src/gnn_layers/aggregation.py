import torch


def normalize_adjacency(adj):
    deg = adj.sum(dim=-1)
    deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
    D_inv_sqrt = torch.diag_embed(deg_inv_sqrt)

    Q = D_inv_sqrt @ adj @ D_inv_sqrt
    return Q
