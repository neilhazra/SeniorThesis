import torch
import torch.nn
import torch.nn as nn


def init_boltzmann_weights(in_dim, out_dim):
    b = nn.Parameter(torch.zeros((1, out_dim)))
    c = nn.Parameter(torch.zeros((1, in_dim)))
    W = nn.Parameter(torch.zeros((1, out_dim, in_dim)))
    torch.nn.init.xavier_uniform(b)
    torch.nn.init.xavier_uniform(c)
    torch.nn.init.xavier_uniform(W)


def compute_boltzmann_energy_func(z, y, b, c, W, in_dim, out_dim, batch_size):
    e1 = -b.view(1, 1, out_dim) @ z.view(batch_size, out_dim, 1)
    e2 = -c.view(1, 1, in_dim) @ y.view(batch_size, in_dim, 1)
    e3 = -b.view(1, 1, out_dim) @ W @ y.view(batch_size, in_dim, 1)
    return e1 + e2 + e3


def conditional_probability_y_given_z_func(z, y, b, c, W):
    return torch.nn.functional.sigmoid((2 * y - 1) @ (c + W @ z)).prod()


def conditional_probability_z_given_y_func(z, y, b, c, W):
    return torch.nn.functional.sigmoid((2 * z - 1) @ (b + W @ y)).prod()
