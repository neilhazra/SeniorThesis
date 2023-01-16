import torch
import torch.nn


def initialize_params(in_dim, out_dim, hidden_dim):
    b1 = torch.zeros(hidden_dim)
    b2 = torch.zeros(out_dim)
    A1 = torch.zeros((hidden_dim, in_dim))
    A2 = torch.zeros((out_dim, hidden_dim))

    torch.nn.init.xavier_uniform(A1)
    torch.nn.init.xavier_uniform(A2)
    torch.nn.init.xavier_uniform(b1)
    torch.nn.init.xavier_uniform(b2)
    return A1, A2, b1, b2


def compute_energy_func(z, w, A1, A2, b1, b2):
    concat = torch.cat((z, w), dim=-1)
    linear1 = torch.nn.functional.linear(concat, A1, b1)
    relu1 = torch.nn.functional.relu(linear1)
    linear2 = torch.nn.functional.linear(relu1, A2, b2)
    return linear2


def unnormalized_log_probs_z_given_w_double_batched(z, w, A1, A2, b1, b2):
    batch_dim = z.shape[0]
    sample_dim = z.shape[1]
    dim = z.shape[2]
    energy = compute_energy_func(z.reshape(-1, dim), w.reshape(-1, dim), A1, A2, b1, b2).reshape(batch_dim, sample_dim,
                                                                                                 1)
    return -energy