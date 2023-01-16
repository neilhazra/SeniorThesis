import torch
import torch.nn
import torch.nn as nn


class BoltzmannEncoderDecoder(nn.Module):
    def __init__(self, batch_size, in_dim, out_dim):
        super().__init__()
        self.batch_size = batch_size
        self.in_dim = in_dim  # dimension of y
        self.out_dim = out_dim  # dimension of z

        self.b = nn.Parameter(torch.zeros((1, self.out_dim)))
        self.c = nn.Parameter(torch.zeros((1, self.in_dim)))
        self.W = nn.Parameter(torch.zeros((1, self.out_dim, self.in_dim)))
        self.init()

    def init(self):
        torch.nn.init.xavier_uniform(self.b)
        torch.nn.init.xavier_uniform(self.c)
        torch.nn.init.xavier_uniform(self.W)

    def energy(self, z, y):
        e1 = -self.b.view(1, 1, self.out_dim) @ z.view(self.batch_size, self.out_dim, 1)
        e2 = -self.c.view(1, 1, self.in_dim) @ y.view(self.batch_size, self.in_dim, 1)
        e3 = -self.b.view(1, 1, self.out_dim) @ self.W @ y.view(self.batch_size, self.in_dim, 1)
        return (e1 + e2 + e3).squeeze(-1)

    def conditional_log_probability_y_given_z(self, z, y):
        return torch.nn.functional.logsigmoid((2 * y.view(self.batch_size, self.in_dim, 1) - 1) *
                                              (self.c.unsqueeze(-1) + torch.transpose(self.W, 1, 2) @
                                               z.view(self.batch_size, self.out_dim, 1))).sum(dim=-2)

    def conditional_log_probability_z_given_y(self, z, y):
        return torch.nn.functional.logsigmoid((2 * z.view(self.batch_size, self.out_dim, 1) - 1) *
                                              (self.b.unsqueeze(-1) + self.W @ y.view(self.batch_size, self.in_dim,
                                                                                      1))).sum(dim=-2)

    # simple since factorial distribution
    def encoder_sample(self, y):
        thresholds = torch.sigmoid(
            (self.b.unsqueeze(-1) + self.W @ y.view(self.batch_size, self.in_dim, 1)))
        return (torch.rand_like(thresholds) < thresholds).float().squeeze(-1)

    def decoder_sample(self, z):
        thresholds = torch.sigmoid(
            (self.c.unsqueeze(-1) + torch.transpose(self.W, 1, 2) @ z.view(self.batch_size, self.out_dim, 1)))
        return (torch.rand_like(thresholds) < thresholds).float().squeeze(-1)


class EnergyBasedModelEmbeddingDynamics(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # dimension of y
        self.energy_function = nn.Sequential(
            nn.Linear(self.dim + self.dim, 2*self.dim),
            nn.ReLU(),
            nn.Linear(2*self.dim, 1)
        )

    # for small state spaces it is possible to manually compute the partition function
    def log_partition_function(self, W):
        W = W.view(-1, 1, self.dim).expand(-1, 2 ** self.dim, -1)
        Z = torch.arange(0, 2 ** self.dim).unsqueeze(-1).bitwise_and(2 ** torch.arange(
            self.dim)).ne(0).unsqueeze(0).expand(W.shape[0], -1, -1)
        log_probs = self.unnormalized_log_probs_z_given_w_double_batched(Z, W)
        partitions = torch.logsumexp(log_probs, dim=1)
        return partitions

    def normalized_log_probabilities_z_given_w(self, z, w):
        return self.unnormalized_log_probs_z_given_w(z, w) - self.log_partition_function(w)

    def _energy(self, z, w):
        return self.energy_function(torch.concat([z, w], dim=-1))

    def unnormalized_log_probs_z_given_w(self, z, w):
        return -self._energy(z, w)

    def unnormalized_log_probs_z_given_w_double_batched(self, z, w):
        batch_dim = z.shape[0]
        sample_dim = z.shape[1]
        dim = z.shape[2]
        energy = self._energy(z.reshape(-1, dim), w.reshape(-1, dim)).reshape(batch_dim, sample_dim, 1)
        return -energy
