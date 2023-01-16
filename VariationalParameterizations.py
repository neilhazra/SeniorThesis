import torch
import torch.nn
import torch.nn as nn


class StochasticEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.hidden = hidden_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.convolutional_network = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
        )

        # predict mean value of multivariate gaussian distribution
        self.mean_network = nn.Sequential(
            nn.Linear(self.hidden, out_dim)
        )
        # predict the upper triangular terms for the covariance matrix
        self.cholesky_nondiag_sigmas_network = nn.Sequential(
            nn.Linear(self.hidden, out_dim * out_dim),  # 2
        )
        # predict the diagonal elements, these must be non zero to ensure invertibility
        self.cholesky_diag_sigmas_network = nn.Sequential(
            nn.Linear(self.hidden, out_dim)
        )

    def get_distribution(self, x):
        parameters = self.convolutional_network(x)
        means = self.mean_network(parameters)
        cholesky_lower_triangular = torch.tril(
            self.cholesky_nondiag_sigmas_network(parameters).view(-1, self.out_dim, self.out_dim), diagonal=-1)

        cholesky_diag = torch.diag_embed(
            torch.exp(self.cholesky_diag_sigmas_network(parameters)).view(-1, self.out_dim))

        cholesky_sigmas = cholesky_diag + cholesky_lower_triangular
        return torch.distributions.MultivariateNormal(loc=means, scale_tril=cholesky_sigmas)

#TODO THIS MIGHT BE TOO NAIVE, since everything is independent here
class VariationalDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden = hidden_dim
        self.convolutional_upsample_network = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.out_dim),
            nn.Sigmoid()
        )

    def log_probs(self, z, y):
        probs = self.convolutional_upsample_network(z)
        inverse_probs = 1 - probs
        return torch.where(y.bool(), torch.log(probs), torch.log(inverse_probs)).sum(dim = -1)

def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1).long()


def dec2bin(x, bits):
    # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

#This is computationally intractible for larger systmes
class VariationalDecoder2(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_digits = out_dim
        self.out_dim = 2**self.out_digits
        self.hidden = hidden_dim
        self.convolutional_upsample_network = nn.Sequential(
            nn.Linear(self.in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.out_dim),
            nn.LogSoftmax()
        )

    def log_probs(self, z, y):
        logits = self.convolutional_upsample_network(z)
        _, idx = torch.max(logits, dim=-1)
        print(dec2bin(idx[0], 8))
        return logits[torch.arange(0,z.shape[0]), bin2dec(y, self.out_digits)]

#TODO Make this multivariate gaussian or something
class VariationalEmbeddingDynamics(nn.Module):
    def __init__(self, embedding_size, hidden_dim):
        super().__init__()
        self.hidden = hidden_dim
        self.in_dim = embedding_size
        self.out_dim = embedding_size
        self.convolutional_network = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
        )
        # predict mean value of multivariate gaussian distribution
        self.mean_network = nn.Sequential(
            nn.Linear(self.hidden, embedding_size)
        )
        # predict the upper triangular terms for the covariance matrix
        self.cholesky_nondiag_sigmas_network = nn.Sequential(
            nn.Linear(self.hidden, embedding_size * embedding_size),  # 2
        )
        # predict the diagonal elements, these must be non zero to ensure invertibility
        self.cholesky_diag_sigmas_network = nn.Sequential(
            nn.Linear(self.hidden, embedding_size)
        )

    def get_distribution(self, w):
        parameters = self.convolutional_network(w)
        means = w #self.mean_network(parameters)
        cholesky_lower_triangular = torch.tril(
            self.cholesky_nondiag_sigmas_network(parameters).view(-1, self.out_dim, self.out_dim), diagonal=-1)

        cholesky_diag = torch.diag_embed(
            torch.exp(self.cholesky_diag_sigmas_network(parameters)).view(-1, self.out_dim))

        cholesky_sigmas = cholesky_diag #+ cholesky_lower_triangular
        return torch.distributions.MultivariateNormal(loc=means, scale_tril=cholesky_sigmas)

    def get_log_prob(self, w, z):
        conditional_distribution = self.get_distribution(w)
        print(torch.det(conditional_distribution.covariance_matrix).mean())
        return conditional_distribution.log_prob(z)


class VariationalMarginal(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.distribution = torch.distributions.MultivariateNormal(loc=torch.zeros(dim), covariance_matrix=torch.eye(dim))

    def log_probs(self, z):
        return self.distribution.log_prob(z)

    def get_distribution(self):
        return self.distribution




