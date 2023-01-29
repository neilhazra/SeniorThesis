import torch
import torch.nn
import torch.nn as nn
import math


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
        torch.nn.init.xavier_uniform(self.W, gain=1)

    @staticmethod
    def conditional_log_probability_x_given_w(w, x, W, c):
        batch_size = x.shape[0]
        return torch.nn.functional.logsigmoid((2 * x.view(batch_size, -1, 1) - 1) *
                                              (c.unsqueeze(-1) + torch.transpose(W, 1, 2) @
                                               w.view(batch_size, -1, 1))).sum(dim=-2)

    @staticmethod
    def conditional_log_probability_w_given_x(w, x, W, b):
        batch_size = x.shape[0]
        return torch.nn.functional.logsigmoid((2 * w.view(batch_size, -1, 1) - 1) *
                                              (b.unsqueeze(-1) + W @ x.view(batch_size, -1,
                                                                            1))).sum(dim=-2)

    @staticmethod
    def encoder_sample(x, W, b):
        batch_size = x.shape[0]
        thresholds = torch.sigmoid(
            (b.unsqueeze(-1) + W @ x.view(batch_size, -1, 1)))
        return (torch.rand_like(thresholds) < thresholds).float().squeeze(-1)

    @staticmethod
    def decoder_sample(w, W, c):
        batch_size = w.shape[0]
        thresholds = torch.sigmoid(
            (c.unsqueeze(-1) + torch.transpose(W, 1, 2) @ w.view(batch_size, -1, 1)))
        return (torch.rand_like(thresholds) < thresholds).float().squeeze(-1)

    def conditional_log_probability_x_given_w_(self, w, x):
        return BoltzmannEncoderDecoder.conditional_log_probability_x_given_w(w, x, self.W, self.c)

    def conditional_log_probability_w_given_x_(self, w, x):
        return BoltzmannEncoderDecoder.conditional_log_probability_w_given_x(w, x, self.W, self.b)

    def conditional_log_probability_w_given_x_double_batched_(self, w, x):
        dim_0 = w.shape[0]
        dim_1 = w.shape[1]
        return BoltzmannEncoderDecoder.conditional_log_probability_w_given_x(w.reshape(dim_0 * dim_1, -1),
                                                                             x.reshape(dim_0 * dim_1, -1), self.W,
                                                                             self.b).reshape(dim_0, dim_1, -1)

    def conditional_log_probability_w_given_x_double_batched(w, x, W, b):
        dim_0 = w.shape[0]
        dim_1 = w.shape[1]
        return BoltzmannEncoderDecoder.conditional_log_probability_w_given_x(w.reshape(dim_0 * dim_1, -1),
                                                                             x.reshape(dim_0 * dim_1, -1), W,
                                                                             b).reshape(dim_0, dim_1, -1)

    # simple since factorial distribution
    def encoder_sample_(self, x):
        return BoltzmannEncoderDecoder.encoder_sample(x, self.W, self.b)

    # simple since factorial distribution
    def batched_encoder_sample_(self, x):
        dim_0 = x.shape[0]
        dim_1 = x.shape[1]
        return BoltzmannEncoderDecoder.encoder_sample(x.reshape(dim_0 * dim_1, -1), self.W, self.b).reshape(dim_0,
                                                                                                            dim_1, -1)

    # simple since factorial distribution
    @staticmethod
    def batched_encoder_sample(x, W, b):
        dim_0 = x.shape[0]
        dim_1 = x.shape[1]
        return BoltzmannEncoderDecoder.encoder_sample(x.reshape(dim_0 * dim_1, -1), W, b).reshape(dim_0, dim_1, -1)

    def decoder_sample_(self, w):
        return BoltzmannEncoderDecoder.decoder_sample(w, self.W, self.c)


class EnergyBasedModelEmbeddingDynamics(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # dimension of y
        self.linear_1_weight = nn.Parameter(torch.zeros((2 * self.dim, 2 * self.dim)))
        self.linear_1_bias = nn.Parameter(torch.zeros((2 * self.dim)))
        self.linear_2_weight = nn.Parameter(torch.zeros((1, 2 * self.dim)))
        self.linear_2_bias = nn.Parameter(torch.zeros((1)))
        self.init()

    def init(self):
        torch.nn.init.xavier_uniform(self.linear_1_weight)
        torch.nn.init.xavier_uniform(self.linear_2_weight)

    def unnormalized_log_probs_w_given_z_double_batched_(self, z, w):
        return EnergyBasedModelEmbeddingDynamics.unnormalized_log_probs_w_given_z_double_batched(z, w,
                                                                                                 self.linear_1_weight,
                                                                                                 self.linear_1_bias,
                                                                                                 self.linear_2_weight,
                                                                                                 self.linear_2_bias)

    @staticmethod
    def energy_function(i, W1, b1, W2, b2):
        temp = torch.nn.functional.linear(i, W1, b1)
        temp = torch.nn.functional.relu(temp)
        o = torch.nn.functional.linear(temp, W2, b2)
        return o

    @staticmethod
    def _energy(z, w, W1, b1, W2, b2):
        return EnergyBasedModelEmbeddingDynamics.energy_function(torch.concat([z, w], dim=-1), W1, b1, W2, b2)

    @staticmethod
    def unnormalized_log_probs_w_given_z(z, w, W1, b1, W2, b2):
        return -EnergyBasedModelEmbeddingDynamics._energy(z, w, W1, b1, W2, b2)

    def unnormalized_log_probs_w_given_z_(self, z, w):
        return -EnergyBasedModelEmbeddingDynamics._energy(z, w, self.linear_1_weight, self.linear_1_bias,
                                                          self.linear_2_weight, self.linear_2_bias)

    @staticmethod
    def unnormalized_log_probs_w_given_z_double_batched(z, w, W1, b1, W2, b2):
        first_dim = z.shape[0]
        second_dim = z.shape[1]
        dim = z.shape[2]
        energy = EnergyBasedModelEmbeddingDynamics._energy(z.reshape(-1, dim), w.reshape(-1, dim), W1, b1, W2,
                                                           b2).reshape(first_dim, second_dim, 1)
        return -energy

    @staticmethod
    def expected_unnormalized_log_probs_w_given_z(z, w, W1, b1, W2, b2):
        samples_dim = z.shape[0]
        batch_dim = z.shape[1]
        dim = z.shape[2]
        energy = EnergyBasedModelEmbeddingDynamics._energy(z.reshape(-1, dim), w.reshape(-1, dim), W1, b1, W2,
                                                           b2).reshape(samples_dim, batch_dim, 1)
        return -energy.mean(dim=0)

    # for small state spaces it is possible to manually compute the partition function
    @staticmethod
    def log_partition_function(z, W1, b1, W2, b2):
        dim = z.shape[-1]
        z = z.view(-1, 1, dim).expand(-1, 2 ** dim, -1)
        W = torch.arange(0, 2 ** dim, device=z.device).unsqueeze(-1).bitwise_and(2 ** torch.arange(
            dim, device=z.device)).ne(0).unsqueeze(0).expand(z.shape[0], -1, -1)
        log_probs = EnergyBasedModelEmbeddingDynamics.unnormalized_log_probs_w_given_z_double_batched(z, W, W1, b1, W2,
                                                                                                      b2)
        partitions = torch.logsumexp(log_probs, dim=1)
        return partitions

    # the initial state will help create the optimal proposal distribution
    @staticmethod
    @torch.no_grad()
    def estimated_log_partition_function_better(z, initial_state, W, b, _, W1, b1, W2, b2, samples=512):
        z = z.expand(samples, -1, -1)
        initial_state = initial_state.expand(samples, -1, -1)
        w_batched = BoltzmannEncoderDecoder.batched_encoder_sample(initial_state, W, b)
        proposal_log_probs = BoltzmannEncoderDecoder.conditional_log_probability_w_given_x_double_batched(w_batched,
                                                                                                          initial_state,
                                                                                                          W, b)
        log_probs = EnergyBasedModelEmbeddingDynamics.unnormalized_log_probs_w_given_z_double_batched(z, w_batched, W1,
                                                                                                      b1, W2, b2)
        return - math.log(samples) + torch.logsumexp(log_probs - proposal_log_probs, dim=0)

    # do some importance sampling here, note that this is probably good enough for training where the gradient can be noisy
    # and is in generally the right direction, but is definitely not good enough for evaluation
    @staticmethod
    @torch.no_grad()
    def estimated_log_partition_function(z, W1, b1, W2, b2, samples=65536):
        dim = z.shape[-1]
        z = z.expand(samples, -1, -1)
        w = (torch.rand_like(z) < 0.5).float()
        log_probs = EnergyBasedModelEmbeddingDynamics.unnormalized_log_probs_w_given_z_double_batched(z, w, W1, b1, W2,
                                                                                                      b2)
        return dim * math.log(2) - math.log(samples) + torch.logsumexp(log_probs, dim=0)

    def estimated_log_partition_function_(self, z):
        return EnergyBasedModelEmbeddingDynamics.estimated_log_partition_function(z, self.linear_1_weight,
                                                                                  self.linear_1_bias,
                                                                                  self.linear_2_weight,
                                                                                  self.linear_2_bias)

    @staticmethod
    def normalized_log_probabilities_w_given_z(z, w, W1, b1, W2, b2):
        return EnergyBasedModelEmbeddingDynamics.unnormalized_log_probs_w_given_z(z, w, W1, b1, W2,
                                                                                  b2) - EnergyBasedModelEmbeddingDynamics.log_partition_function(
            z, W1, b1, W2, b2)

    def normalized_log_probabilities_w_given_z_(self, z, w):
        return EnergyBasedModelEmbeddingDynamics.normalized_log_probabilities_w_given_z(z, w, self.linear_1_weight,
                                                                                        self.linear_1_bias,
                                                                                        self.linear_2_weight,
                                                                                        self.linear_2_bias)

    def estimated_normalized_log_probabilities_w_given_z_(self, z, w):
        return EnergyBasedModelEmbeddingDynamics.estimated_normalized_log_probabilities_w_given_z(z, w,
                                                                                                  self.linear_1_weight,
                                                                                                  self.linear_1_bias,
                                                                                                  self.linear_2_weight,
                                                                                                  self.linear_2_bias)

    @staticmethod
    def estimated_normalized_log_probabilities_w_given_z(z, w, W1, b1, W2, b2):
        return EnergyBasedModelEmbeddingDynamics.unnormalized_log_probs_w_given_z(z, w, W1, b1, W2,
                                                                                  b2) - EnergyBasedModelEmbeddingDynamics.estimated_log_partition_function(
            z, W1, b1, W2, b2)

    @staticmethod
    def estimated_normalized_log_probabilities_w_given_z_better(z, w, x, W, b, _, W1, b1, W2, b2):
        return EnergyBasedModelEmbeddingDynamics.unnormalized_log_probs_w_given_z(z, w, W1, b1, W2,
                                                                                  b2) - EnergyBasedModelEmbeddingDynamics.estimated_log_partition_function_better(
            z, x, W, b, None, W1, b1, W2, b2)

    @staticmethod
    def estimated_normalized_log_probabilities_w_given_z_better_(z, w, x, model, samples=512):
        z_tilde = z.expand(samples, -1, -1)
        initial_state = x.expand(samples, -1, -1)
        w_tilde = model.encoder_decoder.batched_encoder_sample_(initial_state)
        proposal_log_probs = model.encoder_decoder.conditional_log_probability_w_given_x_double_batched_(w_tilde,
                                                                                                         initial_state)
        log_probs = model.embedding_dynamics.unnormalized_log_probs_w_given_z_double_batched_(z_tilde, w_tilde)
        return model.embedding_dynamics.unnormalized_log_probs_w_given_z_(z, w) + math.log(samples) - torch.logsumexp(
            log_probs - proposal_log_probs, dim=0)

