import torch
import numpy as np
from DiscreteVariationalParameterizations import EnergyBasedModelEmbeddingDynamics


# Note k parameterizes the proposal distribution
class ConditionalMetropolisSampler:
    def __init__(self, batch_size, k, mixing_time, conditional_distribution):
        self.conditional_distribution = conditional_distribution
        self.dim = self.conditional_distribution.dim
        self.k = k
        self.batch_size = batch_size
        self.mixing_time = mixing_time
        self.initial_guess = torch.zeros(self.batch_size, self.dim)

    # tested to be correct
    # in the future we should probably use metropolis hastings
    def run_metropolis(self, w):
        log_probs = lambda z: self.conditional_distribution.unnormalized_log_probs_z_given_w(z, w)
        x = self.initial_guess
        for i in range(self.mixing_time):
            x_prime = self.batch_conditional_sample(x)  # batch_size x dim
            log_probs_ratio = log_probs(x_prime) - log_probs(x)  # batch_size x 1
            threshold = torch.log(torch.rand_like(log_probs_ratio))
            acceptor = threshold <= log_probs_ratio  # batch_size x 1
            x = torch.where(acceptor, x_prime, x)
        return x

    # Parameterized by k
    # choose k bit locations and flip it randomly with probability 1/2
    # this will become the proposal distribution for the Metropolis Algorithm
    def batch_conditional_sample(self, w):
        w = w.clone()
        x = np.arange(self.dim)
        rng = np.random.default_rng()
        perms = rng.permuted(np.tile(x, self.batch_size).reshape(self.batch_size, x.size), axis=1)[:, :self.k]
        flipped_bits = np.random.choice(a=[False, True], size=(self.batch_size, self.k), p=[0.5, 0.5])
        w[np.arange(self.batch_size).reshape(self.batch_size, 1), perms] = torch.tensor(flipped_bits, dtype=torch.float)
        return w


# Note k parameterizes the proposal distribution
class BatchedConditionalMetropolisSampler:
    def __init__(self, batch_size, num_samples, k, mixing_time, conditional_distribution):
        self.conditional_distribution = conditional_distribution
        self.dim = self.conditional_distribution.dim
        self.k = k
        self.batch_size = batch_size
        self.mixing_time = mixing_time
        self.num_samples = num_samples
        self.initial_guess = torch.zeros(self.num_samples, self.batch_size, self.dim)

    def estimate_conditional_expected_value(self, w, model_func=None):
        if model_func is None:
            model_func = self.conditional_distribution.unnormalized_log_probs_z_given_w_double_batched
        z_samples = self.run_batched_metropolis(w)
        w_conditioned = w.expand(self.num_samples, -1, -1)
        return model_func(z_samples, w_conditioned).mean(dim=0)

    # batch_size, num_samples, dimension
    # w is of shape batch_size x dimension
    def run_batched_metropolis(self, w):
        x = self.initial_guess
        for i in range(self.mixing_time):
            x_prime = self.proposal_distribution_sampler(x)  # sample_size x batch_size x dim
            conditioned = w.unsqueeze(0).expand(self.num_samples, -1, -1)
            log_probs_ratio = self.conditional_distribution.unnormalized_log_probs_z_given_w_double_batched(x_prime,
                        conditioned) - self.conditional_distribution.unnormalized_log_probs_z_given_w_double_batched(
                                        x, conditioned)  # sample_size x batch_size x 1
            threshold = torch.log(torch.rand_like(log_probs_ratio))
            acceptor = threshold <= log_probs_ratio  # batch_size x 1
            x = torch.where(acceptor, x_prime, x)
        return x

    # Parameterized by k
    # choose k bit locations and flip it randomly with probability 1/2
    # this will become the proposal distribution for the Metropolis Algorithm
    # this is a symmetric distribution :)
    def proposal_distribution_sampler(self, w):
        w = w.clone()
        x = np.arange(self.dim)
        rng = np.random.default_rng()
        perms = rng.permuted(np.tile(x, (self.num_samples, self.batch_size, 1)))[..., :self.k]
        flipped_bits = np.random.choice(a=[False, True], size=(self.num_samples, self.batch_size, self.k), p=[0.5, 0.5])
        grid_i, grid_j = torch.meshgrid(torch.arange(self.num_samples), torch.arange(self.batch_size), indexing='ij')
        w[grid_i.unsqueeze(-1), grid_j.unsqueeze(-1), perms] = torch.tensor(flipped_bits, dtype=torch.float)
        return w


if __name__ == "__main__":
    model = EnergyBasedModelEmbeddingDynamics(10)  # dimension of the model is 10
    # process 32 full samples at once, use 100 metropolis samples to estimate gradient, proposal function flips 3 bits,
    # run MH for 50 iterations
    sampler = BatchedConditionalMetropolisSampler(32, 100, 3, 50, model)

    # expected_value = sampler.estimate_conditional_expected_value(torch.zeros((32, 10)),
    #                                                             model.unnormalized_log_probs_z_given_w_double_batched)
    #print(expected_value.shape)
