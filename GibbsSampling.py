import torch
import torch.nn as nn


class BatchedConditionalGibbsSampler(nn.Module):
    def __init__(self, batch_size, num_samples, mixing_time, joint_distribution):
        super().__init__()
        self.joint_distribution = joint_distribution  # this is the conditional distribution for the process, but
        # joint in the gibbs sense as its a distribution over vectors
        # and not individual bits
        self.dim = self.joint_distribution.dim
        self.batch_size = batch_size
        self.mixing_time = mixing_time
        self.num_samples = num_samples

        # buffers allow for automatic movement onto the GPU
        self.register_buffer('initial_guess', (torch.rand(self.num_samples, self.batch_size, self.dim) < 0.25).float())
        self.register_buffer('zeros', torch.zeros(self.num_samples, self.batch_size))
        self.register_buffer('ones', torch.ones(self.num_samples, self.batch_size))

    def estimate_conditional_expected_value(self, z, model_func=None):
        with torch.no_grad():
            w_samples = self.run_batched_gibbs(z)
        z_conditioned = z.expand(self.num_samples, -1, -1)
        return model_func(w_samples, z_conditioned).mean(dim=0)

    # batch_size, num_samples, dimension
    # w is of shape batch_size x dimension
    @torch.no_grad()
    def run_batched_gibbs(self, z):
        x = self.initial_guess  # don't ruin the initial guess TODO maybe we want to???
        conditioned = z.unsqueeze(0).expand(self.num_samples, -1, -1)  # sample_size x batch_size x cond_dim
        for _ in range(self.mixing_time):
            x_prime = x.detach().clone()
            for j in range(self.dim):
                self.gibbs_update(x, x_prime, j, conditioned)  # sample_size x batch_size x dim
            x = x_prime
        self.initial_guess = x.detach().clone()  # initial guess changes throughout iterations
        return x

    @torch.no_grad()
    def gibbs_update(self, x, x_prime, index, conditioned):
        unnormalized_log_probs = lambda w: self.joint_distribution.unnormalized_log_probs_w_given_z_double_batched_(
            conditioned, w)
        index_0 = torch.concat((x_prime[..., :index], x[..., index:]), dim=-1)
        index_1 = index_0.detach().clone()
        index_0[..., index] = self.zeros
        index_1[..., index] = self.ones
        log_likelihood_zero = unnormalized_log_probs(index_0)
        log_likelihood_one = unnormalized_log_probs(index_1)
        x_prime[..., index] = torch.distributions.bernoulli.Bernoulli(
            logits=log_likelihood_one - log_likelihood_zero).sample().squeeze(-1)
