import math
import torch
import torch.nn as nn
import DiscreteVariationalParameterizations as DVP
from SSEPDatasetGenerator import SEPGenerator
from MetropolisHastings import BatchedConditionalMetropolisSampler


class EmbeddingMI1(nn.Module):
    def __init__(self, batch_size, in_dim, out_dim):
        super().__init__()
        self.encoder_decoder = DVP.BoltzmannEncoderDecoder(batch_size=batch_size, in_dim=in_dim, out_dim=out_dim)
        self.embedding_dynamics = DVP.EnergyBasedModelEmbeddingDynamics(dim=out_dim)
        self.metropolis_sampler = BatchedConditionalMetropolisSampler(batch_size=batch_size, num_samples=50, k=2,
                                                                      mixing_time=200,
                                                                      conditional_distribution=self.embedding_dynamics)

    # this is not really the objective but it's gradient is approximately equal to that of the objectives
    # i.e. this is off by a constant, but is differentiable efficiently
    def quasi_objective(self, x, y):
        W = self.encoder_decoder.encoder_sample(x)
        Z = self.encoder_decoder.encoder_sample(y)
        l1 = self.encoder_decoder.conditional_log_probability_y_given_z(Z, y) - self.encoder_decoder. \
            conditional_log_probability_z_given_y(Z, y)
        l2 = self.embedding_dynamics.unnormalized_log_probs_z_given_w(Z, W)
        l3 = -self.metropolis_sampler.estimate_conditional_expected_value(W)
        quasi_objective = l1 + l2 + l3  # quasi since this is off by a constant (we get around the intractable Z)
        return -quasi_objective.mean()

    def full_objective(self, x, y):
        W = self.encoder_decoder.encoder_sample(x)
        Z = self.encoder_decoder.encoder_sample(y)
        objective = self.encoder_decoder.conditional_log_probability_y_given_z(Z, y) - self.encoder_decoder. \
            conditional_log_probability_z_given_y(Z,
                                                  y) + self.embedding_dynamics.normalized_log_probabilities_z_given_w(Z,
                                                                                                                      W)
        return -objective.mean()


def run_dim_red_process(state_space, inverse_density, embedding_space_size, time_period, num_steps = 8000):
    model = EmbeddingMI1(512, state_space, embedding_space_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for i in range(num_steps):
        gen = SEPGenerator(space_size=state_space, num_samples=512, time_period=time_period,
                           inverse_density=inverse_density)
        initial_state = gen.data.clone()
        gen.run()
        final_state = gen.data.clone()
        # print((initial_state != final_state).sum(dim = -1))
        optimizer.zero_grad()
        loss = model.quasi_objective(initial_state.float(), final_state.float())
        loss.backward()
        # this is for debugging purposes and only possible with small state spaces
        if i % 10 == 0:
            with torch.no_grad():
                score = model.full_objective(initial_state.float(), final_state.float())
            print('Iteration', i, 'I(W,Z) > ',
                  -score.detach().cpu().item() + math.log(math.comb(state_space, state_space // inverse_density)))
        else:
            print('Iteration', i)
        optimizer.step()

experiments = [(10, 2, 10, 0), (16, 2, 16, 0.05), (16, 2, 16, 0.1), (16, 2, 16, 0.15),
                (16, 2, 8, 0), (16, 2, 8, 0.1), (16, 2, 8, 0.15), (16, 2, 8, 0.25),
               (16, 2, 4, 0), (16, 2, 4, 0.1), (16, 2, 4, 0.15), (16, 2, 4, 0.25)]

results = []
for i, params in enumerate(experiments):
    print('Running Experiment', i)
    results.append(run_dim_red_process(*params))
print(results)
