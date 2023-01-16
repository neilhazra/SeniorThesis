import math

import torch
import torch.nn as nn
import VariationalParameterizations as VIB
from SSEPDatasetGenerator import SEPGenerator


class EmbeddingMI1(nn.Module):
    # in this example the decoder represents q(x|w)
    # and we model the embedding dynamics r(W|Z)
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.encoder = VIB.StochasticEncoder(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim)
        self.decoder = VIB.VariationalDecoder2(in_dim=out_dim, out_dim=in_dim, hidden_dim=hidden_dim)
        self.embedding_dynamics = VIB.VariationalEmbeddingDynamics(embedding_size=out_dim, hidden_dim=hidden_dim)

    def forward(self, x, y):
        num_samples_expectation_estimate = 1
        batch_size = x.shape[0]
        space_size = x.shape[1]
        W = self.encoder.get_distribution(x)
        Z = self.encoder.get_distribution(y)
        w_samples = W.rsample((1,num_samples_expectation_estimate)).squeeze(0)
        z_samples = Z.rsample((1,num_samples_expectation_estimate)).squeeze(0)
        x_expanded = x.expand(num_samples_expectation_estimate, -1, -1)
        objective = W.entropy() + self.decoder.log_probs(w_samples.view(-1,space_size), x_expanded.reshape(-1, space_size)).view(-1,batch_size).mean(axis=0) \
                    + self.embedding_dynamics.get_log_prob(z_samples.view(-1,space_size), w_samples.view(-1,space_size)).view(-1, batch_size).mean(axis=0)
        return -objective.mean()

class EmbeddingMI2(nn.Module):
    # in this example the decoder represents q(y|z)
    # and we model the embedding dynamics r(Z|W)
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.encoder = VIB.StochasticEncoder(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim)
        self.decoder = VIB.VariationalDecoder2(in_dim=out_dim, out_dim=in_dim, hidden_dim=hidden_dim)
        self.embedding_dynamics = VIB.VariationalEmbeddingDynamics(embedding_size=out_dim, hidden_dim=hidden_dim)

    def forward(self, x, y):
        W = self.encoder.get_distribution(x)
        Z = self.encoder.get_distribution(y)
        print(x[0])
        objective = 0 + self.decoder.log_probs(Z.mean, y) + \
                    self.embedding_dynamics.get_log_prob(W.mean, Z.mean)
        return -objective.mean()


inverse_density = 2
state_space = 8
model2 = EmbeddingMI2(state_space, 8, 8)
optimizer = torch.optim.Adam(model2.parameters(), lr=0.0001)

for i in range(3000):
    gen = SEPGenerator(space_size=8, num_samples=1024, time_period=0, inverse_density=inverse_density)
    initial_state = gen.data.clone()
    gen.run()
    final_state = gen.data.clone()

    optimizer.zero_grad()

    loss = model2(initial_state.float(), final_state.float())
    loss.backward()
    optimizer.step()
    print('Iteration', i, 'I(W,Z) > ', -loss.detach().cpu().item() + math.log(math.comb(state_space,state_space // inverse_density)))
