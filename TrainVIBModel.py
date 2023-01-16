import math

import torch
import torch.nn as nn
import VariationalParameterizations as VIB
from SSEPDatasetGenerator import SEPGenerator


class VIBModel(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.encoder = VIB.StochasticEncoder(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim)
        self.decoder = VIB.VariationalDecoder2(in_dim=out_dim, out_dim=in_dim, hidden_dim=hidden_dim)
        self.marginal_embedding = VIB.VariationalMarginal(dim=out_dim)

    def forward(self, x, y, beta):
        Z = self.encoder.get_distribution(x)
        losses = -self.decoder.log_probs(Z.rsample(), y) + \
                 beta * torch.distributions.kl.kl_divergence(Z, self.marginal_embedding.get_distribution())
        return losses.mean()

inverse_density = 2
state_space_size = 8
model = VIBModel(state_space_size, 8, 8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for i in range(3000):
    gen = SEPGenerator(space_size=8, num_samples=100, time_period=0.125, inverse_density=inverse_density)
    initial_state = gen.data.clone()
    gen.run()
    final_state = gen.data.clone()
    optimizer.zero_grad()

    loss = model(initial_state.float(), final_state.float(), 0)
    loss.backward()
    optimizer.step()
    print(i, -loss.detach().cpu().item() + math.log(math.comb(state_space_size,state_space_size // inverse_density)))
