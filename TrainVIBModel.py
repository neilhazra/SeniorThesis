###
### ESTIMATE MUTUAL INFORMATION WITH NON INTERACTING PARTICLES
### SHOULD BE RANDOM WALK/ DIFFUSION
### BRUTE FORCE SOLUTION
###
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
        #print(Z.mean[0], torch.det(Z.covariance_matrix[0]))
        print(x[0])
        print(y[0])
        losses = -self.decoder.log_probs(Z.rsample(), y) #+ \
                 #beta * torch.distributions.kl.kl_divergence(Z, self.marginal_embedding.get_distribution())
        return losses.mean()


model = VIBModel(8, 8, 8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for i in range(3000):
    gen = SEPGenerator(space_size=8, num_samples=100, time_period=0)
    initial_state = gen.data.clone()
    gen.run()
    final_state = gen.data.clone()
    optimizer.zero_grad()

    loss = model(initial_state.float(), final_state.float(), 0)
    loss.backward()
    optimizer.step()
    print(i, -loss.detach().cpu().item() + math.log(math.comb(8,4)))
    #WOOHOO it works!!
