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
        self.decoder = VIB.VariationalDecoder2(in_dim=out_dim, out_dim=in_dim, hidden_dim=1024)
        self.embedding_dynamics = VIB.VariationalEmbeddingDynamics(embedding_size=out_dim, hidden_dim=hidden_dim)

    def forward(self, x, y):
        W = self.encoder.get_distribution(x)
        Z = self.encoder.get_distribution(y)
        objective = W.entropy() + self.decoder.log_probs(W.rsample(), x) + \
                    self.embedding_dynamics.get_log_prob(Z.rsample(), W.rsample())
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
        objective = Z.entropy() + self.decoder.log_probs(Z.rsample(), y) + \
                    self.embedding_dynamics.get_log_prob(W.rsample(), Z.rsample())
        return -objective.mean()


model2 = EmbeddingMI1(16, 16, 16)
model = EmbeddingMI2(16, 8, 12)

optimizer = torch.optim.Adam(model2.parameters(), lr=0.001)

for i in range(1000):
    gen = SEPGenerator(space_size=16, num_samples=512, time_period=0)
    initial_state = gen.data.clone()
    gen.run()
    final_state = gen.data.clone()

    optimizer.zero_grad()

    loss = model2(initial_state.float(), final_state.float())
    loss.backward()
    optimizer.step()
    print('Iteration', i, 'I(W,Z) - H(X) > ', -loss.detach().cpu().item())
