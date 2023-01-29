import torch
import torch.nn as nn
import DiscreteVariationalParameterizations as DVP
from SSEPDatasetGenerator import SEPGenerator
from GibbsSampling import BatchedConditionalGibbsSampler
from torch.autograd.functional import vjp
from torch.autograd.function import Function


class EmbeddingMI1(nn.Module):
    def __init__(self, batch_size, in_dim, out_dim):
        super().__init__()
        self.encoder_decoder = DVP.BoltzmannEncoderDecoder(batch_size=batch_size, in_dim=in_dim, out_dim=out_dim)
        self.embedding_dynamics = DVP.EnergyBasedModelEmbeddingDynamics(dim=out_dim)
        self.sampler = BatchedConditionalGibbsSampler(batch_size=batch_size, num_samples=256,  # needs to be tuned
                                                      mixing_time=4,  # seems like this can be low and still work
                                                      joint_distribution=self.embedding_dynamics)
        self.loss_func = CustomMutualInformationLoss.apply

    def NegativeMILoss(self, x, y):
        w = self.encoder_decoder.encoder_sample_(x).detach()
        z = self.encoder_decoder.encoder_sample_(y).detach()
        w_tilde = self.sampler.run_batched_gibbs(z).detach()

        return -self.loss_func(
            z, y, w, x, w_tilde, self.encoder_decoder.W, self.encoder_decoder.b, self.encoder_decoder.c,
            self.embedding_dynamics.linear_1_weight, self.embedding_dynamics.linear_1_bias,
            self.embedding_dynamics.linear_2_weight, self.embedding_dynamics.linear_2_bias)


# we got to do this the old fashioned way because the random sampling does not propagate the
# expected gradient
class CustomMutualInformationLoss(Function):
    # Note that forward, setup_context, and backward are @staticmethods
    def forward(ctx, z, y, w, x, w_tilde, W, b, c, W1, b1, W2, b2):
        p_x_w = DVP.BoltzmannEncoderDecoder.conditional_log_probability_x_given_w(w, x, W, c)
        p_w_x = DVP.BoltzmannEncoderDecoder.conditional_log_probability_w_given_x(w, x, W, b)

        # oom if embedding size is too large
        # r_w_z = DVP.EnergyBasedModelEmbeddingDynamics.normalized_log_probabilities_w_given_z(z, w, W1, b1, W2, b2)
        temp = DVP.EnergyBasedModelEmbeddingDynamics.estimated_normalized_log_probabilities_w_given_z_better(z, w, x, W,
                                                                                                             b, c, W1,
                                                                                                             b1, W2, b2)
        r_w_z = torch.minimum(temp, torch.zeros_like(
            temp))  # importance sampling underestimtaes the partition function so we must clamp at 0

        ctx.save_for_backward(z, y, w, w_tilde, x, W, b, c, W1, b1, W2, b2, p_x_w - p_w_x, r_w_z)

        # ctx.save_for_backward(z,y,w, w_tilde, x,W,b,c,W1, b1, W2, b2, p_x_w - p_w_x, torch.tensor([-1.0], device=z.device))
        # because r is hard to estimate maybe the gradient will roughly be in the right direction this way anyway
        return p_x_w - p_w_x + r_w_z

    @staticmethod
    def backward(ctx, grad_output):
        z, y, w, w_tilde, x, W, b, c, W1, b1, W2, b2, encoder_decoder, embedding = ctx.saved_tensors
        _, (_, _, temp_W_1, temp_c) = vjp(DVP.BoltzmannEncoderDecoder.conditional_log_probability_x_given_w,
                                          (w, x, W, c), grad_output, create_graph=False)
        _, (_, _, temp_W_2, temp_b_1) = vjp(DVP.BoltzmannEncoderDecoder.conditional_log_probability_w_given_x,
                                            (w, x, W, b), grad_output * (embedding + encoder_decoder - 1),
                                            create_graph=False)
        _, (_, _, temp_W_3, temp_b_2) = vjp(DVP.BoltzmannEncoderDecoder.conditional_log_probability_w_given_x,
                                            (z, y, W, b), grad_output * embedding, create_graph=False)
        _, (_, _, temp_linear_W_1, temp_linear_b_1, temp_linear_W_2, temp_linear_b_2) = vjp(
            DVP.EnergyBasedModelEmbeddingDynamics.unnormalized_log_probs_w_given_z, (z, w, W1, b1, W2, b2), grad_output,
            create_graph=False)
        _, (_, _, temp_1_linear_W_1, temp_1_linear_b_1, temp_1_linear_W_2, temp_1_linear_b_2) = vjp(
            DVP.EnergyBasedModelEmbeddingDynamics.expected_unnormalized_log_probs_w_given_z,
            (z.expand(w_tilde.shape[0], -1, -1), w_tilde, W1, b1, W2, b2), grad_output, create_graph=False)
        return None, None, None, None, None, temp_W_1 + temp_W_2 + temp_W_3, temp_b_1 + temp_b_2, temp_c, temp_linear_W_1 - temp_1_linear_W_1, temp_linear_b_1 - temp_1_linear_b_1, temp_linear_W_2 - temp_1_linear_W_2, temp_linear_b_2 - temp_1_linear_b_2


def run_dim_red_process(state_space, inverse_density, embedding_space_size, time_period, num_steps=2000):
    model = EmbeddingMI1(256, state_space, embedding_space_size)
    #model.load_state_dict(torch.load('32_32.model'))
    model = model.cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    for i in range(num_steps):
        gen = SEPGenerator(space_size=state_space, num_samples=256, time_period=time_period,
                           inverse_density=inverse_density)
        initial_state = gen.data.clone().cpu()
        gen.run()
        final_state = gen.data.clone().cpu()
        optimizer.zero_grad()
        loss = model.NegativeMILoss(initial_state.float(), final_state.float()).mean()
        loss.backward()
        print('Iteration', i, 'I(W,Z) - H(X) > ',
              -loss.detach().cpu().item())
        optimizer.step()
    torch.save(model.state_dict(), '32_32.model')


if __name__ == '__main__':
    experiments = [(8, 4, 8, 0.001)]
    results = []
    for i, params in enumerate(experiments):
        print('Running Experiment', i)
        results.append(run_dim_red_process(*params))
    print(results)
