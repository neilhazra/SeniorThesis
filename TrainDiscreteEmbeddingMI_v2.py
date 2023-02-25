import torch
import torch.nn as nn
import DiscreteVariationalParameterizations as DVP
from SSEPDatasetGenerator import SEPGenerator
from GibbsSampling import BatchedConditionalGibbsSampler
from torch.autograd.functional import vjp
from torch.autograd.function import Function

class EmbeddingMI2(nn.Module):
    def __init__(self, batch_size, in_dim, out_dim):
        super().__init__()
        self.encoder = DVP.EnergyBasedEncoderDecoder(batch_size=batch_size, in_dim=in_dim, out_dim=out_dim)
        self.decoder = DVP.EnergyBasedEncoderDecoder(batch_size=batch_size, in_dim=out_dim, out_dim=in_dim)
        self.embedding_dynamics = DVP.EnergyBasedModelEmbeddingDynamics(dim=out_dim)

        self.embedding_sampler = BatchedConditionalGibbsSampler(batch_size=batch_size, num_samples=256, # needs to be tuned
                                                                      mixing_time=4,    # seems like this can be low and still work
                                                                      joint_distribution=self.embedding_dynamics)
        
        self.encoder_sampler = BatchedConditionalGibbsSampler(batch_size=batch_size, num_samples=256, # needs to be tuned
                                                                      mixing_time=4,    # seems like this can be low and still work
                                                                      joint_distribution=self.encoder)
        
        self.decoder_sampler = BatchedConditionalGibbsSampler(batch_size=batch_size, num_samples=256, # needs to be tuned
                                                                      mixing_time=4,    # seems like this can be low and still work
                                                                      joint_distribution=self.decoder)
        


    def full_objective_function(self, x, y):
        w = self.encoder_sampler.run_batched_gibbs(x).detach()
        z = self.encoder_sampler.run_batched_gibbs(y).detach()
        p_x_w = self.decoder.conditional_log_probability_a_given_b_(x, w)        
        p_w_x = self.encoder.conditional_log_probability_a_given_b_(w, x)
        r_w_z = DVP.EnergyBasedModelEmbeddingDynamics.normalized_log_probabilities_w_given_z(z, w, self.embedding_dynamics.linear_1_weight, self.embedding_dynamics.linear_1_bias, self.embedding_dynamics.linear_2_weight, self.embedding_dynamics.linear_2_bias)
        return -(p_x_w - p_w_x + r_w_z)


class EmbeddingMI1(nn.Module):
    def __init__(self, batch_size, in_dim, out_dim):
        super().__init__()
        self.encoder_decoder = DVP.BoltzmannEncoderDecoder(batch_size=batch_size, in_dim=in_dim, out_dim=out_dim)
        self.embedding_dynamics = DVP.EnergyBasedModelEmbeddingDynamics(dim=out_dim)
        self.sampler = BatchedConditionalGibbsSampler(batch_size=batch_size, num_samples=256, # needs to be tuned
                                                                      mixing_time=4,    # seems like this can be low and still work
                                                                      joint_distribution=self.embedding_dynamics)
        self.loss_func = CustomMutualInformationLoss.apply


    def full_objective_function(self, x, y):
        w = self.encoder_decoder.encoder_sample_(x).detach()
        z = self.encoder_decoder.encoder_sample_(y).detach()
        p_x_w = DVP.BoltzmannEncoderDecoder.conditional_log_probability_x_given_w(w, x, self.encoder_decoder.W, self.encoder_decoder.c)
        p_w_x =  DVP.BoltzmannEncoderDecoder.conditional_log_probability_w_given_x(w, x, self.encoder_decoder.W, self.encoder_decoder.b)
        r_w_z = DVP.EnergyBasedModelEmbeddingDynamics.normalized_log_probabilities_w_given_z(z, w, self.embedding_dynamics.linear_1_weight, self.embedding_dynamics.linear_1_bias, self.embedding_dynamics.linear_2_weight, self.embedding_dynamics.linear_2_bias)
        return -(p_x_w - p_w_x + r_w_z)
    
    def quasi_objective_demo(self, x, y):
        w = self.encoder_decoder.encoder_sample_(x).detach()
        z = self.encoder_decoder.encoder_sample_(y).detach()
        w_tilde = self.sampler.run_batched_gibbs(z).detach()

        return -self.loss_func(
            z,y,w,x,w_tilde, self.encoder_decoder.W, self.encoder_decoder.b, self.encoder_decoder.c, 
            self.embedding_dynamics.linear_1_weight, self.embedding_dynamics.linear_1_bias, self.embedding_dynamics.linear_2_weight, self.embedding_dynamics.linear_2_bias)

# we got to do this the old fashioned way because the random sampling does not propagate the 
# expected gradient
class CustomMutualInformationLoss(Function):
    # Note that forward, setup_context, and backward are @staticmethods
    def forward(ctx, z, y, w, x, w_tilde, W, b, c, W1, b1, W2, b2):
        p_x_w = DVP.BoltzmannEncoderDecoder.conditional_log_probability_x_given_w(w, x, W, c)
        p_w_x =  DVP.BoltzmannEncoderDecoder.conditional_log_probability_w_given_x(w, x, W, b)
        
        # oom if embedding size is too large
        #r_w_z = DVP.EnergyBasedModelEmbeddingDynamics.normalized_log_probabilities_w_given_z(z, w, W1, b1, W2, b2)

        # for some reason you need to alternate training between these two modes
        # initially the uniform importance sampling is better, later the boltzmann machine sampling is better
        #temp1 =  DVP.EnergyBasedModelEmbeddingDynamics.estimated_normalized_log_probabilities_w_given_z_better(z, w, x, W, b, c, W1, b1, W2, b2)
        #temp2 =  DVP.EnergyBasedModelEmbeddingDynamics.estimated_normalized_log_probabilities_w_given_z_better(z, w, y, W, b, c, W1, b1, W2, b2)
        temp3 = DVP.EnergyBasedModelEmbeddingDynamics.normalized_log_probabilities_w_given_z(z, w, W1, b1, W2, b2)
        #print(temp3.mean().detach().cpu().item(), temp1.mean().detach().cpu().item(), temp2.mean().detach().cpu().item())
        r_w_z = torch.minimum(temp3 ,torch.zeros_like(temp3)) # importance sampling underestimtaes the partition function so we must clamp at 0
        
        ctx.save_for_backward(z,y,w, w_tilde, x,W,b,c,W1, b1, W2, b2, p_x_w - p_w_x, r_w_z)

        # ctx.save_for_backward(z,y,w, w_tilde, x,W,b,c,W1, b1, W2, b2, p_x_w - p_w_x, torch.tensor([-1.0], device=z.device)) 
        # because r is hard to estimate maybe the gradient will roughly be in the right direction this way anyway
        return p_x_w - p_w_x + r_w_z

    @staticmethod
    def backward(ctx, grad_output):
        z,y,w,w_tilde, x, W,b,c, W1, b1, W2, b2, encoder_decoder, embedding = ctx.saved_tensors
        _, (_,_, temp_W_1, temp_c) = vjp(DVP.BoltzmannEncoderDecoder.conditional_log_probability_x_given_w, (w,x,W,c), grad_output, create_graph=False)
        _, (_, _, temp_W_2, temp_b_1) = vjp(DVP.BoltzmannEncoderDecoder.conditional_log_probability_w_given_x, (w,x,W,b), grad_output*(embedding + encoder_decoder - 1), create_graph=False)
        _, (_, _, temp_W_3, temp_b_2) = vjp(DVP.BoltzmannEncoderDecoder.conditional_log_probability_w_given_x, (z,y,W,b), grad_output*embedding, create_graph=False)
        _, (_, _, temp_linear_W_1, temp_linear_b_1, temp_linear_W_2, temp_linear_b_2) = vjp(DVP.EnergyBasedModelEmbeddingDynamics.unnormalized_log_probs_w_given_z, (z,w, W1, b1, W2, b2), grad_output, create_graph=False)
        _, (_, _, temp_1_linear_W_1, temp_1_linear_b_1, temp_1_linear_W_2, temp_1_linear_b_2) = vjp(DVP.EnergyBasedModelEmbeddingDynamics.expected_unnormalized_log_probs_w_given_z, (z.expand(w_tilde.shape[0], -1, -1),w_tilde, W1, b1, W2, b2), grad_output, create_graph=False)        
        return None, None, None, None, None, temp_W_1 + temp_W_2 + temp_W_3, temp_b_1 + temp_b_2, temp_c,temp_linear_W_1 - temp_1_linear_W_1, temp_linear_b_1 - temp_1_linear_b_1, temp_linear_W_2 - temp_1_linear_W_2 , temp_linear_b_2 - temp_1_linear_b_2


def run_dim_red_process(state_space, inverse_density, embedding_space_size, time_period, num_steps = 3*15000):
    model = EmbeddingMI1(512, state_space, embedding_space_size)
    model.load_state_dict(torch.load(f'experiments/experiement_{state_space}_{embedding_space_size}_{0.5}_{24999}.model'))
    
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for i in range(num_steps):
        gen = SEPGenerator(space_size=state_space, num_samples=512, time_period=time_period,
                           inverse_density=inverse_density)
        initial_state = gen.data.clone().cuda()
        gen.run()
        final_state = gen.data.clone().cuda()
        optimizer.zero_grad()
        #loss = model.quasi_objective_demo(initial_state.float(), final_state.float()).mean()
        loss = model.full_objective_function(initial_state.float(), final_state.float()).mean()
        loss.backward()
        print('Iteration', i, 'I(W,Z) > ',
                  -loss.detach().cpu().item())
        optimizer.step()

        if i % 5000 == 4999:
            torch.save(model.state_dict(), f'experiments/experiement_{state_space}_{embedding_space_size}_{time_period}_{i}.model')


if __name__ == '__main__':
    #experiments = [(16, 4, 16, 1), (16, 4, 16, 2), (16, 4, 16, 4), (16, 4, 16, 8), (16, 4, 16, 16),
    #                (16, 4, 8, 1), (16, 4, 8, 2), (16, 4, 8, 4), (16, 4, 8, 8), (16, 4, 8, 16),
    #                (16, 4, 4, 1), (16, 4, 4, 2), (16, 4, 4, 4), (16, 4, 4, 8), (16, 4, 4, 16),
    #                (16, 4, 2, 1), (16, 4, 2, 2), (16, 4, 2, 4), (16, 4, 2, 8), (16, 4, 2, 16)]
    
    
    #for i, params in enumerate(experiments):
    #    print('Running Experiment', i, params)
    #    run_dim_red_process(*params)
    run_dim_red_process(*(8,4,6,1))
