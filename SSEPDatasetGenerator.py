import torch
import numpy as np
from matplotlib import pyplot as plt
from scipy import special


class SEPGenerator:
    # space size is the length of the grid
    # number of samples is how many processes we want to simulate in parallel (torch optimized)
    # time period is the number of seconds to run the SEP
    # right probability is probability a particle jumps to the right
    def __init__(self, space_size=250, num_samples=500, time_period=10, right_probability=0.5, inverse_density = 2):
        self.right_probability = right_probability
        self.space_size = space_size  # this is the size of the grid
        self.num_samples = num_samples  # number of games we are simulating at once  (i.e batch size)
        self.time_period = time_period  # number of total jumps by all particles
        self.inverse_density = inverse_density

        # all the data for the dataset (# of bits == samples * space size
        self.data = torch.zeros((self.num_samples, self.space_size), dtype=torch.bool)

        self.random_initializer()
        #self.initializer()  # use a  simple initializers that initializes all the particles on one side

        self.num_particles = torch.sum(self.data, dim=-1, keepdim=True)  # number of particles in each sample

        # use fact that minimum of n mean 1 exponentials is a mean n exponential
        # this is the property behind poisson superposition
        self.exponential = torch.distributions.exponential.Exponential(self.num_particles[0].squeeze().float())
        self.cumulative_time = torch.zeros_like(self.exponential.sample(sample_shape=torch.Size([self.num_samples])))

        # using a bernoulli distribution to generate the jump directions
        self.right_jump = torch.distributions.bernoulli.Bernoulli(probs=right_probability)

    # very basic half block initializer
    # initializer must assign the same amount of particles to each sample
    def initializer(self):
        self.data[:, self.space_size // 2:] = True

    def random_initializer(self):
        x = np.arange(self.space_size)
        rng = np.random.default_rng()
        perms = rng.permuted(np.tile(x, self.num_samples).reshape(self.num_samples, x.size), axis=1)[:, :self.space_size//self.inverse_density]
        self.data[np.arange(self.num_samples).reshape(self.num_samples, 1), perms] = True

    def restart_time(self):
        self.cumulative_time = torch.zeros_like(self.exponential.sample(sample_shape=torch.Size([self.num_samples])))

    def increment_time(self, time):
        self.time_period += time

    def step(self):
        new_wait_times = self.exponential.sample(sample_shape=torch.Size([self.num_samples]))
        self.cumulative_time += new_wait_times
        # mask out the processes that are done updating
        running_samples_indices = torch.nonzero(self.cumulative_time <= self.time_period).squeeze()
        hot_particles = torch.distributions.categorical.Categorical(probs=self.data.float() / self.num_particles)
        initial_indices = hot_particles.sample()

        right_left_jump = (self.right_jump.sample(sample_shape=initial_indices.shape) * 2 - 1).to(torch.int64)

        # we want translational symmetry
        #final_indices = torch.clamp(initial_indices + right_left_jump, min=0, max=self.space_size - 1)
        final_indices = (initial_indices + right_left_jump) % self.space_size

        # update the particles following the SEP rule, making sure particles don't collide
        self.data[running_samples_indices, initial_indices[running_samples_indices]] = self.data[
            running_samples_indices, final_indices[running_samples_indices]]
        self.data[running_samples_indices, final_indices[running_samples_indices]] = 1

    def run(self):
        while (self.cumulative_time < self.time_period).any():
            self.step()

    # compute the fourier transform for the space of particles, this is another representation of tha data
    # it might show deterministic patterns in the low frequency components
    def fft(self):
        frequency_components = torch.fft.rfft(self.data.float(), dim=-1)
        return frequency_components.abs().float(), frequency_components.angle().float()

    def barcode_vizualization(self):
        plt.imshow(self.data.int().numpy() * 255)
        plt.show()

    def density_visualization(self):
        t = self.cumulative_time.mean()
        density = self.data.float().mean(dim=0)
        x = np.arange(self.data.shape[1]) / self.data.shape[1]
        function_fit = 0.5 * (1 + special.erf(self.space_size * (x-0.5)/np.sqrt(t)))
        plt.plot(x, density)
        plt.plot(x, function_fit)
        plt.title("density at t = " + str(t.item()))
        plt.show()

if __name__ == '__main__':
    gen = SEPGenerator(time_period=1600)
    initial_state = None
    final_state = None
    initial_state = gen.data.clone()
    '''
    for i in range(20):
        #gen.barcode_vizualization()
        #gen.density_visualization()
        gen.run()  # run x seconds of SEP
        #gen.restart_time()  # restart internal timer so we can run more seconds
        gen.increment_time(10)
    '''
    gen.run()
    final_state = gen.data.clone()

    initial_state = initial_state.view(gen.num_samples, 250, 1)
    final_state = final_state.view(gen.num_samples, 1, 250)
    big_covariance_matrix = (initial_state.float() @ final_state.float()).mean(dim=0) - initial_state.float().mean(dim=0) @ final_state.float().mean(dim=0)

    import matplotlib.pyplot as plt

    plt.imshow(big_covariance_matrix, cmap='hot', interpolation='nearest')
    plt.title('Heat Map of Cov(SEP(t = 0)_i, SEP(t = tf)_j)')
    plt.xlabel('i')
    plt.ylabel('j')
    plt.show()

    print(big_covariance_matrix.shape)
    offset_covariance = np.zeros(250)
    import math
    for i in range(250):
        for j in range(250):
            offset_covariance[abs(j-i)] += big_covariance_matrix[i, j]/250

    plt.title('Heat Map of Cov(SEP(t = 0)_0, SEP(t = tf)_j)')
    plt.imshow(np.tile(offset_covariance[:100], (50, 1)), cmap='hot', interpolation='nearest')
    plt.xlabel('j')
    plt.show()

    print(offset_covariance)
