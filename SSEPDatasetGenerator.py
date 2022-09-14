import torch
from matplotlib import pyplot as plt

class SEPGenerator:
    # space size is the length of the grid
    # number of samples is how many processes we want to simulate in parallel (torch optimized)
    # time period is the number of seconds to run the SEP
    # right probability is probability a particle jumps to the right
    def __init__(self, space_size=250, num_samples=250, time_period=10, right_probability=0.7):
        self.right_probability = right_probability
        self.space_size = space_size  # this is the size of the grid
        self.num_samples = num_samples  # number of games we are simulating at once  (i.e batch size)
        self.time_period = time_period  # number of total jumps by all particles

        # all the data for the dataset (# of bits == samples * space size
        self.data = torch.zeros((self.num_samples, self.space_size), dtype=torch.bool)

        # use fact that minimum of n mean 1 exponentials is a mean n exponential
        # this is the property behind poisson superposition
        self.exponential = torch.distributions.exponential.Exponential(self.space_size)
        self.cumulative_time = torch.zeros_like(self.exponential.sample(sample_shape=torch.Size([self.num_samples])))

        # using a bernoulli distribution to generate the jump directions
        self.right_jump = torch.distributions.bernoulli.Bernoulli(probs=right_probability)

        self.initializer()  # use a  simple initializers that initializes all the particles on one side
        self.num_particles = torch.sum(self.data, dim=-1, keepdim=True)  # number of particles in each sample

    # very basic half block initializer
    def initializer(self):
        self.data[:, :self.space_size // 2] = True

    def restart_time(self):
        self.cumulative_time = torch.zeros_like(self.exponential.sample(sample_shape=torch.Size([self.num_samples])))

    def step(self):
        new_wait_times = self.exponential.sample(sample_shape=torch.Size([self.num_samples]))
        self.cumulative_time += new_wait_times
        # mask out the processes that are done updating
        running_samples_indices = torch.nonzero(self.cumulative_time <= self.time_period).squeeze()
        hot_particles = torch.distributions.categorical.Categorical(probs=self.data.float() / self.num_particles)
        initial_indices = hot_particles.sample()

        right_left_jump = (self.right_jump.sample(sample_shape=initial_indices.shape) * 2 - 1).to(torch.int64)
        final_indices = torch.clamp(initial_indices + right_left_jump, min=0, max=self.space_size - 1)

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

if __name__ == '__main__':
    gen = SEPGenerator()
    for i in range(100):
        gen.barcode_vizualization()
        gen.run()  # run x seconds of SEP
        gen.restart_time()  # restart internal timer so we can run more seconds
