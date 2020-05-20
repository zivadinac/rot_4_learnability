import torch
import torch.nn.functional as F
import pyro
from pyro.distributions import Poisson, MultivariateNormal
import receptive_fields as rf

class LNP(torch.nn.Module):
    def __init__(self, spatial_shape, timesteps, population_size, receptive_fields, noise_correlations=None):
        """ Create torch module for retinal ganglion population of LNP neurons with optional interneuronal correlation.

            Args:
                spatial_shape - tuple (height x width) of spatial input size
                timesteps - integer number of time steps population responds to
                population_size - number of neurons in population
                receptive_fields - 5d (population_size x 1 x timesteps x *spatial_shape) tensor with population receptive_fields
                noise_correlations - correlation matrix between neurons
        """

        super(LNP, self).__init__()
        self.receptive_fields = torch.nn.Parameter(receptive_fields)
        self.__nonlienarity = torch.exp
        self.__nonlienarity = F.relu
        self.__noise= self.__noiseDistribution(noise_correlations, population_size)

    def forward(self, s):
        """ Apply LNP model on stimulus `s` to obtain spikes.

            Args:
                s - stimuli tensor of shape (batch size x 1 x timesteps x *spatial_shape)

            Return:
                Tensor (batch size x population_size) with number of fired spikes.
        """
        interim = F.conv3d(s, self.receptive_fields).flatten(1)

        if self.__noise is not None:
            noise = self.__noise.sample()
            noise *= interim.mean()
            interim = interim + noise

        rates = self.__nonlienarity(interim)
        return pyro.sample("spikes", Poisson(rates))

    def __noiseDistribution(self, cov, population_size):
        if cov is None:
            return None

        if type(cov) is torch.Tensor:
            if cov.shape[0] != population_size or cov.shape[1] != population_size:
                raise ValueError(f"cov must be of shape ({population_size}, {population_size}).")
            c = cov

        if type(cov) is float:
            c = torch.eye(population_size)
            up_inds = torch.triu_indices(population_size, population_size, 1)
            c[up_inds[0], up_inds[1]] = cov
            low_inds = torch.tril_indices(population_size, population_size, -1)
            c[low_inds[0], low_inds[1]] = cov
        else:
            raise ValueError(f"Unsupported type {type(cov)} for noise_correlations.")

        return MultivariateNormal(torch.zeros(population_size), c)


"""
from stimulus import getVideoStimulusLoader
from receptive_fields import randomRetinalGanglionRFs

vs, vs_props = getVideoStimulusLoader("../data/nat_stim/nat_stim_256_long.mkv", 16, batch_size=1)
vs = iter(vs)
rfs, rf_positions = randomRetinalGanglionRFs(vs_props["spatial_shape"], 16, 8, rf_size=(64, 64))
model = LNP(vs_props["spatial_shape"], 16, 8, rfs)
model_n = LNP(vs_props["spatial_shape"], 16, 8, rfs, 0.5)


for i in range(10):
    print("=================================")
    e = next(vs)
    model(e)
    model_n(e)
    print("=================================")
"""
