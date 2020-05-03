import torch
import torch.nn.functional as F
import pyro
from pyro.distributions import Poisson
import receptive_fields as rf

class LNP(torch.nn.Module):
    def __init__(self, spatial_shape, timesteps, population_size, receptive_fields):
        """ Create torch module for retinal ganglion population of LNP neurons.

            Args:
                spatial_shape - tuple (height x width) of spatial input size
                timesteps - integer number of time steps population responds to
                population_size - number of neurons in population
                receptive_fields - 5d (population_size x 1 x timesteps x *spatial_shape) tensor with population receptive_fields
        """

        self.receptive_fields = receptive_fields
        self.__nonlienarity = torch.exp
        self.__nonlienarity = F.relu
        #self.__nonlienarity = lambda x: x**2

    def forward(self, s):
        """ Apply LNP model on stimulus `s` to obtain spikes.

            Args:
                s - stimuli tensor of shape (batch size x 1 x timesteps x *spatial_shape)

            Return:
                Tensor (batch size x population_size) with number of fired spikes.
        """
        rates = self.__nonlienarity(F.conv3d(s, self.receptive_fields).flatten(1))
        return pyro.sample("spikes", Poisson(rates))

