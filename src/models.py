import torch
import torch.nn.functional as F
import pyro
from pyro.distributions import MultivariateNormal
import receptive_fields as rf
from models_Nonnenmacher import Nonnenmacher_model

def _noiseDistribution(cov, population_size):
    if cov is None:
        return None

    if type(cov) == torch.Tensor:
        if cov.shape[0] != population_size or cov.shape[1] != population_size:
            raise ValueError(f"cov must be of shape ({population_size}, {population_size}).")
        c = cov
    elif type(cov) == float:
        c = torch.eye(population_size)
        up_inds = torch.triu_indices(population_size, population_size, 1)
        c[up_inds[0], up_inds[1]] = cov
        low_inds = torch.tril_indices(population_size, population_size, -1)
        c[low_inds[0], low_inds[1]] = cov
    else:
        raise ValueError(f"Unsupported type {type(cov)} for noise_correlations.")

    return MultivariateNormal(torch.zeros(population_size), c)

class LNProb(torch.nn.Module):
    def __init__(self, timesteps, population_size, receptive_fields, distribution, nonlinearity=F.relu, noise_correlations=None):
        """ Create torch module for retinal ganglion population of LN neurons with probabilistic firing and optional interneuronal correlation.

            Args:
                timesteps - integer number of time steps population responds to
                population_size - number of neurons in population
                receptive_fields - 5d (population_size x 1 x timesteps x *spatial_shape) tensor with population receptive_fields
                distribution - distribution to use for generating spikes (string or type from pyro.distributions)
                nonlinearity - nonlinear function to apply after doing L part (default relu)
                noise_correlations - correlation matrix between neurons, if None no additional noise is added
        """

        super(LNProb, self).__init__()
        self.receptive_fields = torch.nn.Parameter(receptive_fields)
        self.__nonlienarity = nonlinearity
        self.__noise = _noiseDistribution(noise_correlations, population_size)
        self.__distribution = eval(f"pyro.distributions.{distribution}") if type(distribution) == str else distribution

    def forward(self, s):
        """ Apply LNProb model on stimulus `s` to obtain spikes.

            Args:
                s - stimuli tensor of shape (batch size x 1 x timesteps x *spatial_shape)

            Return:
                Tensor (batch size x population_size) with number of fired spikes.
        """
        assert s[0].shape == self.receptive_fields[0].shape
        interim = F.conv3d(s, self.receptive_fields).flatten(1)

        if self.__noise is not None:
            noise = self.__noise.sample()
            noise *= interim.mean()
            interim = interim + noise

        rates = self.__nonlienarity(interim)
        #print(rates)
        return pyro.sample("spikes", self.__distribution(rates))

class LNP(LNProb):
    def __init__(self, timesteps, population_size, receptive_fields, noise_correlations=None):
        """ LNProb with Poisson distribution. """
        super(LNP, self).__init__(timesteps, population_size, receptive_fields, "Poisson", F.relu, noise_correlations)


"""
from stimulus import getVideoStimulusLoader
from receptive_fields import randomRetinalGanglionRFs
from receptive_fields_Nonnenmacher import Nonnenmacher_RFs
import correlation_matrix
import matplotlib.pyplot as plt
import numpy as np

ts = 1
ps = 256
rfs = 64
vs, vs_props = getVideoStimulusLoader("../data/nat_stim/nat_stim_256_long.mkv", ts, batch_size=1)
vs = iter(vs)
rfs, rf_positions = randomRetinalGanglionRFs(vs_props["spatial_shape"], ts, ps, rf_size=(rfs, rfs), off_prob=0.)#, sigma_2=1.5
#model = LNP(ts, ps, rfs)
model_n = LNP(ts, ps, rfs, 0.00005)

frame_num = 100
for i in range(frame_num):
    print(i)
    #print("=================================")
    e = next(vs)
    #res.append(model(e))
    res.append(model_n(e))
    #print("=================================")

res = torch.cat(res, 0).T
nnz = (res > 0).sum()
print(nnz, nnz * 100. / (res.shape[0] * res.shape[1]),"%")

cm = correlation_matrix.computeCorrelationMatrix(res)
corrs = cm[np.triu_indices(ps, k=1)]
print(np.isnan(corrs).sum() / len(corrs), corrs[~np.isnan(corrs)].mean())
plt.hist(corrs, bins=np.minimum(500, ps // 10))
plt.show()
"""
