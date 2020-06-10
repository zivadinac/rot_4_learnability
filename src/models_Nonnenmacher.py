import torch
import torch.nn.functional as F
import pyro
from pyro.distributions import MultivariateNormal
from receptive_fields_Nonnenmacher import Nonnenmacher_RFs

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

class LN_signature_paper(torch.nn.Module):
    def __init__(self, timesteps, population_size, receptive_fields, rf_positions):
        """ Create torch module for a population of neurons described in Nonnenmacher paper (fig 2a) (my implementation).

            Args:
                timesteps - integer number of time steps population responds to
                population_size - number of neurons in population
                receptive_fields - 5d (population_size x 1 x timesteps x *spatial_shape) tensor with population receptive_fields
                rf_positions - positions of receptive fields
        """

        assert len(rf_positions) == population_size
        super(LN_signature_paper, self).__init__()
        self.receptive_fields = torch.nn.Parameter(receptive_fields)
        self.__nonlienarity = torch.sigmoid
        self.cov = self.__getNoiseCovarianceMatrix(rf_positions)
        self.cov = 0.9
        self.__noise = _noiseDistribution(self.cov, population_size)


    def forward(self, s):
        """ Apply LN_signature_paper model on stimulus `s` to obtain spikes.

            Args:
                s - stimuli tensor of shape (batch size x 1 x timesteps x *spatial_shape)

            Return:
                Tensor (batch size x population_size) with number of fired spikes.
        """
        offset = 0.168 # offset d from the paper
        interim = F.conv3d(s, self.receptive_fields).flatten(1)
        interim = interim + self.__noise.sample() + offset
        interim[interim > 1.] = 1.
        interim[interim < 0.] = 0.
        #rates = self.__nonlienarity(interim)
        #rates = F.relu(interim)
        rates = interim
        print(rates)
        return rates > 0.5

    def __getNoiseCovarianceMatrix(self, rf_positions, base_cov=0.022, a=0.45, tau=15):
        """ Get covariance matrix described in signature paper.

            Args:
                rf_positions - positions of receptive fields
                base_cov - sigma_noise from the paper
                a - a from the paper
                tau - tau from the paper, but in pixel space
        """

        b = torch.sqrt(torch.tensor(1. - a**2))
        rf_distances = self.__getRFDistancesMatrix(rf_positions)

        return base_cov**2 * (a * torch.eye(len(rf_positions)) + b * torch.exp(-rf_distances / tau))

    def __getRFDistancesMatrix(self, rf_positions):
        ps = len(rf_positions)
        rf_distances = torch.zeros(ps, ps)
        for i in range(ps):
            for j in range(i ,ps):
                rf_distances[i,j] = rf_distances[j,i] = torch.norm(rf_positions[i].to(torch.float32)-rf_positions[j].to(torch.float32))
        return rf_distances

class Nonnenmacher_model(torch.nn.Module):
    # Nonnenmacher RGC model
    # Adapted from: https://github.com/mackelab/critical_retina/blob/master/data/data_generation/Retina_model.ipynb

    def __init__(self, receptive_fields):
        """ Create torch module for a population of neurons described in Nonnenmacher_model paper (fig 2a) (original implementation).

            Args:
                receptive_fields - tuple of cells, mask_small, f_dog_small, f_dog_large (returned from receptive_fields_Nonnenmacher.Nonnenmacher_RFs())
        """
        super(Nonnenmacher_model, self).__init__()
        self.cells = receptive_fields[0]
        self.mask_small = receptive_fields[1]
        self.f_dog_small = receptive_fields[2]
        self.f_dog_large = receptive_fields[3]

    def forward(self, s):
        """ Apply LN_signature_paper model on stimulus `s` to obtain spikes.

            Args:
                s - stimuli tensor of shape (1 x 1 x 1 x *spatial_shape)

            Return:
                Tensor (batch size x population_size) with number of fired spikes.
        """
        import numpy as np
        from scipy.signal import fftconvolve
        def generate_output(rf,mask_small,I,f_dog_small,f_dog_large,Sigma,trials,offset):
            num_stim, N_x, N_y = I.shape
            s_small=np.empty((num_stim,N_x,N_y))
            s_large=np.empty((num_stim,N_x,N_y))
            for i in range(num_stim):
                s_small[i,:,:]=fftconvolve(I[i,:,:],f_dog_small[133:267,133:267],mode='same')
                s_large[i,:,:]=fftconvolve(I[i,:,:],f_dog_large[75:325,75:325],mode='same')
            rf=rf.astype(int)
            num_cells=rf.shape[1]
            b=np.zeros((num_stim,trials,num_cells))
            p=np.zeros((num_stim,trials,num_cells))
            p_im=np.zeros((num_stim,trials,num_cells))
            p_rnd=np.zeros((num_stim,trials,num_cells))
            n=np.random.multivariate_normal(np.zeros(num_cells),Sigma,(num_stim,trials))
            for j in range(num_cells):
                if mask_small[j]:
                    b[:,:,j]=np.rint(np.clip(np.outer(s_small[:,rf[0,j]+N_x//2,rf[1,j]+N_y//2],np.ones(trials))\
                                             +n[:,:,j]+offset,0,1))
                else:
                    b[:,:,j]=np.rint(np.clip(np.outer(s_large[:,rf[0,j]+N_x//2,rf[1,j]+N_y//2],np.ones(trials))\
                                             +n[:,:,j]+offset,0,1))
            return b

        px_length=2 #pixel size in µm
        noise_std=0.022 #standard deviation of white noise
        alpha=0.45 #independent noise fraction
        beta=np.sqrt(1-alpha**2) #correlated noise fraction
        tau=30.0  #spatial decay of noise correlations
        offset=0.168
        trials=1
        num_cells = self.cells.shape[1]

        Delta=np.array([[np.linalg.norm(self.cells[:,i]*px_length-self.cells[:,j]*px_length)\
                 for i in range(num_cells)]for j in range(num_cells)])
        Sigma=noise_std**2 *(beta*np.exp(-Delta/(2*tau))+alpha*np.identity(num_cells))

        # conversions between torch and np tensors should be fast and without copying
        stimulus = s[0,0].numpy()
        output = generate_output(self.cells, self.mask_small, stimulus, self.f_dog_small, self.f_dog_large, Sigma, trials, offset)[0]
        #print(output.shape)
        #print(s.shape)
        #print(stimulus.shape)
        return torch.from_numpy(output)


"""
from stimulus import getVideoStimulusLoader
from receptive_fields import randomRetinalGanglionRFs
import correlation_matrix
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.stats import boxcox
from skimage import exposure

def histogram_equalize(img):
    img_cdf, bin_centers = exposure.cumulative_distribution(img)
    return np.interp(img, bin_centers, img_cdf)

def myBoxCox(e):
    sh = e.shape
    e = histogram_equalize(e[0,0,0].numpy())
    e -= e.min()
    e /= e.max()
    e += 1e-8
    e = torch.from_numpy(boxcox(e.flatten())[0]).reshape(sh)
    e -= e.mean()
    e += 0.5
    return e

res = []
if False:
    rfs, rf_positions = Nonnenmacher_RFs((400,400))
    ps = rfs[0].shape[1]
    model_n = Nonnenmacher_model(rfs)

    textures_400 = loadmat("../../data/critical_retina/textures_400.mat")["textures"]
    textures=[]
    for i in range(textures_400.shape[1]):
        textures.append(textures_400[0,i][0])
    textures=np.array(textures)/255.
    textures=np.delete(textures,64,0)# excluding two textures with outlying mean
    textures=np.delete(textures,38,0)
    num_text=textures.shape[0]

    frame_num = len(textures)
    for i in range(frame_num):
        print(i)
        e = torch.from_numpy(textures[i]).reshape(1, 1, 1, *textures.shape[1:])
        res.append(model_n(e))
else:
    ts = 1
    ps = 256
    rfs = 64
    vs, vs_props = getVideoStimulusLoader("../data/nat_stim/nat_stim_256_long.mkv", ts, batch_size=1)
    vs = iter(vs)

    rfs, rf_positions = Nonnenmacher_RFs(vs_props["spatial_shape"])
    ps = rfs[0].shape[1]
    model_n = Nonnenmacher_model(rfs)

frame_num = 100
for i in range(frame_num):
    print(i)
    #print("=================================")
    # e = torch.from_numpy(np.clip(np.random.randn(1,1,1,256,256) + 0.5, 0., 1.))
    e = next(vs)
    e = myBoxCox(e)
    #e = torch.log(e)
    #e = (e - e.mean()) / e.std()
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
