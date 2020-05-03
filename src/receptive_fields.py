import torch


def __getNormalizedCos(timesteps):
    ns = torch.cos(torch.linspace(-1, 5, timesteps))
    ns -= ns.min()
    ns /= ns.sum()
    return ns.reshape(timesteps, 1, 1)

def __getNormalizedPowerDecrease(timesteps):
    ns = torch.linspace(0.1, 7, timesteps) ** -0.5
    ns /= ns.sum()
    return ns.reshape(timesteps, 1, 1)
    
def __get2DGaussian(shape, mu=0.0, sigma=1.0):
    limit = 3.5
    x, y = torch.meshgrid(torch.linspace(-limit,limit,shape[0]), torch.linspace(-limit,limit,shape[1]))
    d = torch.sqrt(x*x+y*y)
    return (sigma * torch.sqrt(2. * torch.tensor(3.1415)))**-1 * torch.exp(-((d-mu)**2 / (2.0 * sigma**2)))

def randomRetinalGanglionRFs(spatial_shape, timesteps, population_size, rf_size=None, mu=0.0, sigma_1=1.0, sigma_2=None):
    """ Generate random retinal ganglion receptive fields.
        Spatiall RF is difference of two Gaussians of randomly chosen sign (g1-g2 with probability 0.5 or g2-g1 with probability 0.5).
        Temporarilly it is bimodal function.

        Args:
            spatial_shape - tuple (height x width) of spatial input size
            timesteps - integer number of time steps population responds to
            population_size - number of neurons in population
            rf_size - spatial size of rf, default spatial_shape / 4
            mu - mean of two Gaussians, default 0
            sigma_1 - sigma of first Gaussian, default 1
            sigma_2 - sigma of second Gaussian, default random number from [3 * sigma_1, 3 * sigma_1 + 2]

        Return:
            5D (population_size x 1 x timesteps x *spatial_shape) tensor with population receptive_fields
    """
    def __getRF(spatial_shape, timesteps, mu=0.0, sigma_1=1.0, sigma_2=3.):
        spatial_RF = __get2DGaussian(spatial_shape, mu, sigma_1) - __get2DGaussian(spatial_shape, mu, sigma_2)
        if torch.rand(1) > 0.5:
            spatial_RF *= -1.
        spatial_RF = spatial_RF.expand(1,*spatial_shape)
        spatial_RF /= spatial_RF.sum()
        spatial_RF -= spatial_RF.mean()
        spatial_RF = torch.cat([spatial_RF for t in range(timesteps)], 0)

        temporal_RF = __getNormalizedCos(timesteps)
        RF = spatial_RF * temporal_RF

        #RF[torch.abs(RF) < RF.sum()] = 0.
        return RF

    rf_size = (spatial_shape[0] // 4, spatial_shape[1] // 4) if rf_size is None else rf_size

    positions = torch.randint(spatial_shape[0] * spatial_shape[1], (population_size,))
    positions = [(p // spatial_shape[1], p % spatial_shape[1]) for p in positions]

    rfs = torch.zeros(population_size, 1, timesteps, *spatial_shape)
    for i, p in enumerate(positions):
        h_l = max(p[0] - rf_size[0] // 2, 0)
        h_u = min(p[0] + rf_size[0] // 2, spatial_shape[0])
        w_l = max(p[1] - rf_size[1] // 2, 0)
        w_u = min(p[1] + rf_size[1] // 2, spatial_shape[1])

        #sigma_2 = sigma_1 + 2. * torch.rand(1) if sigma_2 is None else sigma_2
        sigma_2 = 3. * sigma_1 + 2. * torch.rand(1) if sigma_2 is None else sigma_2
        rfs[i, 0, :, h_l:h_u, w_l:w_u] = __getRF(rf_size, timesteps, mu, sigma_1, sigma_2)[:, 0:h_u-h_l, 0:w_u-w_l]
        # maybe randomize which part of rf ends in rfs (last indexing)?

    return rfs

