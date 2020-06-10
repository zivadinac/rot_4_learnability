# Receptive fields for Nonnenmacher RGC model
# Adapted from: https://github.com/mackelab/critical_retina/blob/master/data/data_generation/Retina_model.ipynb

import numpy as np
from scipy.io import loadmat

def Nonnenmacher_RFs(spatial_shape, population_size=None, retinainfo_path="retinainfo.mat"):
    """ Generate retinal ganglion receptive fields as described in Nonnenmacher paper.

        Args:
            spatial_shape - tuple (height x width) of spatial input size
            population_size - number of neurons in population, if None positions are read from retinainfo file

        Return:
            cell, mask_small, f_dog_small, f_dog_large, rf_positions
    """
    def dog_filter(sig_cen,ratio,weight,on,N_x,N_y):
        sig_sur=sig_cen*ratio
        dog=lambda x,y: on/(2*np.pi*sig_cen**2)*np.exp(-(x**2+y**2)/(2*sig_cen**2))\
        -on*weight/(2*np.pi*sig_sur**2)*np.exp(-(x**2+y**2)/(2*sig_sur**2))

        arr=[]
        for i in range (N_y):
            new=[]
            for j in range (N_x):
                new.append(dog(-0.5*N_x+(j+0.5),-0.5*N_y+(i+0.5)))
            arr.append(new)
        arr=np.array(arr)
        return arr

    px_length=2 #pixel size in µm
    #DoG filter parameters
    center_surround_ratio_small=2.0
    center_surround_ratio_large=2.0
    weight_surround=0.5
    on=1 #cell type (1:on, -1:off)

    retinainfo=loadmat(retinainfo_path)
    #getting RF positions, centered around 0
    if population_size is None:
        rf_positions=np.array([retinainfo['pos_x'].flatten()-50,retinainfo['pos_y'].flatten()])
    sigma_small=retinainfo['rf_size_small'].item()/2 #Factor of 0.5 since given values seem to be overly large
    sigma_large=retinainfo['rf_size_large'].item()/2 #compared to the overall area


    #randomly masking and removing 40% of the cells to correct for amacrine cells in the sample
    choice=np.array([np.random.random(rf_positions.shape[1])>0.6])
    choice=np.concatenate((choice,choice))
    cells=np.ma.compress_cols(np.ma.masked_array(rf_positions,choice))
    #masking two thirds of the cells that should have small receptive fields
    mask_small=np.array([np.random.random(cells.shape[1])>0.67]).flatten()

    sigma_small=sigma_small/px_length
    sigma_large=sigma_large/px_length
    cells=cells/px_length

    f_dog_small=dog_filter(sigma_small,center_surround_ratio_small,weight_surround,on,*spatial_shape)
    f_dog_large=dog_filter(sigma_large,center_surround_ratio_large,weight_surround,on,*spatial_shape)

    return (cells, mask_small, f_dog_small, f_dog_large), rf_positions

