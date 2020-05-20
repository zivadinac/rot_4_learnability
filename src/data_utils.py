import pickle
from os import path
import numpy as np
import scipy.io
from utils import ObjView
import torch

def spikeRasterToSpikeTimes(spikeRaster, binsize):
    """
        From a spike raster create a neurons list of lists of spike times

        Args:
            spikeRaster - spike raster
            binsize - size of a bin
        Return:
            List of neuron spiking times.

        This function was originally written by Aditya:
        https://github.com/adityagilra/UnsupervisedLearningNeuralData/blob/master/EMBasins_sbatch.py#L73
    """
    nNeurons,tSteps = spikeRaster.shape
    nrnSpikeTimes = []
    # multiply by binsize, so that spike times are given in units of sampling indices
    bins = np.arange(tSteps,dtype=float)*binsize
    for nrnnum in range(nNeurons):
        # am passing a list of lists, convert numpy.ndarray to list,
        #  numpy.ndarray is just used to enable multi-indexing
        nrnSpikeTimes.append( list(bins[spikeRaster[nrnnum,:] != 0]) )
    return nrnSpikeTimes

def spikeTimesToSpikeRaster(nrnSpikeTimes, binsteps):
    """
        Convert neuron spike times to spike raster.
        Args:
            nrnSpikeTimes - spike times, 2d array [neurons x spike times]
            binsteps - 
        Return:
            2D array (neurons x bin_num) with spike indicators in each bin
    
        This function was originally written by Aditya:
        https://github.com/adityagilra/UnsupervisedLearningNeuralData/blob/master/EMBasins_sbatch.py#L85
    """
    maxBins = 1
    nNeurons = len(nrnSpikeTimes)
    # loop since not a numpy array, each neuron has different number of spike times
    for nrnnum in range(nNeurons):
        maxBins = max( ( maxBins, int(np.amax(nrnSpikeTimes[nrnnum])/binsteps) + 1 ) )
    spikeRaster = np.zeros((nNeurons,maxBins))
    for nrnnum in range(nNeurons):
        spikeRaster[ nrnnum, (np.array(nrnSpikeTimes[nrnnum]) / binsteps).astype(int) ] = 1.
    return spikeRaster

def loadPrenticeEtAl2016(data_path, shuffle=True, seed=100, as_object=True):
    """ Load Prentice et al 2016 dataset.
        Args:
            data_path - dataset path
            shuffle - wheter to shuffle or not time bins
            seed - seed for random shuffling, used only if shuffle=True
            as_object - return object containing data
        Return:
            2D array (neurons x bin_num) with spike indicators in each bin

        This function is adapted from loadDataSet function in original Aditya's code:
        https://github.com/adityagilra/UnsupervisedLearningNeuralData/blob/master/EMBasins_sbatch.py#L96
    """
    retinaData = scipy.io.loadmat(data_path)
    # see: https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    #  "For historic reasons, in Matlab everything is at least a 2D array, even scalars.
    #   So scipy.io.loadmat mimics Matlab behavior by default."
    # retinaData['data'][0,0] has .__class__ numpy.void, and has 'keys' (error on .keys() !):
    # description, experiment_date, spike_times, stimulus, hmm_fit
    #  the latter three are matlab structs, so similiar [0,0] indexing.
    #  load into matlab to figure out the 'keys'!
    nrnSpikeTimes = retinaData['data'][0,0]['spike_times'][0,0]['all_spike_times'][0]

    ## obsolete: when I was not converting to spikeRaster (needed for shuffling),
    ##  I int-divided by 200 and converted to list of lists
    ## spikeTimes are in bins of 1/10,000Hz i.e. 0.1 ms
    ## we bin it into 20 ms bins, so integer divide spikeTimes by 20/0.1 = 200 to get bin indices
    #nNeurons = len(nrnSpikeTimes)
    #nrnSpikeTimes = nrnSpikeTimes // 200
    #nrnspiketimes = []
    #tSteps = 0
    #for nrnnum in range(nNeurons):
    #    # am passing a list of lists via boost, convert numpy.ndarray to list
    #    spikeTimes = nrnSpikeTimes[nrnnum][0]
    #    tSteps = np.max( (tSteps,spikeTimes[-1]) )
    #    # somehow np.int32 gave error in converting to double in C++ via boost
    #    nrnspiketimes.append( list(spikeTimes.astype(np.float)) )

    # to shuffle time bins for this dataset, I need to convert spike times to spike raster
    spikeRaster = spikeTimesToSpikeRaster(nrnSpikeTimes, 200)    # bin size = 20ms, i.e. 200 steps @ 10kHz sampling
    nNeurons,tSteps = spikeRaster.shape
    if shuffle:
        # randomly permute the full dataset
        # careful if fitting a temporal model and/or retina has adaptation
        # set seed to ensure repeatability of train/test split later
        np.random.seed(seed)
        shuffled_idxs = np.random.permutation(np.arange(tSteps,dtype=np.int32))
        spikeRaster = spikeRaster[:,shuffled_idxs]        

    d = {"data": spikeRaster}
    return ObjView(d) if as_object else d

def loadSimulatedData(data_path, clip_bins=True, as_object=True):
    """
        Load data generated in a simulation.

        Args:
            data_path - path to data file
            clip_bins - limit number of spikes to 1 per bin (e.g. get binary data)
            as_object - return object containing data

        Return:
            Data saved in `data_path` location
    """
    with open(data_path, "rb") as data_file:
        data = pickle.load(data_file)

    if clip_bins:
        data["data"] = np.clip(data["data"], 0, 1)

    if as_object:
        data = ObjView(data)

    return data

def loadSpikeData(data_path):
    _, ext = path.splitext(data_path) # prentice et al data comes as .mat file, TODO maybe I should resave it as pickle?
    return loadPrenticeEtAl2016(data_path, shuffle=False) if ext == ".mat" else loadSimulatedData(data_path)

def saveSimulatedData(data_path, data):
    with open(data_path, "wb") as data_file:
        pickle.dump(data, data_file, protocol=4)

def save(data_path, data):
    with open(data_path, "wb") as data_file:
        pickle.dump(data, data_file, protocol=4)

def load(data_path, as_object=True):
    with open(data_path, "rb") as data_file:
        data = pickle.load(data_file)

    if as_object:
        data = ObjView(data)

    return data

def excludeNonFiringNeurons(spikes):
    retained_inds = []
    excluded_inds = []
    for i in range(spikes.shape[0]):
        if not np.all(spikes[i] == spikes[i][0]):
            retained_inds.append(i)
        else:
            excluded_inds.append(i)
    #print(f"Excluded neurons: {excluded_inds} ({len(excluded_inds)}, {100*len(excluded_inds)/spikes.shape[0]}%)")
    return spikes[retained_inds, :], retained_inds, excluded_inds

