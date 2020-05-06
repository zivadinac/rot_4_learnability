from argparse import ArgumentParser
import numpy as np
import sys
from os import path
import pickle
import data_utils
import EMBasins
import utils

# This script is based on original script writen by Aditya:
# https://github.com/adityagilra/UnsupervisedLearningNeuralData/blob/master/EMBasins_sbatch.py
# TODO naming convention - change to snake_case as in other files

args = ArgumentParser()
args.add_argument("dataPath")
args.add_argument("outPath")
args.add_argument("--crossValFold", default=2, type=int, help="k-fold validation, 1 for train only")
args.add_argument("--nModes", default=70, type=int, help="Best nModes reported for experimental data in Prentice et al 2016")
args.add_argument("--nIter", default=100, type=int, help="HMM training iterations.")
args.add_argument("--seed", type=int, default=12345)
#args.add_argument("--binSize", default=1)
#args.add_argument("--shuffle", default=0, type=bool, help="Don't shuffle time bins for Prentice et al data,\
#                                                but shuffle for Marre et al data and generated datasets")
#args.add_argument("--treeSpatialCorr", default=1, help="Tree-based spatial correlations or no correlations")
#args.add_argument("--maxModes", default=150, help="Max number of allowed modes")
args = args.parse_args()

np.random.seed(args.seed)
binSize = 1 # here, spikes are given pre-binned into a spikeRaster, just take binSize=1

spikeRaster = data_utils.loadSimulatedData(args.dataPath)
nrnSpikeTimes = data_utils.spikeRasterToSpikeTimes(spikeRaster, binSize)
nNeurons, tSteps = spikeRaster.shape

print("Mixture model fitting for interaction factor = 1., nModes = ", args.nModes)
sys.stdout.flush()
    
trainLogLi = np.zeros(shape=(args.crossValFold, args.nIter))
testLogLi = np.zeros(shape=(args.crossValFold, args.nIter))

# TODO use same naming convention everywhere
if args.crossValFold > 1:
    # translated from getHMMParams.m 
    # if I understand correctly:
    #  to avoid losing temporal correlations,
    #  we specify contiguous chunks of training data
    #  by specifying upper _hi and lower _lo boundaries, as below
    #  the rest becomes contiguous chunks of test data
    #  thus, HMM::logli(true) in EMBasins.cpp gives training logli
    #   and HMM::logli(false) gives test logli
    bins = np.arange(tSteps) * binSize
    shuffledIdxs = np.random.permutation(np.arange(tSteps, dtype=np.int32))
    nTest = int(tSteps/args.crossValFold)
    for k in range(args.crossValFold):
        testIdxs = shuffledIdxs[k*nTest:(k+1)*nTest]
        trainIdxs = np.zeros(tSteps,dtype=np.int32)
        trainIdxs[testIdxs] = 1
        
        # contiguous 1s form a test chunk, i.e. are "unobserved"
        #  see state_list assignment in the HMM constructor in EMBasins.cpp
        flips = np.diff(np.append([0],trainIdxs))
        unobserved_lo = bins[ flips == 1 ]
        unobserved_hi = bins[ flips == -1 ]
        # just in case, a last -1 is not there to close the last chunk
        if (len(unobserved_hi) < len(unobserved_lo)):
            unobserved_hi = np.append(unobserved_hi,[tSteps])

        unobserved_lo = unobserved_lo.astype(np.float64)
        unobserved_hi = unobserved_hi.astype(np.float64)

        EMBasins.pyInit()
        params,trans,P,emiss_prob,alpha,pred_prob,hist,samples,stationary_prob,trainLogLi_this,testLogLi_this = \
            EMBasins.pyHMM(nrnSpikeTimes, unobserved_lo, unobserved_hi, float(binSize), args.nModes, args.nIter)

        print(f"Finished cross validation round {k} of fitting.\
                \nTrain logL = {trainLogLi_this[0][0]}\
                \nTest logL = {testLogLi_this[0][0]}")

        trainLogLi[k,:] = trainLogLi_this.flatten()
        testLogLi[k,:] = testLogLi_this.flatten()
else: # no cross-validation specified, train on full data
    # hmmmm this should be same as crossValFold = 1
    params,trans,P,emiss_prob,alpha,pred_prob,hist,samples,stationary_prob,trainLogLi_this,testLogLi_this = \
        EMBasins.pyHMM(nrnSpikeTimes, np.ndarray([]), np.ndarray([]), float(binSize), args.nModes, args.nIter)
    trainLogLi[0,:] = trainLogLi_this.flatten()
    testLogLi[0,:] = testLogLi_this.flatten()

print(f"Finished fitting mixture model for \
        \n\t interaction factor = 1.\
        \n\t nModes = {args.nModes}\
        \n\t logL = {trainLogLi}\
        \n\t test logL = {testLogLi}")

fitPath = path.join(args.outPath, path.basename(args.dataPath).split('.')[0] + f"_nModes_{args.nModes}_nIter_{args.nIter}.pck")
utils.saveFit(fitPath, args.crossValFold, False, args.nModes, params, trans, P, emiss_prob, alpha, pred_prob, hist, samples, stationary_prob, trainLogLi, testLogLi)
print(f"Fitted model saved to {fitPath}.")
sys.stdout.flush()

