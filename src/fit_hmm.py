from argparse import ArgumentParser
import numpy as np
import shelve, sys, os.path
import data_utils
import EMBasins
#import TreeHMM

def saveFit(dataDir, crossValFold, shuffle, nModes, \
            params, trans, P, emiss_prob, alpha, pred_prob, \
            hist, samples, stationary_prob, \
            trainLogLi, testLogLi):
    dataBase = shelve.open(dataDir + ('data_shuffled' if shuffle else 'data') \
                                        + '_HMM'+(str(crossValFold) if crossValFold>1 else '') \
                                        #+ ('' if treeSpatialCorr else '_notree') \
                                        +'_modes'+str(nModes)+'.shelve')
    dataBase['params'] = params
    dataBase['trans'] = trans
    dataBase['P'] = P
    dataBase['emiss_prob'] = emiss_prob
    dataBase['alpha'] = alpha
    dataBase['pred_prob'] = pred_prob
    dataBase['hist'] = hist
    dataBase['samples'] = samples
    dataBase['stationary_prob'] = stationary_prob
    dataBase['train_logli'] = trainLogLi
    dataBase['test_logli'] = testLogLi
    dataBase.close()



args = ArgumentParser()
args.add_argument("--dataDir", default="../data/Prenticeetal2016_data/unique_natural_movie/")
args.add_argument("--shuffle", default=0, help="Don't shuffle time bins for Prentice et al data,\
                                                but shuffle for Marre et al data and generated datasets")
args.add_argument("--crossValFold", default=2, help="k-fold validation, 1 for train only")
#args.add_argument("--treeSpatialCorr", default=1, help="Tree-based spatial correlations or no correlations")
#args.add_argument("--maxModes", default=150, help="Max number of allowed modes")
args.add_argument("--nModes", default=70, help="Best nModes reported for experimental data in Prentice et al 2016")
#args.add_argument("--binSize", default=1)
args.add_argument("--nIter", default=100, help="HMM training iterations.")
args = args.parse_args()

dataDir = args.dataDir
shuffle = args.shuffle
crossValFold = args.crossValFold
nModes = args.nModes
binSize = 1 # here, spikes are given pre-binned into a spikeRaster, just take binSize=1
nIter = args.nIter
#nIter = 1
# TODO write "SmartArgs wrapper, it will be very useful in the future"

np.random.seed(100)

spikeRaster = data_utils.loadPrenticeEtAl2016(dataDir, shuffle=shuffle)
nNeurons, tSteps = spikeRaster.shape
spikeRaster = spikeRaster[:,:tSteps] # HMM will split the dataset in train and test sets based on crossValFold
nrnSpikeTimes = data_utils.spikeRasterToSpikeTimes(spikeRaster, binSize)

print("Mixture model fitting for interaction factor = 1., nModes = ", nModes)
sys.stdout.flush()
    
trainLogLi = np.zeros(shape=(crossValFold, nIter))
testLogLi = np.zeros(shape=(crossValFold, nIter))

# TODO use same naming convention everywhere
if crossValFold > 1:
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
    nTest = int(tSteps/crossValFold)
    for k in range(crossValFold):
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
            EMBasins.pyHMM(nrnSpikeTimes, unobserved_lo, unobserved_hi, float(binSize), nModes, nIter)
        #EMBasins.pyInit()
        print(f"Finished cross validation round {k} of fitting.\
                \nTrain logL = {trainLogLi_this[0][0]}\
                \nTest logL = {testLogLi_this[0][0]}")
        trainLogLi[k,:] = trainLogLi_this.flatten()
        testLogLi[k,:] = testLogLi_this.flatten()
else: # no cross-validation specified, train on full data
    params,trans,P,emiss_prob,alpha,pred_prob,hist,samples,stationary_prob,trainLogLi_this,testLogLi_this = \
        EMBasins.pyHMM(nrnSpikeTimes, np.ndarray([]), np.ndarray([]), float(binSize), nModes, nIter)
    trainLogLi[0,:] = trainLogLi_this.flatten()
    testLogLi[0,:] = testLogLi_this.flatten()

print(f"Finished fitting mixture model for \
        \n\t interaction factor = 1.\
        \n\t nModes = {nModes}\
        \n\t logL = {trainLogLi}\
        \n\t test logL = {testLogLi}")

saveFit(dataDir, crossValFold, shuffle, nModes, params, trans, P, emiss_prob, alpha, pred_prob, hist, samples, stationary_prob, trainLogLi, testLogLi)
print("Fitted model saved.")
sys.stdout.flush()

