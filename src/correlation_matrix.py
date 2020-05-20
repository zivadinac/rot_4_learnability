from argparse import ArgumentParser
from os import path
import data_utils
import numpy as np
from scipy.stats import pearsonr

def computeCorrelationMatrix(spikes):
    corrs = []
    corr_mat = np.eye(spikes.shape[0])
    for i in range(spikes.shape[0]):
        for j in range(i+1, spikes.shape[0]):
            c = pearsonr(spikes[i], spikes[j])[0]
            corr_mat[i,j] = corr_mat[j,i] = c
            corrs.append(c)
    corrs = np.array(corrs)
    return corr_mat#, corrs.mean(), corrs.abs().mean(), np.median(corrs)

def __createCorrDataFileName(args, ps):
    stim = path.basename(args.data_path).split('.')[0]
    return f"corr_ps_{ps}.pck"

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--data_path")
    args.add_argument("--out_path")
    args = args.parse_args()

    spikes = data_utils.loadSpikeData(args.data_path).data
    ps = spikes.shape[0]
    spikes, retained_neurons, excluded_neurons = data_utils.excludeNonFiringNeurons(spikes)
    assert ps == len(retained_neurons) + len(excluded_neurons)
    corr = computeCorrelationMatrix(spikes)
    d = {"corr_mat": corr, "population_size": ps,\
         "retained_neurons": retained_neurons, "excluded_neurons": excluded_neurons}

    data_utils.save(path.join(args.out_path, __createCorrDataFileName(args, ps)), d)

