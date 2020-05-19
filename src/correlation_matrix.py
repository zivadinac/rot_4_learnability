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
    return corrs, corrs.mean(), np.median(corrs)

def __createCorrDataFileName(args):
    stim = path.basename(args.data_path).split('.')[0]
    ps = stim.split('_')[-1]
    return f"corr_ps_{ps}.pck"

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--data_path")
    args.add_argument("--out_path")
    args.add_argument("--simulated_data", type=int, default=1)
    args = args.parse_args()

    spikes = data_utils.loadSimulatedData(args.data_path).data if args.simulated_data else data_utils.loadPrenticeEtAl2016(args.data_path)
    spikes, retained_inds, excluded_inds = data_utils.excludeNonFiringNeurons(spikes)
    corr = computeCorrelationMatrix(spikes)
    d = {"corr_mat": corr, "retained_neurons": retained_inds, "excluded_neurons": excluded_inds}

    data_utils.save(path.join(args.out_path, __createCorrDataFileName(args)), d)

