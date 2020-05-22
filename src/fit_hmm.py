from argparse import ArgumentParser
import numpy as np
import sys
from os import path
import pickle
import data_utils
import TreeHMM
import EMBasins
import utils

# TODO naming convention - change to snake_case as in other files

args = ArgumentParser()
args.add_argument("data_path")
args.add_argument("out_path")
args.add_argument("--cross_val_folds", default=2, type=int, help="k-fold validation, 1 for train only")
args.add_argument("--n_modes", default=70, type=int, help="Number of modes to use in HMM")
args.add_argument("--eta", default=0.002, type=float, help="HMM regularization param.")
args.add_argument("--n_iter", default=100, type=int, help="HMM training iterations.")
args.add_argument("--seed", type=int, default=12345)
args = args.parse_args()

spikes = data_utils.loadSpikeData(args.data_path).data
hmm = TreeHMM.trainHMM(spikes, args.n_modes, args.n_iter, args.eta, cross_val_folds=args.cross_val_folds, seed=args.seed)

fit_path = path.join(args.out_path, path.basename(args.data_path).split('.')[0] + f"_nModes_{args.n_modes}.pck")
TreeHMM.io.saveTrainedHMM(fit_path, hmm)
print(f"Fitted model saved to {fit_path}.")

