from argparse import ArgumentParser
from os import path
import numpy as np
from scipy.io import loadmat
from skimage.io import imsave
import matplotlib.pyplot as plt
import data_utils
import utils
from scipy.stats import pearsonr
from pandas import DataFrame

args = ArgumentParser()
args.add_argument("--data_paths", nargs='*', help="List of files with correlation matrices for different population sizes.")
args.add_argument("--out_path", help="Directory in which to save resulting figure.")
args = args.parse_args()

col_num = 3
row_num = int(len(args.data_paths) // col_num) + 1
fig, ax = plt.subplots(row_num, col_num, figsize=(25, 13))

sim_means = [{}, {}] # 0 - mean, 1 - abs mean
pre_mean = pre_abs_mean = 0
for i in range(row_num):
    for j in range(col_num):
        f = i * col_num + j
        if f >= len(args.data_paths):
            break

        f = args.data_paths[f]
        corr_data = data_utils.load(f)
        ps = corr_data.population_size
        corrs = corr_data.corr_mat[np.triu_indices(len(corr_data.retained_neurons), k = 1)]
        corr_median = np.round(np.median(corrs), 5)
        corr_mean = np.round(corrs.mean(), 5)
        corr_abs_mean = np.round(np.abs(corrs).mean(), 5)

        is_prentice = "prentice" in f
        if is_prentice:
            pre_mean = corr_mean
            pre_abs_mean = corr_abs_mean
        else:
            sim_means[0][ps] = corr_mean
            sim_means[1][ps] = corr_abs_mean

        data_name = "Prentice" if is_prentice else "Simulated"
        ax[i, j].set_title(f"{data_name} pairwise correlations, population size {ps}")
        ax[i, j].hist(corrs, bins=300, label=f"mean={corr_mean}, abs mean = {corr_abs_mean}, median = {corr_median}")
        ax[i, j].legend()


for i in range(1,3):
    m = np.array(list(sim_means[i-1].values()))
    pre_m = pre_mean if i-1 == 0 else pre_abs_mean
    pss = list(sim_means[i-1].keys())

    plt_name = "Mean" if i-1 == 0 else "Abs mean"
    ax[-1, i].set_title(f"{plt_name} pairwise correlation for different population sizes")
    ax[-1, i].plot(pss, m)
    plt.setp(ax[-1, i], xticklabels=pss)
    plt.sca(ax[-1, i])
    plt.xticks(pss)
    rel = ax[-1, i].twinx()
    rel.plot(pss, m / pre_m)

plt.savefig(path.join(args.out_path, "corr_hists.png"), dpi=fig.dpi)
#plt.show()
