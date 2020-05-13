import sys
import pickle
import numpy as np
from skimage.io import imsave
import matplotlib.pyplot as plt
from matplotlib import cm as cm

class ObjView():
    """
        Simple class that enables viewing dict as an object (useful for REPL workflow).
    """
    def __init__(self, dictionary):
        self.__dict__ = dictionary

    def __getitem__(self, key):
        return self.__dict__[key]

    """
    # for now I just need view, no modification
    def __setitem__(self, key, value):
        self.__dict__[key] = value
    """

def saveTiff(rf, path="rf.tiff"):
    rf -= rf.min()
    rf /= rf.max()
    rf *= 255
    rf = rf.numpy()
    rf = rf.astype(np.uint8)
    imsave(path, rf)

def saveArgs(out_path):
    with open(out_path, 'w') as out_file:
        out_file.write(' '.join(sys.argv[1:]))

def loadFit(fit_path, as_object=True):
    with open(fit_path, "rb") as fit_file:
        fit = pickle.load(fit_file)

    return ObjView(fit) if as_object else fit

def saveFit(fit_path, cross_val_fold, shuffle, n_modes, \
            params, trans, P, emiss_prob, alpha, pred_prob, \
            hist, samples, stationary_prob, \
            train_log_li, test_log_li):
    hmmFit = {}
    hmmFit['params'] = params
    hmmFit['trans'] = trans
    hmmFit['P'] = P
    hmmFit['emiss_prob'] = emiss_prob
    hmmFit['alpha'] = alpha
    hmmFit['pred_prob'] = pred_prob
    hmmFit['hist'] = hist
    hmmFit['samples'] = samples
    hmmFit['stationary_prob'] = stationary_prob
    hmmFit['train_log_li'] = train_log_li
    hmmFit['test_log_li'] = test_log_li
    hmmFit["cross_val_fold"] = cross_val_fold
    hmmFit["n_modes"] = n_modes
    with open(fit_path, "wb") as fitFile:
        pickle.dump(hmmFit, fitFile)

def plotCorrelationMatrix(corr_mat, title):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.title(title)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(corr_mat, interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    labels = np.arange(corr_mat.shape[0])
    ax1.set_xticklabels(labels,fontsize=6)
    ax1.set_yticklabels(labels,fontsize=6)
    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.show()
