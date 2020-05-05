import sys
import pickle
import numpy as np
from skimage.io import imsave

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

def loadFit(fit_path):
    with open(fit_path, "rb") as fit_file:
        fit = pickle.load(fit_file)
    return fit

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
