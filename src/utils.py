import sys
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

