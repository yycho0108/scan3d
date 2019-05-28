"""
compile:
    cython match_local.pyx # produces match_local.c
    gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -o match_local.so match_local.c
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors

_hdref = np.unpackbits(np.arange(256, dtype=np.uint8)).reshape(256,-1).sum(axis=-1)
def hamming_distance(x0, x1):
    x0, x1 = np.broadcast_arrays(x0, x1)
    hd = _hdref[np.bitwise_xor(x0.ravel(), x1.ravel()).view('u1')]
    shape = list(x0.shape[:-1])
    shape.append(-1)
    return hd.reshape(shape).sum(axis=-1)

def match_local(pt1, pt2, dsc1, dsc2,
        radius=15.0,
        lowe  = 0.7,
        maxd  = 64.,
        hamming=True
        ):
    # binning by angular displacement
    nbrs = NearestNeighbors(n_neighbors=2, radius=radius).fit(pt2)
    idx2 = nbrs.radius_neighbors(pt1, return_distance=False)
    i1_out = []
    i2_out = []
    for (i1, i2) in enumerate(idx2):
        if i2.size <= 0: 
            continue
        hd  = hamming_distance(dsc1[None, i1], dsc2[i2])
        sel = np.argmin(hd)
        if hd[sel] >= maxd:
            continue
        i1_out.append( i1 )
        i2_out.append( i2[sel] )
    return i1_out, i2_out
