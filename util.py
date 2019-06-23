#!/usr/bin/env python2

import cv2
import cv_util as cvu
import numpy as np
from db import L_POS, L_VEL, L_ACC, A_POS, A_VEL
from cho_util import vmath as vm
from cho_util.math import transform as tx
from sklearn.neighbors import NearestNeighbors

def pose_to_xfm(pose):
    txn = pose[L_POS]
    rxn = pose[A_POS]
    return tx.compose(r=rxn, t=txn, rtype=tx.rotation.euler)

def index_remap(idx):
    # source = (0, 0, 2, 2, 2)
    # uniq   = (0, 2)
    # alt_idx = (0, 1)
    # i_s2a   = [0, X, 1]
    # i_a2s   = [0, 1],

    ibw, i_new = np.lib.arraysetops.unique(idx,
            return_inverse=True
            )
    return np.arange(len(ibw)), ibw, i_new

def get_transform(source_frame, target_frame):
    if source_frame is None:
        # assume source=map
        xfm_s2m = np.eye(4)
    else:
        xfm_s2m = pose_to_xfm( source_frame['pose'] )

    xfm_t2m = pose_to_xfm( target_frame['pose'] )
    xfm_s2t = tx.invert(xfm_t2m).dot(xfm_s2m)

    R = xfm_s2t[:3, :3]
    t = xfm_s2t[:3, 3:]
    return R, t

def transform_cloud(cloud, source_frame, target_frame):
    R, t = get_transform(source_frame, target_frame)
    return vm.rtx3(R, t.ravel(), cloud)

def project_to_frame(cloud, source_frame, target_frame, K, D):
    R, t = get_transform(source_frame, target_frame)

    cloud = np.float64(cloud)
    rvec = cv2.Rodrigues(R)[0].ravel()
    tvec = np.float32(t).ravel()

    res = cvu.project_points(cloud, rvec, tvec,
            K, D)
    return res

def extract_color(img, pt, wsize=1):
    h, w = img.shape[:2]
    idx = vm.rint(pt)[..., ::-1] # (x,y) -> (i,j)
    idx = np.clip(idx, (0,0), (h-1,w-1)) # TODO : fix this hack

    if wsize == 1:
        col = img[idx[:,0], idx[:,1]]
    else:
        # TODO : actually use wsize argument
        di, dj = np.mgrid[-1:2,-1:2]
        di = di.reshape(-1,1)
        dj = dj.reshape(-1,1)
        col = img[di + idx[:,0], dj + idx[:,1]]
        col = np.mean(col, axis=0)
    return col

def non_max(pt, rsp,
        k=16,
        radius=4.0, # << NOTE : supply valid radius here when dealing with 2D Data
        thresh=0.25
        ):

    # NOTE : somewhat confusing;
    # here suffix c=camera, l=landmark.
    # TODO : is it necessary / proper to take octaves into account?
    if len(pt) < k:
        # Not enough references to apply non-max with.
        return np.arange(len(pt))

    # compute nearest neighbors
    neigh = NearestNeighbors(n_neighbors=k, radius=np.inf)
    neigh.fit(pt)

    # NOTE : 
    # radius_neighbors would be nice, but indexing is difficult to use
    # res = neigh.radius_neighbors(pt_new, return_distance=False)
    d, i = neigh.kneighbors(return_distance=True)
    print(len(i))
    # TODO : `same_octave` condition

    # too far from other landmarks to apply non-max
    msk_d = (d.min(axis=1) >= radius)
    print('msk_d', msk_d.sum())
    # passed non-max
    msk_v = np.all(rsp[i] * thresh < rsp[:,None], axis=1) # 
    print('msk_v', msk_v.sum())

    # format + return results
    msk = (msk_d | msk_v)
    idx = np.where(msk)[0]
    #print_ratio('non-max', len(idx), msk.size)
    return idx
