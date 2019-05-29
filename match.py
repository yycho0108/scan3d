#!/usr/bin/env python2

import cv2
import numpy as np
from cho_util import vmath as M
import time
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

    # generate flattened index pair
    idx1 = [np.full(len(i2), i1) for (i2, i1) in zip(idx2, np.arange(len(pt1))) ]
    idx1 = np.concatenate(idx1)
    idx2 = np.concatenate(idx2)

    # compute distance
    if hamming:
        d = hamming_distance(dsc1[idx1], dsc2[idx2])
    else:
        d = np.linalg.norm(dsc1[idx1] - dsc2[idx2], axis=-1)

    # sort by hamming distance to apply np.unique()
    sort_idx = np.argsort(d)
    d    = d[sort_idx]
    idx1 = idx1[sort_idx]
    idx2 = idx2[sort_idx]

    # apply max distance mask
    msk  = (d < maxd )
    idx1 = idx1[msk]
    d    = d[msk]
    idx2 = idx2[msk]

    # return "best"
    idx1, unsort_idx = np.unique(idx1, return_index=True)
    #idx1 = idx1[unsort_idx]
    idx2 = idx2[unsort_idx]
    return idx1, idx2

    #msk = (d < maxd)
    #msk = (hd < maxd)

    ## apply hamming mask
    #idx1 = idx1[msk]
    #hd   = hd[msk]
    #idx2 = idx2[msk]

    #print idx1.shape
    #print idx2.shape
    #print len(idx), len(pt1)
    # dist, idx = nbrs.kneighbors(pt1)
    # hd        = hamming_distance(dsc1, dsc2[idx])
    # print hd.shape
    # print dist.shape

    #idx0, idx1 = idx[:,0], idx[:,1]
    #hd1 = hamming_distance(dsc1, dsc2[idx0]
    #dist < radius

class Matcher(object):
    PRESET_HARD=dict(
            lowe=0.75,
            maxd=32.0,
            cross=True,
            fold=True
            )
    PRESET_SOFT=dict(
            lowe=1.0,
            maxd=128.0,
            cross=False,
            fold=True
            )

    def __init__(self, ex=None):
        # define un-exported enums from OpenCV
        FLANN_INDEX_LINEAR = 0
        FLANN_INDEX_KDTREE = 1
        FLANN_INDEX_KMEANS = 2
        FLANN_INDEX_COMPOSITE = 3
        FLANN_INDEX_KDTREE_SINGLE = 4
        FLANN_INDEX_HIERARCHICAL = 5
        FLANN_INDEX_LSH = 6
        FLANN_INDEX_SAVED = 254
        FLANN_INDEX_AUTOTUNED = 255

        # TODO : figure out what to set for
        # search_params
        search_params = dict(checks=50)
        # or pass empty dictionary

        # build flann matcher
        self.des_t_ = (np.uint8 if ex.descriptorType() == cv2.CV_8U
                else np.float32)

        if self.des_t_ == np.uint8:
            # probably hamming
            # ~HAMMING
            index_params= dict(algorithm = FLANN_INDEX_LSH,
                       table_number = 12,#6, # 12
                       key_size = 20,#12,     # 20
                       multi_probe_level = 2) #2
            flann = cv2.FlannBasedMatcher(index_params,search_params)
            self.match_ = flann
            #bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            #self.match_ = bf
        else:
            # Euclidean
            index_params = dict(
                    algorithm = FLANN_INDEX_KDTREE,
                    trees = 5)
            flann = cv2.FlannBasedMatcher(index_params,search_params)
            self.match_ = flann

    def _match(self, a, b):
        """ search k matches in b for a """
        return self.match_.knnMatch(
                self.des_t_(a), self.des_t_(b), k=2)

    def filter(self, match, lowe, maxd):
        """
        Apply lowe + distance filter.
        # TODO : set threshold for lowe's filter
        # TODO : set reasonable maxd for GFTT, for instance.
        """
        good = []
        for e in match:
            if not len(e) == 2:
                continue
            (m, n) = e
            if not (m.distance <= lowe * n.distance):
                continue
            #print m.distance
            if not (m.distance <= maxd):
                continue

            # passed all checks
            good.append(m)
        return good

    def match(self, des1, des2,
            lowe=0.75,
            maxd=64.0,
            cross=True,
            fold=True,
            ):
        """
        Arguments :
            des1  : vector of keypoint-corresponding descriptors for first frame.
            des2  : vector of keypoint-corresponding descriptors for second frame.
            lowe  : lowe ratio
            maxd  : maximum acceptable "distance" for matching -- dependent on descriptor type
            cross : whether to apply cross-check (two-way best-match)
            fold  : whether to apply match on pre-filtered data (becomes non-symmetric)

        Returns :
            i1, i2 : index into match data, such that des1[i1] == des2[i2]
        """

        # soft fail in case des1/des2 is empty
        if len(des1) <= 0 or len(des2) <= 0:
            return np.int32([]), np.int32([])

        i1, i2 = None, None
        if cross:
            # check bidirectional
            i1_ab, i2_ab = self.match(des1, des2,
                    lowe, maxd, cross=False)
            if fold:
                # opt1 : apply match on pre-filtered data ( faster )
                i2_ba, i1_ba = self.match(des2[i2_ab], des1[i1_ab],
                        lowe, maxd, cross=False)
                i1, i2 = i1_ab[i1_ba], i2_ab[i2_ba]
            else:
                # opt2 : apply the same operation reversed ( slower, maybe more robust ?? )
                i2_ba, i1_ba = self.match(des2, des1,
                        lowe, maxd, cross=False)
                m1 = np.stack([i1_ab, i2_ab], axis=-1)
                m2 = np.stack([i1_ba, i2_ba], axis=-1)
                m  = M.intersect2d(m1, m2)
                i1, i2 = m.T

        else:
            # check unidirectional (des1->des2)
            if len(des1) < 2:
                # insufficient # of descriptors
                return np.int32([]), np.int32([])

            match = self._match(des1, des2)
            match = self.filter(match, lowe, maxd)

            # extract indices
            i1, i2 = np.int32([
                (m.queryIdx, m.trainIdx) for m in match
                ]).reshape(-1,2).T

        return i1, i2

def main():
    # specific imports
    from cho_util.viz.draw import draw_matches

    np.random.seed( 0 )

    orb = cv2.ORB_create(
        nfeatures=1024,
        scaleFactor=1.2,
        nlevels=8,
        # NOTE : scoretype here influences response-based filters.
        #scoreType=cv2.ORB_FAST_SCORE,
        scoreType=cv2.ORB_HARRIS_SCORE,
        )
    matcher = Matcher(ex=orb)

    img1 = np.random.randint(0, 255, size=(6,8,3), dtype=np.uint8)
    img1 = cv2.resize(img1, (640,480), interpolation=cv2.INTER_NEAREST)

    #img2 = np.random.randint(0, 255, size=(480,640,3), dtype=np.uint8)
    roll_i = np.random.randint( 64 )
    roll_j = np.random.randint( 64 )

    img2 = np.roll(img1, roll_i, axis=0)
    img2 = np.roll(img2, roll_j, axis=1)
    kpt1, des1 = orb.detectAndCompute(img1, None)
    kpt2, des2 = orb.detectAndCompute(img2, None)
    pt1 = cv2.KeyPoint.convert(kpt1)
    pt2 = cv2.KeyPoint.convert(kpt2)

    i1, i2 = matcher.match(des1, des2)

    d = (pt1[i1] - pt2[i2]) + (roll_j, roll_i)
    tol = 5.0
    msk = (np.linalg.norm(d, axis=-1) < tol)
    print msk.sum(), msk.size

    mim = draw_matches(img1, img2,
            pt1[i1], pt2[i2])
    cv2.imshow('win', mim)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
