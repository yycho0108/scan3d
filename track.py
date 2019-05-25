import time
import cv2
import numpy as np
from collections import defaultdict
from cho_util.cam import KeyCallback

class Tracker(object):
    def __init__(self, pLK=None):
        if pLK is None:
            # default LK param
            pLK = self.pLK0()
        self.lk_ = cv2.SparsePyrLKOpticalFlow_create(
                **pLK)
        self.tmp_ = defaultdict(lambda:None)

    def pLK0(self):
        """
        Default LK Params.
        SparsePyrLKOpticalFlow_create([, winSize[, maxLevel[, crit[, flags[, minEigThreshold]]]]]) -> retval
        """
        return dict(
                winSize = (12,6),
                maxLevel = 4, # == effective winsize up to 32*(2**4) = 512x256
                crit= (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.03),
                flags = 0,
                minEigThreshold = 1e-3 # TODO : disable eig?
                )

    def track(self, 
            img1, img2,
            pt1, pt2=None,
            thresh=2.0,
            return_msk=False
            ):
        """
        Arguments:
            img1(np.ndarray) : previous image. (color/mono) (HxWx?)
            img2(np.ndarray) : current image (color/mono) (HxWx?)
            pt1(np.ndarray)  : previous points. (Mx2)
            pt2(np.ndarray)  : [Optional] current points estimate (Mx2)
            thresh(float)    : Flow Back-projection Error threshold

        Returns:
            pt2(np.ndarray)  : current points. (Mx2)
            idx(np.ndarray)  : valid tracked indices from pt1 & pt2.
        """
        if pt1.size <= 0:
            # soft fail
            pt2 = np.empty([0,2], dtype=np.float32)
            if return_msk:
                msk = np.empty([0], dtype=np.bool)
                return pt2, msk
            idx = np.empty([0], dtype=np.int32)
            return pt2, idx

        # stat img
        h, w = np.shape(img2)[:2]

        # convert to grayscale
        # TODO : check if already gray/mono

        if (np.ndim(img1) == 2) or img1.shape[2] == 1:
            # already monochromatic
            img1_gray = img1
            img2_gray = img2
        else:
            # handle image # 1 + pre-allocated data cache
            if self.tmp_['img1g'] is not None:
                cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY, self.tmp_['img1g'])
                img1_gray = self.tmp_['img1g']
            else:
                img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                self.tmp_['img1g'] = np.empty_like(img1_gray)

            # handle image # 2 + pre-allocated data cache
            if self.tmp_['img2g'] is not None:
                cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY, self.tmp_['img2g'])
                img2_gray = self.tmp_['img2g']
            else:
                img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                self.tmp_['img2g'] = np.empty_like(img2_gray)

        # forward flow
        if pt2 is not None:
            # set initial flow flags
            self.lk_.setFlags(self.lk_.getFlags() | cv2.OPTFLOW_USE_INITIAL_FLOW )
            pt2, st, _ = self.lk_.calc(
                    img1_gray, img2_gray, pt1, pt2
                    )
        else:
            pt2, st, _ = self.lk_.calc(
                    img1_gray, img2_gray, pt1, None
                    )
        st_fw = st[:,0].astype(np.bool)

        # backward flow
        # unset initial flow flags
        self.lk_.setFlags(self.lk_.getFlags() & ~cv2.OPTFLOW_USE_INITIAL_FLOW )
        pt1_r, st, _ = self.lk_.calc(
                img2_gray, img1_gray, pt2, None
                )
        st_bw = st[:,0].astype(np.bool)

        # override error with reprojection error
        # (default error doesn't make much sense anyways)
        err = np.linalg.norm(pt1 - pt1_r, axis=-1)

        # apply mask
        msk = np.logical_and.reduce([
            # error check
            err < thresh,
            # bounds check
            0 <= pt2[:,0],
            0 <= pt2[:,1],
            pt2[:,0] < w,
            pt2[:,1] < h,
            # status check
            st_fw,
            st_bw,
            ])

        if return_msk:
            return pt2, msk
        else:
            idx = np.where(msk)[0]
            return pt2, idx

def main():
    from matplotlib import pyplot as plt
    # params
    w = 2*640
    h = 2*480
    n = 2*1024
    di = 8
    dj = 32

    track = Tracker(pLK=dict(
        winSize=(15,15),
        maxLevel=8,
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.03),
        flags = 0,
        minEigThreshold = 1e-3
        ))

    cap = cv2.VideoCapture('./scan_20190212-233625.h264')
    kcb = KeyCallback()

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)

    p_trk = []
    while True:
        # input image
        ret, img = cap.read()
        if not ret:
            break
        #img = cv2.resize(img, None, fx=0.25, fy=0.25)

        # process
        #pt1  = np.random.uniform((0,0), (w,h), size=(n,2)).astype(np.float32)

        # track
        if len(p_trk) < 128:
            mono  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            p_trk = cv2.goodFeaturesToTrack(mono, 512, 0.01, 10)
            p_trk = np.squeeze(p_trk, axis=1)
            prv   = img
            continue

        pt2, idx = track(prv, img, p_trk)
        prv = img
        p_trk = pt2[idx]
        viz = img.copy()
        for p in p_trk:
            cv2.circle(viz, tuple(p), 3, (255,255,0))

        cv2.imshow('img', viz)
        #except Exception as e:
        #    print('e : {} .. {}'.format(e,sys.exc_info()))
        #    break

        # key callback handling
        k = cv2.waitKey(0)
        if kcb(k):
            break

if __name__ == "__main__":
    main()
