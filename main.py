#!/usr/bin/env python2

import numpy as np
import cv2
import pptk

from db import DB, Feature
from track import Tracker
from match import Matcher
from kalman.ekf import build_ekf
from cho_util.viz import draw_matches, print_ratio
from cho_util import vmath as vm
import cv_util as cvu
from tf import transformations as tx
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# use recorded configuration
# avoid cluttering global namespace
def _CFG_K():
    aov = np.deg2rad([62.2, 48.8])
    foc = np.divide((1640.,922.), 2.*np.tan( aov / 2 ))
    foc = np.divide((1280.,720.), (1640.,922.)) * foc
    fx, fy = foc
    cx, cy = 1280/2., 720/2.
    
    #return np.reshape([
    #    499.114583, 0.000000, 325.589216,
    #    0.000000, 498.996093, 238.001597,
    #    0.000000, 0.000000, 1.000000], (3,3))

    return np.float32([
        [fx,0,cx],
        [0,fy,cy],
        [0,0,1]
        ])

# configuration
CFG = dict(
        scale = 1.0,
        state_size = 15,
        camera_matrix = _CFG_K(),
        pLK = dict(
            winSize         = (31,31),
            maxLevel        = 4,
            crit            = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.03),
            flags           = 0,
            minEigThreshold = 1e-3
            )
        )

class Pipeline(object):
    def __init__(self):
        self.db_        = None
        self.extractor_ = cv2.ORB_create(1024)
        self.matcher_   = Matcher(ex=self.extractor_)
        self.tracker_   = Tracker(pLK=CFG['pLK'])
        self.kf_        = build_ekf()
        self.K_         = CFG['scale'] * CFG['camera_matrix']
        self.K_[2,2] = 1.0
        print(self.K_)

    def initialize(self, img):
        ex       = self.extractor_
        img_fmt  = (img.shape, img.dtype)
        dsc_t    = (np.uint8 if ex.descriptorType() == cv2.CV_8U else np.float32)
        dsc_fmt  = (self.extractor_.descriptorSize(), dsc_t)
        self.db_ = DB(img_fmt=img_fmt, dsc_fmt=dsc_fmt)

    def add_frame(self, img, prv=None, dt=None):
        # obtain pose and covariance
        if (prv is not None) and (dt is not None):
            # apply motion model
            self.kf_.x = prv['pose']
            self.kf_.P = prv['cov']
            self.kf_.predict(dt)

            x = self.kf_.x
            P = self.kf_.P
        else:
            # "initial guess"
            x = np.zeros(CFG['state_size'])
            P = 1e-6 * np.eye(CFG['state_size'])

        # automatic index assignment
        # WARN : add_frame() should NOT be called multiple times!
        index = self.db_.frame.size

        # by default, not a keyframe
        is_kf = False
        kpt, dsc = self.extractor_.detectAndCompute(img, None)
        feat = Feature(kpt, dsc, cv2.KeyPoint.convert(kpt))
        frame = (index, img, x, P, is_kf, feat)

        self.db_.frame.append(frame)

        return frame

    def process(self, img, dt, data={}):
        if self.db_ is None:
            # initialization
            self.initialize(img)
            self.add_frame(img)
            kpt = self.db_.frame[-1]['feat'].kpt
            self.db_.frame[-1]['is_kf'] = True
            self.db_.state_['track'] = cv2.KeyPoint.convert(kpt)
        else:
            # populate frame
            print('\t\tcurrent index : {}'.format(self.db_.frame.size))
            self.add_frame(img, prv=self.db_.frame[-1], dt=dt)

            # fetch prv+cur frames
            #frame0 = self.db_.frame[-2]
            frame0 = self.db_.keyframe[-1] # last **keyframe**
            frame1 = self.db_.frame[-1]

            # process ...
            img0, img1   = frame0['image'], frame1['image']
            feat0, feat1 = frame0['feat'], frame1['feat']

            # match
            mi0, mi1 = self.matcher_.match(feat0.dsc, feat1.dsc,
                    lowe=0.75, fold=False) # harsh lowe to avoid pattern collision
            pt0m = feat0.pt[mi0]
            pt1m = feat1.pt[mi1]
            
            #E, msk = cvu.E(pt1m, pt0m, self.K_)
            #_, R, t, msk = cv2.recoverPose(E, pt1m, pt0m, self.K_)
            suc = cvu.Reconstructor(pt0m, pt1m, self.K_).compute(data=data)
            print('suc?', suc)
            if not suc:
                return
            frame1['is_kf'] = True # TODO : not really keyframe right now -- just reference frame

            #cv2.imshow('img0', img0)
            #cv2.imshow('img1', img1)

            data['img1'] = img1
            data['pt1m'] = pt1m

            R = data['R']
            t = data['t']
            msk = data['msk_cld']

            #print('msk', msk)
            #print parallax.min(), parallax.max(), parallax.mean()
            #print_ratio(msk_good.sum(), msk_good.size)

            #plt.hist(parallax, bins='auto')
            #plt.show()

            #print('d0', np.median(pt_R3_0[..., 2]) )
            #print('d1', np.median(pt_R3_1[..., 2]) )
            print('R', np.rad2deg(tx.euler_from_matrix(R)))

            viz0 = img0 #cv2.drawKeypoints(img0, feat0.kpt, None)
            viz1 = img1 #cv2.drawKeypoints(img1, feat1.kpt, None)
            viz  = draw_matches(viz0, viz1, pt0m[msk], pt1m[msk])
            cv2.imshow('viz', viz)
            data['viz'] = viz

            # track
            #t_pt, t_idx = self.tracker_.track(img0, img1, self.db_.state_['track'])
            #self.db_.state_['track'] = t_pt[t_idx]
            
def main():
    pl  = Pipeline()
    src = './scan_20190212-233625.h264'
    #src = '/tmp/tmp.mp4'
    cap = cv2.VideoCapture(src)
    dt = (1.0 / 25.0) # (??)
    cv2.namedWindow('viz', cv2.WINDOW_NORMAL)
    cnt = 0
    while True:
        res, img = cap.read()
        cnt += 1
        if(cnt <= 3):
            continue
        if not res: break
        img = cv2.resize(img, None, fx=CFG['scale'], fy=CFG['scale'])
        data = {}
        pl.process(img, dt, data)
        if 'viz' in data:
            cv2.imshow('viz', data['viz'])
        k = cv2.waitKey(0)
        if k in [27, ord('q')]: break

        #try:
        #    msk1 = data['msk_cld']
        #    idx1 = vm.rint(data['pt1m'][msk1][...,::-1])
        #    col1 = data['img1'][idx1[:,0], idx1[:,1]]
        #    cld1 = data['cld1'][msk1]
        #    ax = plt.gca(projection='3d')
        #    ax.scatter(cld1[:,0], cld1[:,1], cld1[:,2],
        #            c = (col1[...,::-1] / 255.0))
        #    ax.set_xlabel('x')
        #    ax.set_ylabel('y')
        #    ax.set_zlabel('z')
        #    #plt.pause(0.001)
        #    plt.show()
        #    #v = pptk.viewer(
        #    #        data['cld1'][msk1], col1[...,::-1]
        #    #        )
        #    #v.wait()
        #    #v.close()
        #except Exception as e:
        #    print e
        #    continue

if __name__ == '__main__':
    main()
