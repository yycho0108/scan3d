#!/usr/bin/env python2

import numpy as np
import cv2
import pptk

from db import DB, Feature
from track import Tracker
from match import Matcher, match_local
from kalman.ekf import build_ekf
from cho_util.viz import draw_matches, print_ratio
from cho_util import vmath as vm
import cv_util as cvu
from tf import transformations as tx
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from reconstructor import Reconstructor
from reader.advio import AdvioReader

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
        K = _CFG_K(),
        pLK = dict(
            winSize         = (31,31),
            maxLevel        = 4,
            crit            = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.03),
            flags           = 0,
            minEigThreshold = 1e-3
            )
        )

from enum import Enum

class PipelineState(Enum):
    IDLE     = 0 # default state: do nothing
    NEED_REF = 1 # need reference frame
    NEED_MAP = 2 # need triangulated map
    TRACK    = 3 # tracking!
    LOST     = 4 # lost and sad!

class Pipeline(object):
    def __init__(self, cfg):
        self.cfg_       = self.build_cfg(cfg)
        self.extractor_ = cv2.ORB_create(1024, edgeThreshold=19)
        self.matcher_   = Matcher(ex=self.extractor_)
        self.tracker_   = Tracker(pLK=cfg['pLK'])
        self.kf_        = build_ekf()
        self.db_        = self.build_db()
        self.state_     = PipelineState.NEED_REF

    def build_cfg(self, cfg):
        # build derived values

        # image shape
        shape = (cfg['h'], cfg['w'], 3) # TODO : handle monochrome

        # scaled camera matrix
        K0 = cfg['K']
        K = cfg['scale'] * cfg['K']
        K[2,2] = 1.0

        # first, make a copy from argument
        cfg = dict(cfg)

        # insert derived values
        cfg['shape'] = shape
        cfg['K0']    = K0
        cfg['K']     = K

        return cfg

        ## create dynamic type
        #ks = cfg.keys()
        #cfg_t = namedtuple('PipelineConfig', ks)
        ## setup dot-referenced aliases
        #for k, v in cfg.iteritems():
        #    setattr(cfg, k, v)

        return cfg

    def build_db(self):
        cfg = self.cfg_
        img_shape = (cfg['h'], cfg['w'], 3)
        ex       = self.extractor_
        img_fmt  = (img_shape, np.uint8)
        dsc_t    = (np.uint8 if ex.descriptorType() == cv2.CV_8U else np.float32)
        dsc_fmt  = (self.extractor_.descriptorSize(), dsc_t)
        return DB(img_fmt=img_fmt, dsc_fmt=dsc_fmt)

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
            x = np.zeros(self.cfg_['state_size'])
            P = 1e-6 * np.eye(self.cfg_['state_size'])

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

    def transition(self, new_state):
        print('[state] ({} -> {})'.format(
            self.state_, new_state))
        self.state_ = new_state

    def init_ref(self, img, dt, data):
        """ initialize reference """
        # TODO : handle failure to initialize reference frame
        # == possibly filter for richness of tracking features?
        self.add_frame(img)
        kpt = self.db_.frame[-1]['feat'].kpt
        self.db_.frame[-1]['is_kf'] = True # reference frame - considered keyframe
        self.db_.state_['track'] = cv2.KeyPoint.convert(kpt)
        self.transition( PipelineState.NEED_MAP )

    def init_map(self, img, dt, data):
        """ initialize map """
        # populate frame from motion model
        self.add_frame(img, prv=self.db_.keyframe[-1], dt=0) # TODO : restore dt maybe

        # fetch prv+cur frames
        #frame0 = self.db_.frame[-2]
        frame0 = self.db_.keyframe[-1] # last **keyframe**
        frame1 = self.db_.frame[-1]

        #print('target pair : {}-{}'.format(
        #    frame0['index'],
        #    frame1['index']))

        # process ...
        img0, img1   = frame0['image'], frame1['image']
        feat0, feat1 = frame0['feat'], frame1['feat']

        # match
        mi0, mi1 = self.matcher_.match(feat0.dsc, feat1.dsc,
                lowe=0.75, fold=False) # harsh lowe to avoid pattern collision
        pt0m = feat0.pt[mi0]
        pt1m = feat1.pt[mi1]
        
        suc, det = Reconstructor(pt0m, pt1m, self.cfg_['K']).compute(data=data)
        if not suc:
            # unsuccessful frame-to-frame reconstruction
            #print('\t det : {}'.format(det))
            if not det['num_pts']:
                # condition: `tracking lost`
                # BEFORE sufficient parallax was observed.
                # need to reset the reference frame.
                print('\t -- reset keyframe')
                # reset keyframe
                frame1['is_kf'] = True
            return
        
        # here, init successful
        cld_msk = data['msk_cld']
        cld1 = data['cld1'][cld_msk]
        # IMPORTANT : everything references **frame1** !!
        # (so frame0 geometric information is effectively ignored.)
        self.local_map_ = [
                np.arange(self.db_.landmark.size, self.db_.landmark.size + len(cld1)),
                feat1.dsc[mi1][cld_msk],
                cld1 # ,is_traking=True?
                ]
        self.db_.landmark.extend(zip(*self.local_map_))
        #print self.db_.landmark.size

        # unroll reconstruct information
        R   = data['R']
        t   = data['t']
        msk = data['msk_cld']

        # update frame data
        #[0pos,3vel,6acc,9rpos,12rvel]
        frame1['pose'][0  : 3]  = 0.0 # t.reshape(-1)
        frame1['pose'][3  : 6]  = 0.0 # reserve judgement about vel here.
        frame1['pose'][6  : 9]  = 0.0 # reserve judgement about acc here.
        frame1['pose'][9  : 12] = 0.0 # tx.euler_from_matrix(R) # TODO : apply proper R?
        frame1['pose'][12 : 15] = 0.0 # reserve judgement about rvel here.
        frame1['is_kf'] = True

        # YAY! Able to start tracking the landmarks.
        # state transition ...
        self.transition( PipelineState.TRACK )

        # update output debug data
        data['img1'] = img1
        data['pt1m'] = pt1m

        # by default, ALL visualizations MUST come
        # at the end of each processing function.
        if True:
            # == if cfg['dbg']:
            print('R', np.rad2deg(tx.euler_from_matrix(R)))
            print('frame pair : {}-{}'.format(
                frame0['index'],
                frame1['index']))

            viz0 = cv2.drawKeypoints(img0, feat0.kpt, None)
            viz1 = cv2.drawKeypoints(img1, feat1.kpt, None)
            viz  = draw_matches(viz0, viz1, pt0m[msk], pt1m[msk])
            cv2.imshow('viz', viz)
            data['viz'] = viz

    def track(self, img, dt, data={}):
        """ Track landmarks"""
        # populate frame from motion model
        print('track-dt', dt)
        self.add_frame(img, prv=self.db_.frame[-1], dt=dt)

        # unroll data
        frame0 = self.db_.keyframe[-1] # last **keyframe**
        frame1 = self.db_.frame[-1]
        img0  , img1  = frame0['image'] , frame1['image']
        feat0 , feat1 = frame0['feat']  , frame1['feat']

        idx, dsc, cld = self.local_map_
        #[0pos,3vel,6acc,9rpos,12rvel]
        rmat = tx.euler_matrix(*frame1['pose'][9:12])[:3,:3]
        R, t = rmat, frame1['pose'][0:3]
        #R, t = vm.Rti(R, t)
        rvec = cv2.Rodrigues(R)[0]
        tvec = t
        pt0  = cvu.project_points(cld, rvec, tvec,
                self.cfg_['K'], self.cfg_['D']) # cld -> cam coord.

        # search local map
        #self.tracker_.track()...
        mi0, mi1 = match_local(
                pt0, feat1.pt,
                dsc, feat1.dsc
                )
        #suc, rvec, tvec = cv2.solvePnP(
        #        cld[mi0], feat1.pt[mi1],
        #        self.cfg_['K'], self.cfg_['D']
        #        ) # T(rv,tv) . cld = cam
        #print 'euler', tx.euler_from_matrix(cv2.Rodrigues(rvec)[0])
        suc, rvec, tvec, inl = cv2.solvePnPRansac(
                cld[mi0], feat1.pt[mi1],
                self.cfg_['K'], self.cfg_['D'],
                iterationsCount=256,
                reprojectionError=4.0,
                confidence=0.99
                )
        # inversion?
        #R = cv2.Rodrigues(rvec)[0]
        #t = tvec
        #R, t = vm.Rti(R, t)
        #rvec = cv2.Rodrigues(R)[0]
        #tvec = t
        pt0r = cvu.project_points(cld[mi0],
                rvec, tvec,
                self.cfg_['K'], self.cfg_['D'])

        #viz  = draw_matches(img1, img1,
        #        pt0[mi0], feat1.pt[mi1])
        viz  = draw_matches(img1, img1,
                pt0r, feat1.pt[mi1])
        #print pt0[mi0] - feat1.pt[mi1]
        #viz  = draw_matches(img1, img1,
        #        pt0[mi0], feat1.pt[mi1])
        cv2.imshow('pnp', viz)
              
        # obtained position!
        print('rvec?', rvec)
        R   = cv2.Rodrigues(rvec)[0]
        t   = np.float32(tvec)
        #R, t = vm.Rti(R, t)
        rxn = np.float32( tx.euler_from_matrix(R) )
        txn = np.float32( t )

        if True:
            # motion_update()
            self.kf_.x = self.db_.frame[-2]['pose']
            self.kf_.P = self.db_.frame[-2]['cov']
            self.kf_.predict(dt)
            self.kf_.update( np.concatenate([txn.ravel(), rxn.ravel()]) )
            print 'pre', frame1['pose']
            frame1['pose'] = self.kf_.x
            print 'post', frame1['pose']
            frame1['cov']  = self.kf_.P

        print 'rvec', rxn

        #self.kf_.update(

        #t_pt, t_idx = self.tracker_.track(img0, img1, self.db_.state_['track'])
        #self.db_.state_['track'] = t_pt[t_idx]

    def process(self, img, dt, data={}):
        if self.state_ == PipelineState.IDLE:
            return
        if self.state_ == PipelineState.NEED_REF:
            return self.init_ref(img, dt, data)
        elif self.state_ == PipelineState.NEED_MAP:
            return self.init_map(img, dt, data)
        elif self.state_ == PipelineState.TRACK:
            return self.track(img, dt, data)
            
def main():
    #src = './scan_20190212-233625.h264'
    #reader = CVCameraReader(src)
    root = '/media/ssd/datasets/ADVIO'
    reader = AdvioReader(root)

    # update configuration based on input
    cfg = dict(CFG)
    cfg['K'] = reader.meta_['K']
    cfg.update(reader.meta_)

    pl  = Pipeline(cfg=cfg)
    cv2.namedWindow('viz', cv2.WINDOW_NORMAL)
    prv = 0.0
    while True:
        suc, idx, stamp, img = reader.read()
        dt  = (stamp - prv)
        prv = stamp
        cv2.imshow('img', img)
        if(idx <= 1000):
            continue
        if not suc: break
        img = cv2.resize(img, None, fx=CFG['scale'], fy=CFG['scale'])
        data = {}
        pl.process(img, dt, data)
        if 'viz' in data:
            cv2.imshow('viz', data['viz'])
        if pl.state_ != PipelineState.TRACK:
            k = cv2.waitKey(1)
        else:
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
