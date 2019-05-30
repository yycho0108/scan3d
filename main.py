#!/usr/bin/env python2

import numpy as np
import cv2
import pptk
import time

from db import DB, Feature
from track import Tracker
from match import Matcher, match_local
from kalman.ekf import build_ekf
from cho_util.viz import draw_matches, draw_points, print_ratio
from cho_util import vmath as vm
from cho_util.viz.mpl import set_axes_equal
import cv_util as cvu
from tf import transformations as tx
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from twoview import TwoView
from reader.advio import AdvioReader
from dense_rec import DenseRec

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
        scale = 0.5,
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

def project_to_frame(cloud, frame, K, D):
    rmat = tx.euler_matrix(*frame['pose'][9:12])[:3,:3]
    R, t = rmat, frame['pose'][0:3]
    R, t = vm.Rti(R, t)
    cloud = np.float64(cloud)
    rvec = cv2.Rodrigues(R)[0].ravel()
    tvec = np.float32(t).ravel()
    res = cvu.project_points(cloud, rvec, tvec,
            K, D)
    return res

def extract_color(img, pt):
    h, w = img.shape[:2]
    idx = vm.rint(pt)[..., ::-1] # (x,y) -> (i,j)
    idx = np.clip(idx, (0,0), (h-1,w-1)) # TODO : fix this hack
    col = img[idx[:,0], idx[:,1]]
    # TODO : support sampling window (NxN) average
    return col

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

        # apply scale
        w = int( cfg['scale'] * cfg['w'] )
        h = int( cfg['scale'] * cfg['h'] )
        K0 = cfg['K']
        K = cfg['scale'] * cfg['K']
        K[2,2] = 1.0

        # image shape
        shape = (h, w, 3) # TODO : handle monochrome

        # first, make a copy from argument
        cfg = dict(cfg)

        # insert derived values
        cfg['w']     = w
        cfg['h']     = h
        cfg['shape'] = shape
        cfg['K0']    = K0
        cfg['K']     = K

        ## create dynamic type
        #ks = cfg.keys()
        #cfg_t = namedtuple('PipelineConfig', ks)
        ## setup dot-referenced aliases
        #for k, v in cfg.iteritems():
        #    setattr(cfg, k, v)
        return cfg

    def build_db(self):
        cfg = self.cfg_
        ex       = self.extractor_
        img_fmt  = (cfg['shape'], np.uint8)
        dsc_t    = (np.uint8 if ex.descriptorType() == cv2.CV_8U else np.float32)
        dsc_fmt  = (self.extractor_.descriptorSize(), dsc_t)
        return DB(img_fmt=img_fmt, dsc_fmt=dsc_fmt)

    def motion_model(self, f0, f1, use_kf=False, dt=None):
        if not use_kf:
            # simple `repetition` model
            txn0, rxn0 = f0['pose'][0:3], f0['pose'][9:12]
            txn1, rxn1 = f1['pose'][0:3], f1['pose'][9:12]
            R0 = tx.euler_matrix(*rxn0)
            R1 = tx.euler_matrix(*rxn1)

            T0 = tx.compose_matrix(angles=rxn0, translate=txn0)
            T1 = tx.compose_matrix(angles=rxn1, translate=txn1)

            Tv = np.dot(T1, vm.inv(T0))# Tv * T0 = T1
            T2 = np.dot(Tv, T1)

            txn = tx.translation_from_matrix(T2)
            rxn = tx.euler_from_matrix(T2)

            x = f1['pose'].copy()
            P = f1['cov'].copy()
            x[0:3] = txn
            x[9:12] = rxn
            return x, P
        else:
            # dt MUST NOT BE None
            self.kf_.x = prv['pose']
            self.kf_.P = prv['cov']
            self.kf_.predict(dt)
            return self.kf_.x.copy(), self.kf_.P.copy()

    def is_keyframe(self, frame):
        feat = (frame['feat']).item()
        return len(feat.kpt) > 100

    def build_frame(self, img, dt=None):
        """ build a simple frame """
        # automatic index assignment
        index = self.db_.frame.size

        # by default, not a keyframe
        is_kf = False

        # extract features
        kpt, dsc = self.extractor_.detectAndCompute(img, None)
        feat = Feature(kpt, dsc, cv2.KeyPoint.convert(kpt))

        # apply motion model? initialize pose anyway
        if self.db_.frame_.size >= 2:
            x, P = self.motion_model(
                    f0 = self.db_.frame_[-2],
                    f1 = self.db_.frame_[-1],
                    use_kf = False, dt=dt)
        else:
            x = np.zeros(self.cfg_['state_size'])
            P = 1e-6 * np.eye(self.cfg_['state_size'])

        frame = (index, img, x, P, is_kf, feat)
        res = np.array(frame, dtype=self.db_.frame.dtype)
        return res

    def transition(self, new_state):
        print('[state] ({} -> {})'.format(
            self.state_, new_state))
        self.state_ = new_state

    def init_ref(self, img, dt, data):
        """ initialize reference """
        # TODO : handle failure to initialize reference frame
        # == possibly filter for richness of tracking features?
        frame = self.build_frame(img)
        if not self.is_keyframe(frame):
            # if not keyframe-worthy, continue trying initialization
            return
        frame['is_kf'] = True
        self.db_.frame.append( frame )
        self.db_.state_['track'] = (frame['feat']).item().pt
        self.transition( PipelineState.NEED_MAP )

    def init_map(self, img, dt, data):
        """ initialize map """
        # fetch prv+cur frames
        # populate frame from motion model
        frame0 = self.db_.keyframe[-1] # last **keyframe**
        #frame0 = self.db_.frame[-2]
        # TODO : restore dt maybe
        frame1 = self.build_frame(img)
        #frame1 = self.db_.frame[-1]

        #print('target pair : {}-{}'.format(
        #    frame0['index'],
        #    frame1['index']))

        # process ...
        img0, img1   = frame0['image'], frame1['image']
        feat0, feat1 = frame0['feat'],  frame1['feat'].item()

        # match
        mi0, mi1 = self.matcher_.match(feat0.dsc, feat1.dsc,
                lowe=0.8, fold=False) # harsh lowe to avoid pattern collision
        pt0m = feat0.pt[mi0]
        pt1m = feat1.pt[mi1]
        
        suc, det = TwoView(pt0m, pt1m, self.cfg_['K']).compute(data=data)
        print(data['dbg-tv'])
        if not suc:
            # unsuccessful frame-to-frame reconstruction
            if not det['num_pts']:
                # condition: `tracking lost`
                # BEFORE sufficient parallax was observed.
                # need to reset the reference frame.
                #print('\t -- reset keyframe')
                # reset keyframe
                #frame1['is_kf'] = True
                self.transition( PipelineState.NEED_REF )
                pass
            return
        
        # here, init successful
        cld_msk = data['msk_cld']
        cld1 = data['cld1'][cld_msk]
        # IMPORTANT : everything references **frame1** !!
        # (so frame0 geometric information is effectively ignored.)

        col = extract_color(img1, pt1m[cld_msk])
        lmk_idx0 = self.db_.landmark.size
        local_map = dict(
                index   = lmk_idx0 + np.arange(len(cld1)), # landmark index
                src     = np.full(len(cld1), frame1['index']), # source index
                dsc     = feat1.dsc[mi1][cld_msk], # landmark descriptor
                pos     = cld1, # landmark position [[ map frame ]]
                pt      = pt1m[cld_msk], # tracking point initialization
                tri     = np.ones(len(cld1), dtype=np.bool),
                col     = col, # debug : point color information
                track   = np.ones(len(cld1), dtype=np.bool) # trackin status
                )
        self.db_.landmark.extend(zip(*[
            local_map[k] for k in self.db_.landmark.dtype.names]
            ))

        # unroll reconstruct information
        R   = data['R']
        t   = data['t']
        msk = data['msk_cld']

        #print( 'Rpriori', tx.euler_from_matrix(R))
        #R, t = vm.Rti(R, t)
        frame1['pose'][0:3]  = t.ravel()
        frame1['pose'][9:12] = tx.euler_from_matrix(R)
        #x, P = self.motion_model(
        #            f0 = self.db_.frame_[-2],
        #            f1 = frame1,
        #            use_kf = False, dt=dt)
        self.nxt_ = frame1['pose'].copy(), frame1['cov'].copy() # TODO : get rid of this hack

        # update frame data
        #[0pos,3vel,6acc,9rpos,12rvel]
        frame1['pose'][0  : 3]  = 0.0 # t.reshape(-1)
        frame1['pose'][3  : 6]  = 0.0 # reserve judgement about vel here.
        frame1['pose'][6  : 9]  = 0.0 # reserve judgement about acc here.
        frame1['pose'][9  : 12] = 0.0 # tx.euler_from_matrix(R) # TODO : apply proper R?
        frame1['pose'][12 : 15] = 0.0 # reserve judgement about rvel here.
        frame1['is_kf'] = True # successful initialization, frame1 considered keyframe
        self.db_.frame.append( frame1 )

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
            print('R', tx.euler_from_matrix(R))
            print('frame pair : {}-{}'.format(
                frame0['index'],
                frame1['index']))

            viz0 = cv2.drawKeypoints(img0, feat0.kpt, None)
            viz1 = cv2.drawKeypoints(img1, feat1.kpt, None)
            viz  = draw_matches(viz0, viz1, pt0m[msk], pt1m[msk])
            cv2.imshow('viz', viz)
            data['viz'] = viz

            # == if cfg['dbg-cloud']:
            dr_data = {}
            print(self.cfg_['K'])
            cld_viz, col_viz = DenseRec(self.cfg_['K']).compute(
                    img0, img1,
                    P1=data['P0'],
                    P2=data['P1'],
                    data=dr_data)
            cdist = vm.norm(cld_viz)
            data['cld_viz'] = cld_viz[cdist < np.percentile(cdist, 95)]
            data['col_viz'] = col_viz[cdist < np.percentile(cdist, 95)]
            # cv2.imshow('flow', dr_data['viz'])


    def track(self, img, dt, data={}):
        """ Track landmarks"""
        # unroll data
        # fetch frame pair
        frame0 = self.db_.keyframe[-1] # last **keyframe**
        frame1 = self.build_frame(img, dt=dt)
        landmark = self.db_.landmark

        img0  , img1  = frame0['image'] , frame1['image']
        feat0 , feat1 = frame0['feat']  , frame1['feat'].item()

        if self.nxt_ is not None:
            x, P = self.nxt_
            print('x', x)
            frame1['pose'] = x
            frame1['cov'] = P
            self.nxt_ = None

        # bypass match_local if already tracking ...
        pt0_l = landmark['pt'][landmark['track']]
        pt1_l, msk_t = self.tracker_.track(
                img0, img1, pt0_l, return_msk=True)

        # apply tracking mask
        pt0_l = pt0_l[msk_t]
        pt1_l = pt1_l[msk_t]

        # update tracking status
        landmark['track'][landmark['track'].nonzero()[0][~msk_t]] = False
        landmark['pt'][landmark['track']] = pt1_l

        # search additional points
        cld0_l = landmark['pos'][~landmark['track']]
        dsc_l  = landmark['dsc'][~landmark['track']]

        #if len(cld0_l) >= 16:
        #    pt0_cld_l = project_to_frame(cld0_l, frame1,
        #            self.cfg_['K'], self.cfg_['D'])
        #    mi0, mi1 = match_local(
        #            pt0_cld_l, feat1.pt,
        #            dsc_l, feat1.dsc
        #            )

        #    # collect all paris
        #    pt0  = np.concatenate([pt0_l, pt0_cld_l[mi0]], axis=0)
        #    pt1  = np.concatenate([pt1_l, feat1.pt[mi1]], axis=0)
        #    cld0 = np.concatenate([
        #        landmark['pos'][landmark['track']],
        #        landmark['pos'][~landmark['track']][mi0]
        #        ], axis=0)
        #else:
        pt0  = pt0_l
        pt1  = pt1_l
        cld0 = landmark['pos'][landmark['track']]

        # debug ... 
        pt_dbg = project_to_frame(
                landmark['pos'][landmark['track']],
                frame1,
                self.cfg_['K'], self.cfg_['D'])
        #img_dbg = draw_points(img1.copy(), pt_dbg, color=(255,0,0) )
        #draw_points(img_dbg, pt1, color=(0,0,255) )
        img_dbg = draw_matches(img1, img1, pt_dbg, pt1)
                
        cv2.imshow('dbg', img_dbg)

        print_ratio(len(pt0_l), len(pt0), name='point source')
        print('dbg')

        #if len(mi0) <= 0:
        #    viz1 = draw_points(img1.copy(), pt0)
        #    viz2 = draw_points(img1.copy(), feat1.pt)
        #    viz = np.concatenate([viz1, viz2], axis=1)
        #    cv2.imshow('pnp', viz)
        #    return False

        #print_ratio(len(mi0), len(pt0))
        #suc, rvec, tvec = cv2.solvePnP(
        #        cld0[:, None], pt1[:, None],
        #        self.cfg_['K'], self.cfg_['D'],
        #        flags = cv2.SOLVEPNP_EPNP
        #        ) # T(rv,tv) . cld = cam
        #inl = None
        #print 'euler', tx.euler_from_matrix(cv2.Rodrigues(rvec)[0])
        _, rvec, tvec, inl = cv2.solvePnPRansac(
                cld0[:,None], pt1[:,None],
                self.cfg_['K'], self.cfg_['D'],
                useExtrinsicGuess=False,
                iterationsCount=65535,
                reprojectionError=1.0,
                confidence=0.9999,
                flags=cv2.SOLVEPNP_EPNP
                #minInliersCount=0.5*_['pt0']
                )
        suc = (inl is not None)
        #suc, rvec, tvec, inl = cv2.solvePnPRansac(
        #        cld0, pt1,
        #        self.cfg_['K'], self.cfg_['D'],
        #        #iterationsCount=4096,
        #        reprojectionError=0.1,
        #        #confidence=0.99
        #        )
        print('pnp success : {}'.format(suc))

        if inl is not None:
            # print 'inl', inl
            print_ratio(len(inl), len(cld0), name='pnp')
        
        # inversion?
        #R = cv2.Rodrigues(rvec)[0]
        #t = tvec
        #R, t = vm.Rti(R, t)
        #rvec = cv2.Rodrigues(R)[0]
        #tvec = t
        #pt0r = cvu.project_points(cld[mi0],
        #        rvec, tvec,
        #        self.cfg_['K'], self.cfg_['D'])

        viz  = draw_matches(img1, img1,
                pt0, pt1)
        #viz  = draw_matches(
        #        frame0['image'], img1,
        #        self.db_.landmark[idx][mi0]['pt'], feat1.pt[mi1])
        #print pt0[mi0] - feat1.pt[mi1]
        #viz  = draw_matches(img1, img1,
        #        pt0[mi0], feat1.pt[mi1])
        cv2.imshow('pnp', viz)
              
        # obtained position!
        R   = cv2.Rodrigues(rvec)[0]
        t   = np.float32(tvec)
        #R, t = vm.Rti(R, t)

        if suc:
            frame1['pose'][0:3] = t.ravel()
            frame1['pose'][9:12] = tx.euler_from_matrix(R)
            self.db_.frame.append( frame1 )

        if False:
            # motion_update()
            self.kf_.x = self.db_.frame[-2]['pose']
            self.kf_.P = self.db_.frame[-2]['cov']
            self.kf_.predict(dt)
            self.kf_.update( np.concatenate([txn.ravel(), rxn.ravel()]) )
            print 'pre', frame1['pose'][0:3], frame1['pose'][9:12]
            frame1['pose'] = self.kf_.x
            print 'post', frame1['pose'][0:3], frame1['pose'][9:12]
            frame1['cov']  = self.kf_.P


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
    reader = AdvioReader(root, idx=3)

    # update configuration based on input
    cfg = dict(CFG)
    cfg['K'] = reader.meta_['K']
    cfg.update(reader.meta_)

    reader.set_pos(750)

    pl  = Pipeline(cfg=cfg)
    cv2.namedWindow('viz', cv2.WINDOW_NORMAL)
    prv = 0.0
    while True:
        suc, idx, stamp, img = reader.read()
        dt  = (stamp - prv)
        prv = stamp
        #cv2.imshow('img', img)
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

        if ('col_viz' in data) and ('cld_viz' in data):
            cld = data['cld_viz']
            col = data['col_viz']
            # optical -> base 
            cld = vm.tx3( vm.tx.euler_matrix(-np.pi/2, 0, -np.pi/2), cld)
            ax = plt.gca(projection='3d')
            ax.scatter(cld[:,0], cld[:,1], cld[:,2],
                    c = (col[...,::-1] / 255.0))

            ax.view_init(elev=0,azim=180)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            #set_axes_equal(ax)

            plt.show()
            pass

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
