#!/usr/bin/env python2

import numpy as np
import cv2
import pptk
import time
import os

from db import DB, Feature
from db import L_POS, L_VEL, L_ACC, A_POS, A_VEL
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
from reader import AdvioReader, CVCameraReader
from dense_rec import DenseRec

from optimize.ba import BundleAdjustment

from profilehooks import profile

from util import *

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
            ),
        kalman = True
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
        self.extractor_ = cv2.ORB_create(2048, edgeThreshold=19)
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

    def motion_model(self, f0, f1, use_kalman=False):
        if not use_kalman:
            # simple `repetition` model
            txn0, rxn0 = f0['pose'][L_POS], f0['pose'][A_POS]
            txn1, rxn1 = f1['pose'][L_POS], f1['pose'][A_POS]
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
            self.kf_.x = f0['pose']
            self.kf_.P = f0['cov']
            dt = (f1['stamp'] - f0['stamp'])
            self.kf_.predict(dt)
            return self.kf_.x.copy(), self.kf_.P.copy()

    def is_keyframe(self, frame):
        # TODO : more robust keyframe heuristic
        # == possibly filter for richness of tracking features?
        feat = (frame['feat']).item()
        return len(feat.kpt) > 100 # TODO: arbitrary threshold

    def build_frame(self, img, stamp):
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
                    use_kalman = True)
        else:
            x = np.zeros(self.cfg_['state_size'])
            P = 1e-6 * np.eye(self.cfg_['state_size'])

        frame = (index, stamp, img, x, P, is_kf, feat)
        res = np.array(frame, dtype=self.db_.frame.dtype)
        return res

    def transition(self, new_state):
        print('[state] ({} -> {})'.format(
            self.state_, new_state))
        self.state_ = new_state

    def init_ref(self, img, stamp, data):
        """ initialize reference """
        frame = self.build_frame(img, stamp)
        if not self.is_keyframe(frame):
            # if not keyframe-worthy, continue trying initialization
            return
        frame['is_kf'] = True
        self.db_.frame.append( frame )
        self.db_.state_['track'] = (frame['feat']).item().pt
        self.transition( PipelineState.NEED_MAP )

    def init_map(self, img, stamp, data):
        """ initialize map """
        # fetch prv+cur frames
        # populate frame from motion model
        frame0 = self.db_.keyframe[-1] # last **keyframe**
        #frame0 = self.db_.frame[-2]
        # TODO : restore dt maybe
        frame1 = self.build_frame(img, stamp)
        #frame1 = self.db_.frame[-1]

        print('target pair : {:.2f}-{:.2f}'.format(
            frame0['stamp'],
            frame1['stamp']))

        # process ...
        img0, img1   = frame0['image'], frame1['image']
        feat0, feat1 = frame0['feat'],  frame1['feat'].item()

        # bookkeeping: track reference points
        # if tracking is `completely` lost then new keyframe is desired.
        pt1_l, msk_t = self.tracker_.track(
                img0, img1, self.db_.state['track'], return_msk=True)
        self.db_.state['track'] = pt1_l[msk_t]
        print('tracking state : {}'.format( len(self.db_.state['track'])))

        # match
        mi0, mi1 = self.matcher_.match(feat0.dsc, feat1.dsc,
                lowe=0.6, fold=False) # harsh lowe to avoid pattern collision
        pt0m = feat0.pt[mi0]
        pt1m = feat1.pt[mi1]
        
        suc, det = TwoView(pt0m, pt1m, self.cfg_['K']).compute(data=data)
        print(data['dbg-tv'])
        if not suc:
            data['viz'] = draw_matches(img0, img1, pt0m, pt1m)
            # unsuccessful frame-to-frame reconstruction
            if not len(self.db_.state['track']) >= 128:
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
        msk_cld = data['msk_cld']
        cld1 = data['cld1'][msk_cld]
        # IMPORTANT : everything references **frame1** !!
        # (so frame0 geometric information is effectively ignored.)

        col = extract_color(img1, pt1m[msk_cld])
        lmk_idx0 = self.db_.landmark.size
        lmk_idx  = lmk_idx0 + np.arange(len(cld1))
        local_map = dict(
                index   = lmk_idx, # landmark index
                src     = np.full(len(cld1), frame1['index']), # source index
                dsc     = feat1.dsc[mi1][msk_cld], # landmark descriptor
                rsp     = [feat1.kpt[i].response for i in np.arange(len(feat1.kpt))[mi1][msk_cld]],
                pos     = cld1, # landmark position [[ map frame ]]
                pt      = pt1m[msk_cld], # tracking point initialization
                tri     = np.ones(len(cld1), dtype=np.bool),
                col     = col, # debug : point color information
                track   = np.ones(len(cld1), dtype=np.bool) # tracking status
                )
        self.db_.landmark.extend(zip(*[
            local_map[k] for k in self.db_.landmark.dtype.names]
            ))
        self.db_.observation.extend(zip(*[
                    np.full_like(lmk_idx, frame1['index']), # observation frame source
                    lmk_idx, # landmark index
                    pt1m[msk_cld]
                    ]))

        # unroll reconstruct information
        R   = data['R']
        t   = data['t']
        msk = data['msk_cld']

        #print( 'Rpriori', tx.euler_from_matrix(R))
        #R, t = vm.Rti(R, t)
        frame1['pose'][L_POS] = t.ravel()
        frame1['pose'][A_POS] = tx.euler_from_matrix(R)
        #x, P = self.motion_model(
        #            f0 = self.db_.frame_[-2],
        #            f1 = frame1,
        #            use_kf = False, dt=dt)
        self.nxt_ = frame1['pose'].copy(), frame1['cov'].copy() # TODO : get rid of this hack

        # TODO : danger : if init_map() is NOT created from zero position, need to account for that
        # update frame data
        #[0pos,3vel,6acc,9rpos,12rvel]
        frame1['pose'][L_POS] = 0.0 # t.reshape(-1)
        frame1['pose'][L_VEL] = 0.0 # reserve judgement about vel here.
        frame1['pose'][L_ACC] = 0.0 # reserve judgement about acc here.
        frame1['pose'][A_POS] = 0.0 # tx.euler_from_matrix(R) # TODO : apply proper R?
        frame1['pose'][A_VEL] = 0.0 # reserve judgement about rvel here.
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

    @profile(sort='cumtime')
    def track(self, img, stamp, data={}):
        """ Track landmarks"""
        # unroll data
        # fetch frame pair
        # TODO : add landmarks along the way
        # TODO : update landmarks through optimization
        mapframe = self.db_.keyframe[0] # first keyframe = map frame
        keyframe = self.db_.keyframe[-1] # last **keyframe**
        frame0 = self.db_.frame[-1] # last **frame**
        frame1 = self.build_frame(img, stamp)
        # print('prior position',
        #         frame1['pose'][L_POS], frame1['pose'][A_POS])
        landmark = self.db_.landmark

        img1  = frame1['image']
        feat1 = frame1['feat'].item()

        if self.nxt_ is not None:
            # TODO : get rid of this ugly hack
            x, P = self.nxt_
            print('x', x)
            frame1['pose'] = x
            frame1['cov'] = P
            self.nxt_ = None

        # bypass match_local for already tracking points ...
        pt0_l = landmark['pt'][landmark['track']]
        pt1_l, msk_t = self.tracker_.track(
                frame0['image'], img1, pt0_l, return_msk=True)

        # apply tracking mask
        pt0_l = pt0_l[msk_t]
        pt1_l = pt1_l[msk_t]

        # update tracking status
        landmark['track'][landmark['track'].nonzero()[0][~msk_t]] = False
        landmark['pt'][landmark['track']] = pt1_l

        # search additional points
        cld0_l = landmark['pos'][~landmark['track']]
        dsc_l  = landmark['dsc'][~landmark['track']]

        msk_prj = None
        if len(cld0_l) >= 128:
            # merge with projections
            pt0_cld_l = project_to_frame(cld0_l,
                    source_frame=mapframe,
                    target_frame=frame1,
                    K=self.cfg_['K'],
                    D=self.cfg_['D'])

            # in-frame projection mask
            msk_prj = np.logical_and.reduce([
                0 <= pt0_cld_l[..., 0],
                pt0_cld_l[..., 0] < self.cfg_['w'],
                0 <= pt0_cld_l[..., 1],
                pt0_cld_l[..., 1] < self.cfg_['h'],
                ])

        if (msk_prj is not None) and msk_prj.sum() >= 16:
            mi0, mi1 = match_local(
                    pt0_cld_l[msk_prj], feat1.pt,
                    dsc_l[msk_prj], feat1.dsc,
                    hamming = (not feat1.dsc.dtype == np.float32)
                    )

            # collect all parts
            pt0  = np.concatenate([pt0_l, pt0_cld_l[msk_prj][mi0]], axis=0)
            pt1  = np.concatenate([pt1_l, feat1.pt[mi1]], axis=0)
            cld0 = np.concatenate([
                landmark['pos'][landmark['track']],
                landmark['pos'][~landmark['track']][msk_prj][mi0]
                ], axis=0)

            obs_lmk_idx = np.concatenate([
                landmark['index'][landmark['track']],
                landmark['index'][~landmark['track']][msk_prj][mi0]
                ], axis=0)
        else:
            # only use tracked points
            pt0  = pt0_l
            pt1  = pt1_l
            cld0 = landmark['pos'][landmark['track']]
            obs_lmk_idx = landmark['index'][landmark['track']]

        

        # debug ... 
        #pt_dbg = project_to_frame(
        #        landmark['pos'][landmark['track']],
        #        frame1,
        #        self.cfg_['K'], self.cfg_['D'])
        ##img_dbg = draw_points(img1.copy(), pt_dbg, color=(255,0,0) )
        ##draw_points(img_dbg, pt1, color=(0,0,255) )
        #img_dbg = draw_matches(img1, img1, pt_dbg, pt1)
        #cv2.imshow('dbg', img_dbg)
        #print_ratio(len(pt0_l), len(pt0), name='point source')

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

        T_i = tx.inverse_matrix(
                tx.compose_matrix(
                    translate=frame1['pose'][L_POS],
                    angles=frame1['pose'][A_POS]
                    ))
        rvec0 = cv2.Rodrigues(T_i[:3,:3])[0]
        tvec0 = T_i[:3,3:]

        if len(pt1) >= 1024:
            # prune
            nmx_idx = non_max(
                    pt1,
                    landmark['rsp'][obs_lmk_idx]
                    )
            print_ratio(len(nmx_idx), len(pt1), name='non_max')
            cld_pnp, pt1_pnp = cld0[nmx_idx], pt1[nmx_idx]
        else:
            cld_pnp, pt1_pnp = cld0, pt1

        suc, rvec, tvec, inl = cv2.solvePnPRansac(
                cld_pnp[:,None], pt1_pnp[:,None],
                self.cfg_['K'], self.cfg_['D'],
                useExtrinsicGuess=True,
                rvec=rvec0,
                tvec=tvec0,
                iterationsCount=1024,
                reprojectionError=1.0,
                confidence=0.999,
                flags=cv2.SOLVEPNP_EPNP
                #flags=cv2.SOLVEPNP_DLS
                #flags=cv2.SOLVEPNP_ITERATIVE
                #minInliersCount=0.5*_['pt0']
                )

        n_pnp_in  = len(cld_pnp)
        n_pnp_out = len(inl) if (inl is not None) else 0

        suc = (suc and (inl is not None) and (n_pnp_out >= 128 or n_pnp_out >= 0.25 * n_pnp_in))
        print('pnp success : {}'.format(suc))
        if inl is not None:
            print_ratio(n_pnp_out, n_pnp_in, name='pnp')

        # visualize match statistics
        viz_pt0 = project_to_frame(cld0,
                source_frame=mapframe,
                target_frame=keyframe, # TODO: keyframe may no longer be true?
                K=self.cfg_['K'],
                D=self.cfg_['D'])
        viz_msk = np.logical_and.reduce([
            0 <= viz_pt0[:,0],
            viz_pt0[:,0] < self.cfg_['w'],
            0 <= viz_pt0[:,1],
            viz_pt0[:,1] < self.cfg_['h'],
            ])
        viz1 = draw_points(img1.copy(), feat1.pt)
        viz  = draw_matches(keyframe['image'], viz1,
                viz_pt0[viz_msk], pt1[viz_msk])
        data['viz'] = viz
              
        # obtained position!
        R   = cv2.Rodrigues(rvec)[0]
        t   = np.float32(tvec)
        R, t = vm.Rti(R, t)
        rxn = np.reshape(tx.euler_from_matrix(R), 3)
        txn = t.ravel()

        if suc:
            # print('pnp-txn', t)
            # print('pnp-rxn', tx.euler_from_matrix(R))
            # motion_update()
            if self.cfg_['kalman']:
                # kalman_update()
                self.kf_.x = frame0['pose']
                self.kf_.P = frame0['cov']
                self.kf_.predict( frame1['stamp'] - frame0['stamp'] )
                self.kf_.update(np.concatenate([txn, rxn])) 
                frame1['pose'] = self.kf_.x
                frame1['cov']  = self.kf_.P
            else:
                # hard_update()
                frame1['pose'][L_POS] = t.ravel()
                frame1['pose'][A_POS] = tx.euler_from_matrix(R)
            self.db_.frame.append( frame1 )

            self.db_.observation.extend(zip(*[
                    np.full_like(obs_lmk_idx, frame1['index']), # observation frame source
                    obs_lmk_idx, # landmark index
                    pt1
                    ]))
        x = 1

        need_kf = np.logical_or.reduce([
            not suc,  # PNP failed -- try new keyframe
            suc and (n_pnp_out < 256), # PNP was decent but would be better to have a new frame
            (frame1['index'] - keyframe['index'] > 32) and (msk_t.sum() < 256) # = frame is somewhat stale
            ]) and self.is_keyframe(frame1)

        run_ba = (frame1['index'] % 8) == 0 # ?? better criteria for running BA?
        #run_ba = False
        #run_ba = need_kf

        if run_ba:
            idx0, idx1 = max(keyframe['index'], frame1['index']-8), frame1['index']
            #idx0, idx1 = keyframe['index'], frame1['index']
            obs = self.db_.observation
            msk = np.logical_and(
                    idx0 <= obs['src_idx'],
                    obs['src_idx'] <= idx1)

            # parse observation
            i_src = obs['src_idx'][msk]
            print('i_src', i_src)
            i_lmk = obs['lmk_idx'][msk]
            p_obs = obs['point'][msk]

            # index pruning relevant sources
            i_src_alt, i_a2s, i_s2a = index_remap(i_src)
            i_lmk_alt, i_a2l, i_l2a = index_remap(i_lmk)

            # 1. select targets based on new index
            i_src     = i_s2a
            i_lmk     = i_l2a
            frames    = self.db_.frame[i_a2s[i_src_alt]]
            landmarks = self.db_.landmark[i_a2l[i_lmk_alt]]

            # parse data
            txn       = frames['pose'][:, L_POS]
            rxn       = frames['pose'][:, A_POS]
            lmk       = landmarks['pos']

            data_ba = {}
            # NOTE : txn/rxn will be internally inverted to reduce duplicate compute.
            suc = BundleAdjustment(
                    i_src, i_lmk, p_obs, # << observation
                    txn, rxn, lmk, self.cfg_['K']).compute(data=data_ba)       # << data

            if suc:
                # TODO : apply post-processing kalman filter?

                #print('{}->{}'.format(txn, data_ba['txn']))
                #print('{}->{}'.format(rxn, data_ba['rxn']))
                txn = data_ba['txn']
                rxn = data_ba['rxn']
                lmk = data_ba['lmk']
                self.db_.frame['pose'][i_a2s[i_src_alt], L_POS] = txn
                self.db_.frame['pose'][i_a2s[i_src_alt], A_POS] = rxn
                self.db_.landmark['pos'][i_a2l[i_lmk_alt]]      = lmk

        if need_kf:
            for index in reversed(range(keyframe['index'], frame1['index'])):
                feat0, feat1 = self.db_.frame[index]['feat'], frame1['feat'].item()
                mi0, mi1 = self.matcher_.match(
                        feat0.dsc, feat1.dsc,
                        lowe=0.8, fold=False)
                data_tv = {}
                suc_tv, det_tv = TwoView(feat0.pt[mi0],
                        feat1.pt[mi1],
                        self.cfg_['K']).compute(data=data_tv)

                if suc_tv:
                    print('======================= NEW KEYFRAME ===')
                    xfm0 = pose_to_xfm(self.db_.frame[index]['pose'])
                    xfm1 = pose_to_xfm(frame1['pose'])
                    scale_ref = np.linalg.norm(
                            tx.translation_from_matrix(vm.Ti(xfm1).dot(xfm0))
                            )
                    scale_tv  = np.linalg.norm(data_tv['t'])
                    # TODO : does NOT consider "duplicate" landmark identities

                    # IMPORTANT: frame1  is a `copy` of "last_frame"
                    #frame1['is_kf'] = True
                    self.db_.frame[-1]['is_kf'] = True

                    lmk_idx0 = self.db_.landmark.size
                    print 'lmk_idx0', lmk_idx0
                    msk_cld = data_tv['msk_cld']
                    cld1 = data_tv['cld1'][msk_cld] * (scale_ref / scale_tv)
                    cld = transform_cloud(cld1,
                            source_frame=frame1,
                            target_frame=mapframe,
                            )
                    col = extract_color(frame1['image'], feat1.pt[mi1][msk_cld])
                    local_map = dict(
                            index   = lmk_idx0 + np.arange(len(cld)), # landmark index
                            src     = np.full(len(cld), frame1['index']), # source index
                            dsc     = feat1.dsc[mi1][msk_cld], # landmark descriptor
                            rsp     = [feat1.kpt[i].response for i in np.arange(len(feat1.kpt))[mi1][msk_cld]], # response "strength"
                            pos     = cld, # landmark position [[ map frame ]]
                            pt      = feat1.pt[mi1][msk_cld], # tracking point initialization
                            tri     = np.ones(len(cld), dtype=np.bool),
                            col     = col, # debug : point color information
                            track   = np.ones(len(cld), dtype=np.bool) # tracking status
                            )
                    # hmm?
                    self.db_.landmark.extend(zip(*[
                        local_map[k] for k in self.db_.landmark.dtype.names]
                        ))
                    break
            else:
                print('Attempted new keyframe but failed')

    def process(self, img, stamp, data={}):
        if self.state_ == PipelineState.IDLE:
            return
        if self.state_ == PipelineState.NEED_REF:
            return self.init_ref(img, stamp, data)
        elif self.state_ == PipelineState.NEED_MAP:
            return self.init_map(img, stamp, data)
        elif self.state_ == PipelineState.TRACK:
            return self.track(img, stamp, data)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        D_ = lambda p : os.path.join(path, p)
        np.save(D_('config.npy'), self.cfg_)
        self.db_.save(path)
            
def main():
    #src = './scan_20190212-233625.h264'
    #reader = CVCameraReader(src, K=CFG['K'])
    root = '/media/ssd/datasets/ADVIO'
    reader = AdvioReader(root, idx=2)
    # update configuration based on input
    cfg = dict(CFG)
    cfg['K'] = reader.meta_['K']
    cfg.update(reader.meta_)

    reader.set_pos(1000)
    auto = True

    pl  = Pipeline(cfg=cfg)
    cv2.namedWindow('viz', cv2.WINDOW_NORMAL)
    while True:
        suc, idx, stamp, img = reader.read()
        #cv2.imshow('img', img)
        if not suc: break
        img = cv2.resize(img, None, fx=CFG['scale'], fy=CFG['scale'])
        data = {}
        pl.process(img, stamp, data)
        #try:
        #    pl.process(img, stamp, data)
        #except Exception as e:
        #    print('Exception while processing: {}'.format(e))
        #    break
        if 'viz' in data:
            cv2.imshow('viz', data['viz'])
        k = cv2.waitKey(1 if auto else 0)
        if k in [27, ord('q')]: break
        if k in [ord(' ')]:
            auto = (not auto)

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

    pl.save('/tmp/db')

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
