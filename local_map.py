#!/usr/bin/env python2

import numpy as np
from db import DB
from twoview import TwoView
from util import *
from cho_util.viz import draw_matches, draw_points, print_ratio
from optimize.ba import BundleAdjustment

class MapInitializer(object):
    def __init__(self, db, matcher, tracker, cfg):
        # cache handles
        self.db_ = db
        self.matcher_ = matcher
        self.tracker_ = tracker
        self.cfg_ = cfg
        
    def is_keyframe(self, frame):
        # TODO : more robust keyframe heuristic
        # == possibly filter for richness of tracking features?
        feat = (frame['feat']).item()
        return len(feat.kpt) > 100 # TODO: arbitrary threshold

    def reset(self):
        # throw away all data in db
        self.db_.reset()

    def init_ref(self, frame):
        # append frame
        frame['index'] = 0
        frame['is_kf'] = True # move this after append() if input mod is undesirable
        self.db_.frame.append(frame)

        feat  = frame['feat'].item()
        n_pt  = len(feat.pt)
        col = extract_color(frame['image'], feat.pt)

        # make a pos guess
        fx,fy,cx,cy = self.cfg_['K'][(0,1,0,1),(0,1,2,2)]

        lmk_x = (feat.pt[:,0] - cx) / fx
        lmk_y = (feat.pt[:,1] - cy) / fy
        lmk_z = np.ones_like(lmk_x)
        pos   = np.stack([lmk_x, lmk_y, lmk_z], axis=-1)

        entry = dict(
                index   = np.arange(n_pt), # landmark index
                src     = np.full(n_pt, frame['index']), # source index
                dsc     = feat.dsc, # landmark descriptor
                rsp     = [k.response for k in feat.kpt],
                pt0      = feat.pt, # tracking point initialization
                invd    = (1.0 / lmk_z),
                depth   = lmk_z,
                pos     = pos, # landmark position [[ map frame ]]
                track   = np.ones(n_pt, dtype=np.bool), # tracking status
                pt      = feat.pt, # tracking point initialization
                tri     = np.ones(n_pt, dtype=np.bool), # **NOT** triangulated?
                col     = col, # debug : point color information
                )
        # TODO : something more efficient than zip(*[]) ... ?
        self.db_.landmark.extend(zip(*[
            entry[k] for k in self.db_.landmark.dtype.names]
            ))
        self.db_.observation.extend(zip(*[
                    np.full(n_pt, frame['index']), # observation frame source
                    np.arange(n_pt), # landmark index
                    feat.pt # observed point location
                    ]))

    def track(self, frame):
        # unroll data
        landmark    = self.db_.landmark
        observation = self.db_.observation
        frame0 = self.db_.frame[-1]
        frame1 = frame
        img0 = frame0['image']
        img1 = frame1['image']

        # track
        pt0 = landmark['pt'][landmark['track']]
        pt1, msk_t = self.tracker_.track(
                img0, img1, pt0, return_msk=True)

        # update landmark data
        idx_lost = landmark['track'].nonzero()[0][~msk_t]
        landmark['track'][idx_lost] = False
        landmark['pt'][landmark['track']] = pt1[msk_t]
        lmk_idx = landmark['index'][landmark['track']]

        # update observation
        self.db_.observation.extend(zip(*[
                    np.full_like(lmk_idx, frame['index']), # observation frame source
                    lmk_idx, # landmark index
                    pt1 # observed point location
                    ]))

        # report tracking status
        # TODO : maybe skip redundant compute upon failure

        # data['viz'] = draw_matches(img0, img1, pt0m, pt1m)

        return (len(lmk_idx) > 128)

    def bundle_adjust(self, data={}):
        obs = self.db_.observation
        i_src = obs['src_idx']
        i_lmk = obs['lmk_idx']
        p_obs = obs['point']

        frames = self.db_.frame
        txn = frames['pose'][:, L_POS]
        rxn = frames['pose'][:, A_POS]

        lmk = self.db_.landmark['pos']

        #print ('fini')
        #print np.all(np.isfinite(txn))
        #print np.all(np.isfinite(rxn))
        #print np.all(np.isfinite(lmk))
        #print ('lmk', lmk)

        data_ba = {}
        suc = BundleAdjustment(
                i_src, i_lmk, p_obs,
                txn, rxn, lmk, self.cfg_['K'],
                axa=True).compute(data = data_ba)

        if suc:
            txn = data_ba['txn']
            rxn = data_ba['rxn']
            lmk = data_ba['lmk']
            self.db_.frame['pose'][:, L_POS] = txn
            self.db_.frame['pose'][:, A_POS] = rxn
            self.db_.landmark['pos'] = lmk

        # return updated data
        # TODO: ensure keys do not collide
        # data.update( data_ba )

    def optimize(self):
        # TODO: run bundle adjustment here ...

        # 1. get rid of `useless` landmarks
        lmk_idx = np.nonzero(self.db_.landmark['track'])[0]
        #lmk_idx = np.nonzero(self.db_.landmark['tri'])[0]
        self.db_.prune(lmk_idx=lmk_idx)

        # 2. run BA
        data_ba = {}
        self.bundle_adjust(data=data_ba)

        # 3. Finalize outputs
        src_idx = np.int32([self.db_.frame[0]['index'], self.db_.frame[-1]['index']])
        self.db_.prune(src_idx=src_idx)

    def compute(self, frame, data={}):
        if self.db_.frame.size <= 0:
            # initialize reference keyframe
            if self.is_keyframe(frame):
                # progress stage -> search matching keyframe
                self.init_ref(frame)
            # False==map initialization is not complete
            return False

        frame['index'] = self.db_.frame[-1]['index'] + 1
        if not self.track(frame):
            # tracking (probably) lost;
            # re-initialization required
            self.reset()
            return False

        self.db_.frame.append(frame)
        if not self.is_keyframe(frame):
            # don't bother trying to triangulate
            return False

        ref_frame = self.db_.keyframe[0]
        cur_frame = self.db_.frame[-1]

        # match (ref<->cur)
        feat0 = ref_frame['feat']
        feat1 = cur_frame['feat']

        mi0, mi1 = self.matcher_.match(
                feat0.dsc, feat1.dsc,
                lowe=0.8, fold=False
                ) # harsh lowe to avoid pattern collision
        data['mi0'] = mi0
        data['mi1'] = mi1
        pt0m = feat0.pt[mi0]
        pt1m = feat1.pt[mi1]

        # attempt triangulation and map generation
        suc, det = TwoView(pt0m, pt1m, self.cfg_['K']).compute(data=data)
        if not suc:
            return False
        cur_frame['is_kf'] = True

        # assign results
        msk_cld = data['msk_cld']
        idx_cld = msk_cld.nonzero()[0]
        cld0 = data['cld0'][idx_cld]
        self.db_.landmark['pos'][mi0[idx_cld]] = cld0
        self.db_.landmark['tri'][mi0[idx_cld]] = True

        # assign frame-data results
        R   = data['R']
        t   = data['t']
        cur_frame['pose'][L_POS] = t.ravel()
        cur_frame['pose'][A_POS] = tx.euler_from_matrix(R)

        #print 'hmmm.................'
        #print ref_frame['index']
        #print cur_frame['index']
        #print cur_frame['pose']

        # cld0 <==> pt1m[idx_cld]
        #dbg = draw_matches(
        #        cur_frame['image'], cur_frame['image'],
        #        project_to_frame(cld0, source_frame=ref_frame, target_frame=cur_frame,
        #        K=self.cfg_['K'], D=self.cfg_['D']),
        #        #pt1m[idx_cld]
        #        self.db_.landmark['pt'][mi0[idx_cld]]
        #        )
        #cv2.imshow('dbg-pnp', dbg)
        #cv2.waitKey(0)

        # print 'validation ...'
        # print len(self.db_.landmark[idx_cld])
        # print len(cld0)
        # print cld0[:5]
        # print self.db_.landmark['pos'][mi0[idx_cld]][:5]

        # optimize map + finalize
        self.optimize()

        return True
