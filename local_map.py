#!/usr/bin/env python2

import numpy as np
from db import DB
from twoview import TwoView
from util import *

class MapInitializer(object):
    def __init__(self, db, matcher, tracker, cfg):
        # cache handles
        self.db_ = db
        self.matcher_ = matcher
        self.tracker_ = tracker
        self.cfg_ = cfg

    def reset(self):
        # throw away all data in db
        self.db_.reset()

    def init_ref(self, frame):
        # append frame
        frame['is_kf'] = True # move this after append() if input mod is undesirable
        self.db_.frame.append(frame)

        feat  = frame['feat']
        n_pt  = len(feat.pt)
        col = extract_color(frame['image'], feat.pt)

        entry = dict(
                index   = np.arange(n_pt), # landmark index
                src     = np.full(n_pt, frame['index']), # source index
                dsc     = feat.dsc, # landmark descriptor
                rsp     = [k.response for k in feat.kpt],
                pos     = np.zeros(shape=(n_pt,3), dtype=np.float32), # landmark position [[ map frame ]]
                pt      = feat.pt, # tracking point initialization
                tri     = np.zeros(len(cld1), dtype=np.bool), # **NOT** triangulated
                col     = col, # debug : point color information
                track   = np.ones(len(cld1), dtype=np.bool) # tracking status
                )
        # TODO : something more efficient than zip(*[]) ... ?
        self.db_.landmark.extend(zip(*[
            entry[k] for k in self.db_.landmark.dtype.names]
            ))

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
        landmark['pt'][landmark['track']] = pt1
        lmk_idx = landmark['index'][landmark['track']]

        # update observation
        self.db_.observation.extend(zip(*[
                    np.full_like(lmk_idx, frame['index']), # observation frame source
                    lmk_idx, # landmark index
                    pt1 # observed point location
                    ]))

        # report tracking status
        # TODO : maybe skip redundant compute upon failure
        return (len(lmk_idx) > 128)

    def optimize(self):
        pass

    def compute(self, frame, data={}):
        if self.db_.size <= 0:
            # initialize reference keyframe
            if self.is_keyframe(frame):
                # progress stage -> search matching keyframe
                self.init_ref(frame)
            # False==map initialization is not complete
            return False

        ref_frame = self.db_.keyframe[0]
        landmark  = self.db_.landmark

        if not self.track():
            # tracking (probably) lost;
            # re-initialization required
            self.reset()
            return False

        # match (ref<->cur)
        mi0, mi1 = self.matcher_.match(
                self.ref_.dsc, frame.dsc,
                lowe=0.8, fold=False
                ) # harsh lowe to avoid pattern collision
        pt0m = feat0.pt[mi0]
        pt1m = feat1.pt[mi1]

        # attempt triangulation and map generation
        data_tv = {}
        suc, det = TwoView(pt0m, pt1m, self.cfg_['K']).compute(data=data_tv)
        if not suc:
            return False

        # optimize map + finalize
        self.optimize()

        return True
