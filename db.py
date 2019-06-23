#!/usr/bin/env python2
import numpy as np
import cv2
import os
from record import NDRecord
from collections import namedtuple

Feature = namedtuple('Feature',
                     ['kpt', 'dsc', 'pt'],
                     )

L_POS = np.s_[0:3]
L_VEL = np.s_[3:6]
L_ACC = np.s_[6:9]
A_POS = np.s_[9:12]
A_VEL = np.s_[12:15]

# two of the most useful conversions ...
def lpos_from_pose(pose):
    return pose[..., L_POS]
def apos_from_pose(pose):
    return pose[..., A_POS]

class DB(object):
    """
    Visual SLAM Database.

    DB.frame : NDRecord( [index, image, pose, is_kf] )
        index : frame index
        image : frame image.
        pose  : 15DoF locational state : [pos,vel,acc,rpos,rvel]
        is_kf : whether or not the frame is a keyframe.

    DB.landmark : NDRecord( [index, dsc, pos] )
        index   : landmark index.
        dsc     : landmark (visual) descriptor.
        pos     : landmark position.
        kpt     : tracked keypoints.
        trk     : if the point is currently tracked.

    DB.observation : NDRecord( (frame_id, landmark_id, point) )
        frame_id    : frame index.
        landmark_id : landmark index.
        point       : pixel-coordinate keypoint location.

    """

    def __init__(self, img_fmt=None, dsc_fmt=None, path=''):
        # reset data fields
        self.frame_ = None
        self.landmark_ = None
        self.observation_ = None

        # build ...
        if path:
            # load from file
            self.load(path)
        else:
            if (img_fmt is not None):
                self.frame_ = self.build_frame(img_fmt)
            if (dsc_fmt is not None):
                self.landmark_ = self.build_landmark(dsc_fmt)
            self.observation_ = self.build_observation()

        # report construction status
        data = (self.frame_, self.landmark_, self.observation_)
        if np.any([e is None for e in data]):
            print('Not enough information : DB Initialization will be delayed.')

    """ container wrappers """

    def reset(self):
        recs = [self.frame_, self.landmark_, self.observation_]
        for rec in recs:
            if rec is None:
                continue
            rec.reset()

    def extend(self, db):
        rec_old = [self.frame_, self.landmark_, self.observation_]
        rec_new = [db.frame_, db.landmark_, db.observation_]
        for (r_o, r_n) in zip(rec_old, rec_new):
            r_o.extend(r_n)

    """ datatype wrappers """

    def build_frame(self, img_fmt):
        img_s, img_t = img_fmt
        return NDRecord([
            ('index', np.int32, 1),
            ('stamp', np.float32, 1),
            ('image', img_t, img_s),
            ('pose', np.float32, 15),
            ('cov', np.float32, (15, 15)),
            ('is_kf', np.bool, 1),
            ('feat', Feature, 1)
        ])

    def build_landmark(self, dsc_fmt):
        dsc_s, dsc_t = dsc_fmt
        # instantiate data
        return NDRecord([
            # index tracking
            ('index', np.int32, 1),
            ('src', np.int32, 1),
            # feature descriptor
            ('dsc', dsc_t, dsc_s),
            ('rsp', np.float32, 1),
            # initial observation
            ('pt0', np.float32, 2),
            ('invd', np.float32, 1),
            # derived : depth/position cache
            ('depth', np.float32, 1),
            ('pos', np.float32, 3),
            # tracking
            ('track', np.bool, 1),
            ('pt', np.float32, 2),
            # flag: `triangulated`
            ('tri', np.bool, 1),
            # visualization
            ('col', np.uint8, 3),
        ])

    def build_observation(self):
        return NDRecord([
            ('src_idx', np.int32, 1),
            ('lmk_idx', np.int32, 1),
            ('point', np.float32, 2)
        ])

    """ properties """
    @property
    def frame(self):
        return self.frame_

    @property
    def landmark(self):
        return self.landmark_

    @property
    def observation(self):
        return self.observation_

    @property
    def keyframe(self):
        kf_idx = np.nonzero(self.frame['is_kf'])[0]
        #print('keyframe index : {}'.format(kf_idx))
        return self.frame[kf_idx]

    """ size """
    @property
    def n_frame(self):
        return self.frame_.size

    @property
    def n_landmark(self):
        return self.landmark_.size

    @property
    def n_observation(self):
        return self.observation_.size

    """ persistence """

    def load(self, path):
        def D_(p): return os.path.join(path, p)
        print('Please wait while loading DB ...')
        self.frame_ = np.load(D_('frame.npy'))
        self.landmark_ = np.load(D_('landmark.npy'))
        self.observation_ = np.load(D_('observation.npy'))
        print('DB Load Complete!')

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        def D_(p): return os.path.join(path, p)
        np.save(D_('frame.npy'), self.frame)
        np.save(D_('landmark.npy'), self.landmark)
        np.save(D_('observation.npy'), self.observation)
        #np.save(D_('pose.npy'), self.frame['pose'])
        # np.save(D_('map_pos.npy'), self.landmark['pos']) #np.save(D_('map_col.npy'), self.landmark['col'])

    def prune(self, src_idx=None, lmk_idx=None, obs_idx=None):
        # compute masks
        if src_idx is not None:
            msk_src = np.in1d(self.observation_['src_idx'], src_idx)
        else:
            # keep all
            msk_src = np.ones(self.observation_.size, dtype=np.bool)

        if lmk_idx is not None:
            msk_lmk = np.in1d(self.observation_['lmk_idx'], lmk_idx)
        else:
            # keep all
            msk_lmk = np.ones(self.observation_.size, dtype=np.bool)

        if obs_idx is not None:
            msk_obs = np.zeros(self.observation_.size, dtype=np.bool)
            msk_obs[obs_idx] = True
        else:
            # keep all
            msk_obs = np.ones(self.observation_.size, dtype=np.bool)

        msk = np.logical_and.reduce([
            msk_src, msk_lmk, msk_obs
        ])
        n_obs = msk.sum()
        self.observation_[:n_obs] = self.observation[msk]
        self.observation_.resize(n_obs)

        # == handle source ==
        #print 'sort-check[src]', np.all( np.diff(self.observation['src_idx']) >= 0 )
        idx_src = np.unique(self.observation['src_idx'])
        n_src = len(idx_src)
        self.frame_[:n_src] = self.frame[idx_src]
        self.frame_.resize(n_src)
        # remap_indices()
        map_src = np.empty(
            shape=(1 + self.frame['index'].max()), dtype=np.int32)
        map_src[idx_src] = np.arange(n_src)
        self.observation_['src_idx'] = map_src[self.observation_['src_idx']]
        self.landmark_['src'] = map_src[self.landmark_['src']]
        self.frame['index'] = np.arange(n_src)

        # == handle landmark ==
        #print 'sort-check', np.all( np.diff(self.landmark['index']) >= 0 )
        #print self.landmark['index'][0]
        idx_lmk = np.unique(self.observation['lmk_idx'])
        n_lmk = len(idx_lmk)
        # print 'pre', self.landmark[idx_lmk][:5]
        self.landmark_[:n_lmk] = self.landmark[idx_lmk].copy()
        # print 'post', self.landmark[:5]
        self.landmark_.resize(n_lmk)
        # remap_indices()
        map_lmk = np.empty(
            shape=(1 + self.landmark['index'].max()), dtype=np.int32)
        map_lmk[idx_lmk] = np.arange(n_lmk)
        self.observation_['lmk_idx'] = map_lmk[self.observation_['lmk_idx']]
        self.landmark['index'] = np.arange(n_lmk)

def main():
    ex = cv2.ORB_create()
    img_t = np.uint8
    img_s = (3, 2, 1)
    dsc_t = (np.uint8 if ex.descriptorType() == cv2.CV_8U
             else np.float32)
    dsc_s = ex.descriptorSize()

    img_fmt = (img_s, img_t)
    dsc_fmt = (dsc_s, dsc_t)

    db = DB(img_fmt=img_fmt, dsc_fmt=dsc_fmt)
    entry = (1, np.zeros(dtype=img_t, shape=img_s),
             np.zeros(6), np.zeros(6), True
             )

    # db.frame.append(entry)
    # db.frame.append(entry)
    # db.frame.data[-1]['image'] += 1
    # print db.frame.data
    # print db.frame.size


if __name__ == '__main__':
    main()
