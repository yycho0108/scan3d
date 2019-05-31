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
        self.frame_       = None
        self.landmark_    = None
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
            # current frame state
            self.state_ = {
                    'track' : [],
                    'path'  : []
                    }

        # report construction status
        data = (self.frame_, self.landmark_, self.observation_)
        if np.any([e is None for e in data]):
            print('Not enough information : DB Initialization will be delayed.')

    def build_frame(self, img_fmt):
        img_s, img_t = img_fmt
        return NDRecord([
            ('index' , np.int32   , 1       ) ,
            ('stamp' , np.float32 , 1       ) ,
            ('image' , img_t      , img_s   ) ,
            ('pose'  , np.float32 , 15      ) ,
            ('cov'   , np.float32 , (15, 15)) ,
            ('is_kf' , np.bool    , 1       ) ,
            ('feat'  , Feature    , 1       )
            ])

    def build_landmark(self, dsc_fmt):
        dsc_s, dsc_t = dsc_fmt
        # instantiate data
        return NDRecord([
            ('index' , np.int32   , 1     ) ,
            ('src'   , np.int32   , 1     ) ,
            ('dsc'   , dsc_t      , dsc_s ) ,
            ('rsp'   , np.float32 , 1     ) ,
            ('pos'   , np.float32 , 3     ) ,
            ('pt'    , np.float32 , 2     ) ,
            ('tri'   , np.bool    , 1     ) ,
            ('col'   , np.uint8   , 3     ) ,
            ('track' , np.bool    , 1     )
            ])

    def build_observation(self):
        return NDRecord([
            ('src_idx', np.int32, 1),
            ('lmk_idx', np.int32, 1),
            ('point' , np.float32, 2)
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
    def state(self):
        return self.state_

    @property
    def keyframe(self):
        kf_idx = np.nonzero(self.frame['is_kf'])[0]
        #print('keyframe index : {}'.format(kf_idx))
        return self.frame[kf_idx]

    def load(self, path):
        D_ = lambda p : os.path.join(path, p)

        self.frame_       = np.load(D_('frame.npy'))
        self.landmark_    = np.load(D_('landmark.npy'))
        self.observation_ = np.load(D_('observation.npy'))

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        D_ = lambda p : os.path.join(path, p)
        np.save(D_('frame.npy'), self.frame)
        np.save(D_('landmark.npy'), self.landmark)
        np.save(D_('observation.npy'), self.observation)
        #np.save(D_('pose.npy'), self.frame['pose'])
        #np.save(D_('map_pos.npy'), self.landmark['pos'])
        #np.save(D_('map_col.npy'), self.landmark['col'])

def main():
    ex = cv2.ORB_create()
    img_t = np.uint8
    img_s = (3,2,1)
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
