#!/usr/bin/env python2
import numpy as np
import cv2
from record import NDRecord
from collections import namedtuple

Feature = namedtuple('Feature',
        ['kpt', 'dsc', 'pt'],
        )

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

    DB.observation : NDRecord( (frame_id, landmark_id, point) )
        frame_id    : frame index.
        landmark_id : landmark index.
        point       : pixel-coordinate keypoint location.

    """
    def __init__(self, img_fmt, dsc_fmt):
        # parse data formats
        img_s, img_t = img_fmt
        dsc_s, dsc_t = dsc_fmt

        # instantiate data
        self.frame_ = NDRecord([
            ('index' , np.int32   , 1)     ,
            ('image' , img_t      , img_s) ,
            ('pose'  , np.float32 , 15)    ,
            ('cov'   , np.float32 , (15, 15)) ,
            ('is_kf' , np.bool    , 1)     ,
            ('feat'   , Feature     , 1)
            ])
        self.landmark_ = NDRecord([
            ('index' , np.int32   , 1)     ,
            ('dsc'   , dsc_t      , dsc_s) ,
            ('pos'   , np.float32 , 3)     ,
            ])

        # current frame state
        self.state_ = {
                'track' : [],
                'path'  : []
                }

    """ properties """
    @property
    def frame(self):
        return self.frame_
    @property
    def landmark(self):
        return self.landmark_
    @property
    def state(self):
        return self.state_

    @property
    def keyframe(self):
        kf_idx = np.nonzero(self.frame['is_kf'])[0]
        #print('keyframe index : {}'.format(kf_idx))
        return self.frame[kf_idx]

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

    db.frame.append(entry)
    db.frame.append(entry)
    db.frame.data[-1]['image'] += 1
    print db.frame.data
    print db.frame.size

if __name__ == '__main__':
    main()
