#!/usr/bin/env python2

import os
import cv2
import numpy as np
import time

class CVCameraReader(object):
    def __init__(self, src, K=None, D=None):
        self.src_ = src
        self.cam_ = cv2.VideoCapture( src )
        w   = self.cam_.get(cv2.CAP_PROP_FRAME_WIDTH)
        h   = self.cam_.get(cv2.CAP_PROP_FRAME_HEIGHT)
        n   = self.cam_.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = self.cam_.get(cv2.CAP_PROP_FPS)

        self.shape_  = (h, w)
        self.length_ = n
        self.dt_     = (1.0 / fps)

        self.pos_    = 0
        self.meta_   = dict(
                w=w,
                h=h,
                K=K,
                D=D,
                )
        self.tref_   = None
        self.time_   = 0.0

    def read(self):
        if self.length_ < 0:
            # real-time
            if self.tref_ is None:
                self.tref_ = time.time()
            self.time_ = time.time() - self.tref_
        else:
            self.time_ += self.dt_
        suc, img = self.cam_.read()
        self.pos_ += 1
        return suc, self.pos_, self.time_, img

    def set_pos(self, pos):
        if self.length_ < 0:
            return False
        self.pos_ = pos
        self.cap_.set( cv2.CAP_PROP_POS_FRAMES, pos)

def main():
    source = os.path.expanduser('~/Downloads/scan_20190212-233625.h264')
    reader = CVCameraReader(source)

    while True:
        suc, idx, stamp, img = reader.read()
        if not suc:
            break
        cv2.imshow('img', img)
        print('stamp : {}'.format(stamp))
        k = cv2.waitKey(0)
        if k in [27, ord('q')]:
            break

if __name__ == '__main__':
    main()
