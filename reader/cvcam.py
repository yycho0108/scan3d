#!/usr/bin/env python2

import os
import cv2
import numpy as np

class CVCameraReader(object):
    def __init__(self, src):
        self.src_ = src
        self.cam_ = cv2.VideoCapture( src )
        w   = self.cam_.get(cv2.CAP_PROP_FRAME_WIDTH)
        h   = self.cam_.get(cv2.CAP_PROP_FRAME_HEIGHT)
        n   = self.cam_.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = self.cam_.get(cv2.CAP_PROP_FPS)

        self.shape_  = (h, w)
        self.length_ = n
        self.dt_     = (1.0 / fps)

        self.time_   = 0.0
        self.idx_    = 0

    def read(self):
        self.time_ += self.dt_
        suc, img = self.cam_.read()
        self.idx_ += 1
        return suc, self.idx_, self.time_, img

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
