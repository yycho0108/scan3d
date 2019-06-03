#!/usr/bin/env python2

import os
import numpy as np
import cv2
import yaml
import subprocess

def get_rotation(f):
    """
    Hack to get video rotation state.
    (OpenCV Does not appear to support getting rotation metadata)
    """

    cmd = 'ffprobe -loglevel error -select_streams v:0 -show_entries stream_tags=rotate -of default=nw=1:nk=1 {}'.format(f)
    p = subprocess.Popen(cmd.split(" "), stdout=subprocess.PIPE, close_fds=True)
    so, se = p.communicate()
    return int(so)

class AdvioReader(object):
    def __init__(self, src, idx=1):
        # cache params
        self.src_ = src
        self.idx_ = idx
        
        self.meta_ = {}
        self.parse(self.meta_)
        self.build(self.meta_)

        self.dir_ = os.path.join(src,
                'advio-{:02d}'.format(idx),
                'iphone')
        r = get_rotation(os.path.join(self.dir_, 'frames.mov'))
        assert( (r % 90) == 0)
        self.rotation_ = r
        self.cap_ = cv2.VideoCapture(
                os.path.join(self.dir_, 'frames.mov')
                )
        self.stamp_ = np.loadtxt(
                os.path.join(self.dir_, 'frames.csv'),
                delimiter=','
                )[:,0]
        self.pos_ = 0
        assert( self.cap_.get(cv2.CAP_PROP_FRAME_COUNT) == len(self.stamp_) )

    def parse(self, _={}):
        """ 
        Populate w, h, K, D
        """
        # unroll params
        src = self.src_
        idx = self.idx_

        # determine camera index
        if (1 <= idx <= 12):
            k_idx = 2
        elif (13 <= idx <= 17):
            k_idx = 3
        elif (18 <= idx <= 19):
            k_idx = 1
        elif (20 <= idx <= 23):
            k_idx = 4

        # open file and parse data
        yml = os.path.join(src,
                'iphone-{:02d}.yaml'.format(k_idx)
                )
        with open(yml, 'r') as f:
            data = yaml.safe_load(f)
        cam = data['cameras'][0]['camera']

        fx,fy,cx,cy=cam['intrinsics']['data']
        w = cam['image_width']
        h = cam['image_height']
        r1, r2, k1, k2 = cam['distortion']['parameters']['data']
        # cv2 distCoeffs() order = (k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,tx,ty ...)
        # k1 k2 radial, p1 p2 tangential

        # format + fill metadata
        _['w'] = w
        _['h'] = h
        _['K'] = np.float32([
            fx,0,cx,
            0,fy,cy,
            0,0,1
            ]).reshape(3,3)
        _['D'] = np.float32([k1, k2, r1, r2])

    def build(self, _):
        """ Prepare for rectification and update K matrix. """
        size = (_['w'],_['h'])
        K2, roi = cv2.getOptimalNewCameraMatrix(
                _['K'], _['D'], size, 0) # TODO : 0 or 1 depending on use case ...
        #K2 = _['K']
        m1, m2 = cv2.initUndistortRectifyMap(
                _['K'], _['D'], None, K2, size,
                cv2.CV_32FC1
                )
        # save unrectified parameters
        _['K0'] = _['K']
        _['D0'] = _['D']
        # update with rectified params
        _['K']  = K2 # IMPORTANT : overriding default intrinsic matrix.
        _['D']  = np.zeros_like(_['D'])
        _['m1'] = m1
        _['m2'] = m2

    def rectify(self, img):
        _ = self.meta_
        return cv2.remap(img, _['m1'], _['m2'],
                cv2.INTER_LINEAR)

    def rotate(self, img):
        if (self.rotation_ == 0):
            return img
        if (self.rotation_ == 90):
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if (self.rotation_ == -90):
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if (self.rotation_ == 180):
            return cv2.rotate(img, cv2.ROTATE_180)

    def read(self):
        suc, img = self.cap_.read()
        if not suc:
            return suc, self.pos_, -1, None
        img = self.rotate(img)
        img = self.rectify(img)
        stamp = self.stamp_[self.pos_]
        self.pos_ += 1
        return suc, self.pos_, stamp, img

    def read_all(self):
        while True:
            res = self.read()
            if not res[0]:
                break
            yield res[1:]

    def set_pos(self, pos):
        self.pos_ = pos
        self.cap_.set( cv2.CAP_PROP_POS_FRAMES, pos)

def main():
    root = '/media/ssd/datasets/ADVIO'
    reader = AdvioReader(root)
    for res in reader.read_all():
        idx, stamp, img = res
        cv2.imshow('img', img)
        #print('stamp : {}'.format(stamp))
        k = cv2.waitKey(1)
        if k in [27, ord('q')]:
            break

if __name__ == '__main__':
    main()
