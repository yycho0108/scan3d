import numpy as np
import cv2

from db import DB
from track import Tracker
from match import Matcher
from kalman.ekf import build_ekf
from cho_util.viz import draw_matches

# configuration
CFG = dict(
        pLK = dict(
            winSize         = (31,31),
            maxLevel        = 4,
            crit            = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.03),
            flags           = 0,
            minEigThreshold = 1e-3
            )
        )

class Pipeline(object):
    def __init__(self):
        self.db_        = None
        self.extractor_ = cv2.ORB_create()
        self.matcher_   = Matcher(ex=self.extractor_)
        self.tracker_   = Tracker(pLK=CFG['pLK'])
        self.kf_        = build_ekf()

    def initialize(self, img):
        ex       = self.extractor_
        img_fmt  = (img.shape, img.dtype)
        dsc_t    = (np.uint8 if ex.descriptorType() == cv2.CV_8U else np.float32)
        dsc_fmt  = (self.extractor_.descriptorSize(), dsc_t)
        self.db_ = DB(img_fmt=img_fmt, dsc_fmt=dsc_fmt)

    def process(self, img, dt, data={}):
        if self.db_ is None:
            self.initialize(img)
            frame = (0, img, np.zeros(15), 1e-6*np.eye(15), True)
            self.db_.frame.append(frame)
            kpt, dsc = self.extractor_.detectAndCompute(img, None)
            self.db_.observation.append([kpt, dsc])
            self.db_.state_['track'] = cv2.KeyPoint.convert(kpt)
        else:
            # populate frame
            frame0 = self.db_.frame[-1]
            index = self.db_.frame.size
            self.kf_.x = frame0['pose']
            self.kf_.P = frame0['cov']
            self.kf_.predict(dt)
            frame1 = (index, img, self.kf_.x, self.kf_.P, True)
            self.db_.frame.append(frame1)

            # populate observation
            kpt1, dsc1 = self.extractor_.detectAndCompute(img, None)
            self.db_.observation.append([kpt1, dsc1])

            # process ...
            frame0, frame1 = self.db_.frame[-2], self.db_.frame[-1]
            img0, img1 = frame0['image'], frame1['image']
            kpt0, dsc0 = self.db_.observation[-2]
            kpt1, dsc1 = self.db_.observation[-1]

            # track
            t_pt, t_idx = self.tracker_.track(img0, img1, self.db_.state_['track'])
            self.db_.state_['track'] = t_pt[t_idx]

            # match
            mi0, mi1 = self.matcher_.match(dsc0, dsc1)
            viz0 = cv2.drawKeypoints(img0, kpt0, None)
            viz1 = cv2.drawKeypoints(img1, kpt1, None)
            viz = draw_matches(viz0, viz1, cv2.KeyPoint.convert(kpt0)[mi0], cv2.KeyPoint.convert(kpt1)[mi1])
            data['viz'] = viz
            

def main():
    pl  = Pipeline()
    src = './scan_20190212-233625.h264'
    cap = cv2.VideoCapture(src)
    dt = (1.0 / 25.0) # (??)
    cv2.namedWindow('viz', cv2.WINDOW_NORMAL)
    while True:
        res, img = cap.read()
        if not res: break
        data = {}
        pl.process(img, dt, data)
        if 'viz' in data:
            cv2.imshow('viz', data['viz'])
        k = cv2.waitKey(0)
        if k in [27, ord('q')]: break

if __name__ == '__main__':
    main()
