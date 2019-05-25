import numpy as np
import cv2
import sys

from skimage.feature import canny
from skimage.transform import hough_ellipse

from cho_util.cam import KeyCallback
from track import Tracker

def create_det():
    p = cv2.SimpleBlobDetector_Params()
    p.minThreshold = 0
    p.maxThreshold = 100

    p.filterByCircularity = True
    p.minCircularity = 0.7

    p.filterByConvexity = True
    p.minConvexity = 0.8

    p.filterByArea = True
    p.minArea = 100

    p.filterByInertia = True
    p.minInertiaRatio = 0.8
    p.maxInertiaRatio = 1.2

    det = cv2.SimpleBlobDetector_create(p)
    return det

def find_ellipse(det, img):
    mono = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ##print mono.shape
    ##return cv2.Canny(mono, 0.55, 0.8)
    edges = canny(mono, sigma=2.0,
            low_threshold=0.55, high_threshold=0.8)
    ##return np.uint8( edges * 255 )
#
    ##cv2.imshow('canny', 255*edges.astype(np.uint8))
    ##cv2.waitKey(0)
    ##return
#

    #res = hough_ellipse(edges, 
    #        accuracy=1.,
    #        threshold=10,
    #        min_size=20,
    #        max_size=40)
    #res.sort(order='accumulator')

    mono = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kpt = det.detect(mono)

    mono_blur = cv2.medianBlur(mono, 5)
    cir = cv2.HoughCircles(mono_blur, cv2.HOUGH_GRADIENT,
            1, 20,
            param1=60, param2=30,
            minRadius=0, maxRadius=30
            )
    if cir is not None:
        for cx,cy,r in cir[0].astype(np.int32):
            cv2.circle(img, (cx,cy), r, (0,255,0), 2)
    return cv2.drawKeypoints(img, kpt, img.copy())

cap = cv2.VideoCapture('./scan_20190212-233625.h264')
kcb = KeyCallback()
det = create_det()

cv2.namedWindow('img', cv2.WINDOW_NORMAL)

while True:
    # input image
    ret, img = cap.read()
    if not ret:
        break
    img = cv2.resize(img, None, fx=0.25, fy=0.25)

    # process
    #try:
    viz = find_ellipse(det, img)
    #viz = cv2.cvtColor(viz, cv2.COLOR_GRAY2BGR)
    #viz = cv2.addWeighted(img, 0.5, viz, 0.5, 0.0)
    cv2.imshow('img', viz)
    #except Exception as e:
    #    print('e : {} .. {}'.format(e,sys.exc_info()))
    #    break

    # key callback handling
    k = cv2.waitKey(0)
    if kcb(k):
        break

#cv2.destroyAllWindows()
