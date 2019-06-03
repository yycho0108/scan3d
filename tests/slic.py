#!/usr/bin/env python2
import numpy as np 
import sys
sys.path.append('..')
import cv2
from reader import AdvioReader

root = '/media/ssd/datasets/ADVIO'
reader = AdvioReader(root)

# spx = cv2.ximgproc.createSuperpixelLSC()
# spx = cv2.ximgproc.createSuperpixelSLIC()
#help(spx)

def proc(img):
    img = cv2.GaussianBlur(img, (3,3), 1.0)#, dst=img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    return img

for res in reader.read_all():
    idx, stamp, img = res

    spx = cv2.ximgproc.createSuperpixelSEEDS(
        image_width  = reader.meta_['w'],
        image_height = reader.meta_['h'],
        image_channels = 3,
        num_superpixels = 512,
        num_levels      = 1
        )
    spx.iterate(proc(img), 16)

    # spx = cv2.ximgproc.createSuperpixelSLIC(
    #         image=proc(img),
    #         algorithm=cv2.ximgproc.SLIC,
    #         region_size=64,
    #         )
    # spx.iterate(4)

    # spx = cv2.ximgproc.createSuperpixelLSC(
    #         image=cv2.cvtColor(img, cv2.COLOR_BGR2Lab),
    #         region_size=32,
    #         ratio=0.075)
    # spx.iterate(4)

    #lbl = spx.getLabels(img)
    msk = spx.getLabelContourMask(image=img, thick_line=True)
    n = spx.getNumberOfSuperpixels()
    #spx.getLabelContour
    #print (n, lbl.max())
    #print lbl.max(), lbl.min()
    cv2.imshow('img', img)
    viz = cv2.addWeighted(img, 0.75, cv2.cvtColor(msk, cv2.COLOR_GRAY2BGR), 0.5, 0.0)
    cv2.imshow('viz', viz)
    #cv2.imshow('lbl', msk)
    k = cv2.waitKey(1)
    if k in [27, ord('q')]:
        break
