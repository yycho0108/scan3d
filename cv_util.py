#!/usr/bin/env python2
import cv2
import numpy as np
from cho_util import vmath as vm
from collections import namedtuple

def H(*args, **kwargs):
    """ cv2.findHomography() wrapper """
    h, msk = cv2.findHomography(
            *args, **kwargs)
    if msk is not None:
        msk = msk[:,0].astype(np.bool)
    return h, msk

def E(*args, **kwargs):
    """ cv2.findEssentialMat() wrapper """
    e, msk = cv2.findEssentialMat(
            *args, **kwargs)
    if msk is not None:
        msk = msk[:,0].astype(np.bool)
    return e, msk

def F(*args, **kwargs):
    """ cv2.findFundamentalMat() wrapper """
    f, msk = cv2.findFundamentalMat(
            *args, **kwargs)
    if msk is not None:
        msk = msk[:,0].astype(np.bool)
    return f, msk

def project_points(*args, **kwargs):
    """ cv2.projectPoints() wrapper """
    pt2, jac = cv2.projectPoints(*args, **kwargs)
    return np.squeeze(pt2, axis=1)

def correct_matches(F, pta, ptb):
    """ cv2.correctMatches() wrapper """
    pta_f, ptb_f = cv2.correctMatches(F, pta[None,...], ptb[None,...])
    pta_f, ptb_f = [np.squeeze(p, axis=0) for p in (pta_f, ptb_f)]
    return pta_f, ptb_f

def triangulate_points(Pa, Pb, pta, ptb,
        *args, **kwargs):
    """ cv2.triangulatePoints() wrapper """
    pt_h = cv2.triangulatePoints(
            Pa, Pb,
            pta[None,...],
            ptb[None,...])
    return vm.from_h(pt_h.T)
