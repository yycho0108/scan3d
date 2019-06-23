#!/usr/bin/env python2

import numpy as np
import cv2
import cv_util as cvu
from reader.advio import AdvioReader
from cho_util import math as cm
from cho_util.math import transform as tx
from cho_util.viz.draw import draw_matches, flow_to_image
from cho_util.viz.mpl import set_axes_equal
from twoview import TwoView

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class DenseRec(object):
    def __init__(self, K):
        self.K_ = K
        self.opt_ = cv2.DISOpticalFlow_create(
                cv2.DISOpticalFlow_PRESET_MEDIUM
                )
        #print(dir(cv2.create))
        #self.opt_ = cv2.optflow.createOptFlow_DIS(
        #        cv2.optflow.DISOPTICAL_FLOW_PRESET_MEDIUM
        #        )

    def filter(self, pt0, pt1):
        try:
            # filter by epipolar constraint
            F, msk = cvu.F(pt0, pt1)
            if F is not None:
                # 1. apply mask
                pt0 = pt0[msk]
                pt1 = pt1[msk]
                # 2. correct matches
                pt0, pt1 = cv2.correctMatches(F, pt0[None,...], pt1[None,...])
                pt0, pt1 = pt0[0], pt1[0]
        except Exception as e:
            print('exception', e)
        return pt0, pt1

    def compute(self, img1, img2,
            P1=None, P2=None, sparse=True,
            data={}
            ):
        """
        Compute Dense Reconstruction.

        Returns:
            cld (points)         : Reconstructed Point Cloud.
            col (color)          : Corresponding Color.

        Populates:
            data['fimg'] (image) : Optical Flow Visualization
            data['viz']  (image) : Match Visualization from draw_matches()
        """
        # 1. flow
        mono1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        mono2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        flow = self.opt_.calc(mono1, mono2, None)
        data['fimg'] = flow_to_image(flow)

        # 2. apply flow
        h, w = img1.shape[:2]
        if sparse:
            pi, pj = np.random.uniform(
                    low=(0,0), high=(h,w), size=(2048,2)
                    ).T.astype(np.int32)
        else:
            pi, pj = np.mgrid[:h,:w].reshape(2,-1)
        dx, dy = flow[pi, pj].swapaxes(0, -1)
        pt0 = np.stack([pj, pi], axis=-1)
        dpt = np.stack([dx, dy], axis=-1)
        pt1 = pt0 + dpt

        pt0 = np.float32(pt0)
        pt1 = np.float32(pt1)

        #from matplotlib import pyplot as plt
        #plt.plot(pt0[:,0], pt0[:,1], 'r.')
        #plt.plot(pt1[:,0], pt1[:,1], 'g+')
        #plt.show()

        # 3. filter
        pt0, pt1 = self.filter(pt0, pt1)

        # [optional] guess projection matrices
        # WARNING : only use this for debugging.
        if (P1 is None and P2 is None):
            suc, det = TwoView(pt0, pt1, self.K_).compute(data=data)
            if not suc:
                return None
            print ( data['dbg-tv'] )
            R = np.float32( data['R'] )
            t = np.float32( data['t'] )

            P1 = np.concatenate([R,t.reshape(3,1)], axis=1)
            P2 = np.eye(3,4)

            # 4. parse data
            msk = data['msk_cld']
            cld = data['cld1'][msk]
        else:
            cld = cvu.triangulate_points(self.K_.dot(P2), self.K_.dot(P1), pt1, pt0)
            msk = np.ones(len(pt0), dtype=np.bool)

        idx = cm.rint(pt1[msk][...,::-1])
        idx = np.clip(idx, (0,0), (h-1,w-1))
        col = img2[idx[:,0], idx[:,1]]

        data['viz']  = draw_matches(img1, img2, pt0[msk], pt1[msk])
        return cld, col

def main():
    root = '/media/ssd/datasets/ADVIO'
    reader = AdvioReader(root, idx=2)
    reader.set_pos(0)
    rec = DenseRec(K = reader.meta_['K'])

    imgs = []

    cv2.namedWindow('viz', cv2.WINDOW_NORMAL)

    h, w = reader.meta_['h'], reader.meta_['w']
    pi, pj = np.random.uniform(low=(0,0), high=(h,w), size=(256,2)).T.astype(np.int32)

    while True:
        suc, idx, stamp, img = reader.read()
        if not suc:
            print('Read failure : likely finished.')
            break

        imgs.append( img )
        if len(imgs) < 2:
            continue

        i0 = max(0, len(imgs) - 8)
        i1 = -1 
        data  = {}
        res = rec.compute(imgs[i0], imgs[i1],
                data=data)

        viz = img
        if 'viz' in data:
            viz = data['viz']

        # draw pointcloud
        if res is not None:
            cld,col = res

            plt.clf()
            plt.subplots_adjust(0,0,1,1,0,0)
            plt.margins(0.0)

            ax = plt.subplot2grid((2,4), (0,0), 2, 1)
            ax.imshow(imgs[i0][..., ::-1])
            ax = plt.subplot2grid((2,4), (0,3), 1, 1)
            ax.imshow(data['viz'][..., ::-1])
            ax = plt.subplot2grid((2,4), (1,3), 1, 1)
            ax.imshow(data['fimg'][..., ::-1])

            ax = plt.subplot2grid((2,4), (0,1), 2, 2, projection='3d')
            ax.cla()
            cld = tx.rotation.euler.rotate([-np.pi/2,0,-np.pi/2], cld)
            ax.scatter(cld[:,0], cld[:,1], cld[:,2]
                    ,c = (col[...,::-1] / 255.0)
                    )

            ax.view_init(elev=0,azim=180)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            set_axes_equal(ax)

            plt.gcf().canvas.draw()
            buf = plt.gcf().canvas.tostring_rgb()
            nc, nr = plt.gcf().canvas.get_width_height()

            # override 'viz'
            #viz = np.fromstring(buf, dtype=np.uint8).reshape(nr,nc,3)

        if viz is not None:
            if res is None:
                cv2.imshow('viz', viz)
            else:
                plt.show()

        # handle key
        k = cv2.waitKey(1)
        if k in [27, ord('q')]:
            break

if __name__ == '__main__':
    main()
