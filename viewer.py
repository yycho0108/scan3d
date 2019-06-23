#!/usr/bin/env python2
import sys
import numpy as np
import pyqtgraph as pg
import pyqtgraph.examples
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui
from db import DB
from db import L_POS, L_VEL, L_ACC, A_POS, A_VEL

from cho_util import vmath as vm
from cho_util.math import transform as tx
from util import *

from GLViewWidget import GLViewWidget
#GLViewWidget = gl.GLViewWidget


class Viewer(object):
    def __init__(self, cfg):
        self.app_ = QtGui.QApplication(sys.argv)
        self.win_, self.panels_ = self.build_win()
        self.cfg_ = cfg

    def on_frame(self, frame, db):
        #print('frame index : {}'.format(frame['index']))
        self.win_.setWindowTitle('Frame {:04d}'.format(frame['index']))

        # draw_image()
        image = frame['image'].swapaxes(0, 1)[..., ::-1]
        self.panels_['img'].setImage(image)

        self.panels_['kpt'].setData(
            pos=frame['feat'].pt)

        pt_obs = db.observation['point'][
            (db.observation['src_idx'] == frame['index'])
        ]
        self.panels_['obs_pt'].setData(
            pos=pt_obs)

        map_frame = db.keyframe[0]
        pt_prj = project_to_frame(db.landmark['pos'],
                                  source_frame=map_frame,
                                  target_frame=frame,  # TODO: keyframe may no longer be true?
                                  K=self.cfg_['K'],
                                  D=self.cfg_['D'])
        prj_msk = np.logical_and.reduce([
            0 <= pt_prj[:, 0],
            pt_prj[:, 0] < self.cfg_['w'],
            0 <= pt_prj[:, 1],
            pt_prj[:, 1] < self.cfg_['h'],
        ])
        self.panels_['prj_pt'].setData(
            pos=pt_prj[prj_msk],
            #col=db.landmark['col'][prj_msk, ::-1] / 255.0
        )
        self.panels_['prj_pt'].setBrush(
            [QtGui.QBrush(QtGui.QColor(*c)) for c in db.landmark['col'][prj_msk, ::-1]])

        # draw_cloud()
        msk = db.landmark['tri']
        pos = db.landmark['pos'][msk]
        col = db.landmark['col'][msk][..., ::-1] / 255.0
        self.panels_['cld'].setData(pos=pos, color=col)

        # draw_camera()
        rxn = tx.rotation.euler.to_axis_angle(frame['pose'][A_POS])
        u = rxn[:3]
        h = rxn[3]
        self.panels_['cam'].resetTransform()
        self.panels_['cam'].rotate(np.rad2deg(h), u[0], u[1], u[2])
        self.panels_['cam'].translate(*frame['pose'][L_POS])

    def build_win(self):
        panels = {}
        win = pg.GraphicsWindow(title='SLAM Database Viewer')
        win.resize(640, 480)
        win.setWindowTitle('DB Viewer')
        pg.setConfigOptions(antialias=True)

        # image at current frame
        imvw = pg.PlotWidget()
        imvw.invertY(True)
        imvw.setAspectLocked(True)
        img = pg.ImageItem()
        imvw.addItem(img)
        panels['img'] = img

        # (frame) detected keypoints
        sp = pg.ScatterPlotItem()
        sp.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0)))
        imvw.addItem(sp)
        panels['kpt'] = sp

        # observed keypoints
        sp = pg.ScatterPlotItem()
        sp.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 255)))
        imvw.addItem(sp)
        panels['obs_pt'] = sp

        # projected landmark keypoints
        sp = pg.ScatterPlotItem()
        imvw.addItem(sp)
        panels['prj_pt'] = sp

        # cloud 3d view
        glvw = GLViewWidget()
        #glvw = GLViewWidget0()
        #glvw.orbit(0, 90)
        sp_cld = gl.GLScatterPlotItem()
        glvw.addItem(sp_cld)
        panels['cld'] = sp_cld
        glvw.setCameraPosition(
            rotation=QtGui.QQuaternion.fromEulerAngles(90, 0, 90).conjugated())

        # map origin
        gx = gl.GLAxisItem()
        glvw.addItem(gx)

        # camera pose
        gx = gl.GLAxisItem()
        gx.setSize(0.1, 0.1, 0.1)
        glvw.addItem(gx)
        panels['cam'] = gx

        # (grid)
        gx = gl.GLGridItem()
        gx.rotate(90, 0, 0, 1)
        gx.rotate(90, 1, 0, 0)
        glvw.addItem(gx)

        # finalize layout?
        layoutgb = QtGui.QGridLayout()
        win.setLayout(layoutgb)
        layoutgb.addWidget(imvw, 0, 1)
        layoutgb.addWidget(glvw, 0, 0)

        # manage the whole size thing
        imvw.sizeHint = lambda: pg.QtCore.QSize(100, 100)
        glvw.sizeHint = lambda: pg.QtCore.QSize(100, 100)
        glvw.setSizePolicy(imvw.sizePolicy())
        return win, panels

    def show(self):
        self.win_.show()
        self.app_.exec_()


class DBViewer(object):
    def __init__(self, db, cfg):
        self.db_ = db
        self.cfg_ = cfg
        self.viewer_ = Viewer(cfg)
        self.index_ = 0
        self.auto_ = True
        self.change_ = False

    def start(self):
        self.change_ = True
        QtCore.QTimer.singleShot(1, self.step)

    def step(self):
        if self.change_:
            # actually show the frame
            frame = self.db_.frame[self.index_]
            self.viewer_.on_frame(frame, self.db_)

        if self.auto_:
            # automatic increment
            if (self.index_+1) < self.db_.frame.size:
                self.index_ += 1
                self.change_ = True

        # call show again
        QtCore.QTimer.singleShot(1, self.step)

    def run(self):
        self.start()
        self.viewer_.show()


def main():
    ex = False
    if ex:
        pyqtgraph.examples.run()
    else:
        db = DB(path='/tmp/db')
        cfg = np.load('/tmp/db/config.npy').item()
        DBViewer(db, cfg).run()


if __name__ == '__main__':
    main()
