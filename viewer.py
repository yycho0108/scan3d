#!/usr/bin/env python2
import sys
import numpy as np
import pyqtgraph as pg
import pyqtgraph.examples
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui
from db import DB
from db import L_POS, L_VEL, L_ACC, A_POS, A_VEL
from optimize.rot import rpy_to_axa
from cho_util import vmath as vm

class Viewer(object):
    def __init__(self):
        self.app_ = QtGui.QApplication(sys.argv)
        self.win_, self.panels_ = self.build_win()

    def on_frame(self, frame, db):
        print('frame index : {}'.format(frame['index']))

        # draw_image()
        image = frame['image'].swapaxes(0,1)[..., ::-1] 
        self.panels_['img'].setImage(image)

        # draw_cloud()
        pos = db.landmark['pos']
        col = db.landmark['col'][..., ::-1] / 255.0
        self.panels_['cld'].setData(pos=pos, color=col)

        # draw_camera()
        axa = rpy_to_axa(frame['pose'][A_POS])
        h = vm.norm(axa)
        u = axa / h
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

        # cloud 3d view
        glvw = gl.GLViewWidget()
        sp_cld = gl.GLScatterPlotItem()
        glvw.addItem(sp_cld)
        panels['cld'] = sp_cld

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
        self.db_     = db
        self.cfg_    = cfg
        self.viewer_ = Viewer()
        self.index_  = 0
        self.auto_   = True
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
