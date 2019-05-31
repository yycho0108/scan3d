#!/usr/bin/env python2

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cho_util.viz.mpl import set_axes_equal
from cho_util import vmath as vm
from cho_util.viz.draw import draw_points, draw_matches
from tf import transformations as tx
from db import DB
from util import *

from profilehooks import profile

def sub_axis(ax, rect, axisbg='w', **axargs):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],axisbg=axisbg, **axargs)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

def draw_frame(ax, frame, db, cfg):
    map_frame = db.keyframe[0]

    img = frame['image'].copy()

    obs = db.observation[
            db.observation['src_idx'] == frame['index']
            ]
    cld = db.landmark[obs['lmk_idx']]
    xyz = cld['pos']
    col = cld['col']

    draw_points(img, frame['feat'].pt, color=(0,255,0))

    if len(xyz) > 0:
        # red = projected points
        pt2 = project_to_frame(xyz, map_frame, frame,
                cfg['K'], cfg['D'])
        img = draw_matches(img, img, pt2, obs['point'],
                single=True)

        #draw_points(img, pt2, color=(0,0,255) )
        #draw_points(img, obs['point'], color=(255,0,0))

    ax_img = sub_axis(ax, [0.0, 0.0, 0.5, 1.0])
    ax_img.imshow(img[..., ::-1])

    # local cloud
    ax_cld = sub_axis(ax, [0.5, 0.0, 0.5, 1.0], projection='3d')
    xyz = transform_cloud(xyz, map_frame, frame)
    xyz = vm.tx3( vm.tx.euler_matrix(-np.pi/2, 0, -np.pi/2), xyz)
    ax_cld.scatter(xyz[:,0], xyz[:,1], xyz[:,2],
            c = (col[...,::-1] / 255.0))
    ax_cld.view_init(elev=0, azim=180)

def draw_map(ax, frame, db, cfg):
    # global - top-down view

    # extract data
    map_frame = db.keyframe[0]
    xyz = db.landmark['pos']
    col = db.landmark['col']

    #idx = np.random.choice(len(xyz), size=2048, replace=False)
    #xyz = xyz[idx]
    #col = col[idx]

    # draw (3d)
    ax3 = sub_axis(ax, [0.0, 0.0, 1.0, 0.5], projection='3d')
    ax3.scatter(xyz[:,0], xyz[:,1], xyz[:,2],
            s = 0.1,
            c = (col[...,::-1] / 255.0),
            )
    for fr in db.frame:
        xfm_pose = pose_to_xfm(fr['pose'])
        txn = tx.translation_from_matrix(xfm_pose)
        rxn = tx.euler_from_matrix(xfm_pose)
        draw_pose(ax3, txn, rxn, alpha=0.02)
    draw_pose(ax3, frame['pose'][0:3], frame['pose'][9:12])
    set_axes_equal(ax3)

    # draw (2d)
    T_R = vm.tx.euler_matrix(-np.pi/2, 0, -np.pi/2)
    ax2 = sub_axis(ax, [0.0, 0.5, 1.0, 0.5])
    xyz = vm.tx3(T_R, xyz)
    ax2.scatter(xyz[:,0], xyz[:,1],
            s = 0.1,
            c = (col[...,::-1] / 255.0)
            )
    for fr in db.frame:
        xfm_pose = pose_to_xfm(fr['pose'])
        r_xfm_pose = T_R.dot(xfm_pose)
        txn = tx.translation_from_matrix(r_xfm_pose)
        rxn = tx.euler_from_matrix(r_xfm_pose)
        draw_pose(ax2, txn, rxn, alpha=0.02)
    fr = frame
    xfm_pose = pose_to_xfm(fr['pose'])
    r_xfm_pose = T_R.dot(xfm_pose)
    txn = tx.translation_from_matrix(r_xfm_pose)
    rxn = tx.euler_from_matrix(r_xfm_pose)
    draw_pose(ax2, txn, rxn, alpha=1.0)
    ax2.set_aspect('equal')

def transform_cloud(cloud, source_frame, target_frame):
    R, t = get_transform(source_frame, target_frame)
    return vm.rtx3(R, t.ravel(), cloud)

def draw_pose(ax, p, a, s=0.1, style='-', alpha=1.0):
    draw_3d = isinstance(ax, Axes3D)

    o = p # translational origin
    V = np.eye(3) # orthogonal basis xyz
    R = tx.euler_matrix(*a, axes='rzyx')

    ux, uy, uz = R[:3, :3].dot(V).T
    if draw_3d:
        ax.plot([o[0],o[0]+s*ux[0]], [o[1],o[1]+s*ux[1]], [o[2],o[2]+s*ux[2]], 'r'+style, alpha=alpha)
        ax.plot([o[0],o[0]+s*uy[0]], [o[1],o[1]+s*uy[1]], [o[2],o[2]+s*uy[2]], 'g'+style, alpha=alpha)
        ax.plot([o[0],o[0]+s*uz[0]], [o[1],o[1]+s*uz[1]], [o[2],o[2]+s*uz[2]], 'b'+style, alpha=alpha)
    else:
        ax.plot([o[0],o[0]+s*ux[0]], [o[1],o[1]+s*ux[1]], 'r'+style, alpha=alpha)
        ax.plot([o[0],o[0]+s*uy[0]], [o[1],o[1]+s*uy[1]], 'g'+style, alpha=alpha)
        ax.plot([o[0],o[0]+s*uz[0]], [o[1],o[1]+s*uz[1]], 'b'+style, alpha=alpha)

#@profile
def main():
    db = DB(path='/tmp/db')
    db.load('/tmp/db')
    cfg = np.load('/tmp/db/config.npy').item()
    pose = db.frame['pose']
    xyz  = pose[:, 0:3]
    rpy  = pose[:, 9:12]
    cld  = db.landmark['pos']
    col  = db.landmark['col']

    fig = plt.figure(dpi=200)

    for frame in db.frame:
        print frame['index']
        fig.clf()
        ax = fig.gca()
        ax.cla()

        ax1 = plt.subplot2grid((1,2), (0,0), 1, 1)
        ax2 = plt.subplot2grid((1,2), (0,1), 1, 1)
        plt.subplots_adjust(0,0,1,1,0,0)

        draw_frame(ax1, frame, db, cfg)
        draw_map(ax2, frame, db, cfg)

        plt.pause(0.001)
        plt.savefig('/tmp/db/gui{:04d}.png'.format(frame['index']))
        #plt.show()

    #ax = plt.gca(projection='3d')
    ######xyz = vm.tx3( vm.tx.euler_matrix(-np.pi/2, 0, -np.pi/2), xyz)
    #msk = (np.linalg.norm(cld, axis=-1) < 1000)
    #cld = cld[msk]
    #col = col[msk]
    #ax.scatter(cld[:,0], cld[:,1], cld[:,2],
    #        c = (col[...,::-1] / 255.0))

    #ax.plot(xyz[:,0], xyz[:,1], xyz[:,2], '+-')
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    #ax.set_zlabel('z')
    #set_axes_equal(ax)

    #for p, a in zip(xyz, rpy):
    #    draw_pose(ax, p, a)
    #plt.show()

if __name__ == '__main__':
    main()
