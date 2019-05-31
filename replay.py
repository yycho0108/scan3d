#!/usr/bin/env python2

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cho_util.viz.mpl import set_axes_equal
from cho_util import vmath as vm
from tf import transformations as tx

def draw_pose(ax, p, a, s=0.01, style='-'):
    o = p # translational origin
    V = np.eye(3) # orthogonal basis xyz
    R = tx.euler_matrix(*a, axes='rzyx')

    ux, uy, uz = R[:3, :3].dot(V).T
    ax.plot([o[0],o[0]+s*ux[0]], [o[1],o[1]+s*ux[1]], [o[2],o[2]+s*ux[2]], 'r'+style)
    ax.plot([o[0],o[0]+s*uy[0]], [o[1],o[1]+s*uy[1]], [o[2],o[2]+s*uy[2]], 'g'+style)
    ax.plot([o[0],o[0]+s*uz[0]], [o[1],o[1]+s*uz[1]], [o[2],o[2]+s*uz[2]], 'b'+style)

pose = np.load('/tmp/db/pose.npy')[:-100]
print pose.shape

xyz = pose[:, 0:3]
rpy = pose[:, 9:12]

ax = plt.gca(projection='3d')

#####xyz = vm.tx3( vm.tx.euler_matrix(-np.pi/2, 0, -np.pi/2), xyz)
cld = np.load('/tmp/db/map_pos.npy')
col = np.load('/tmp/db/map_col.npy')
msk = (np.linalg.norm(cld, axis=-1) < 1000)
cld = cld[msk]
col = col[msk]
ax.scatter(cld[:,0], cld[:,1], cld[:,2],
        c = (col[...,::-1] / 255.0))

ax.plot(xyz[:,0], xyz[:,1], xyz[:,2], '+-')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
set_axes_equal(ax)

for p, a in zip(xyz, rpy):
    draw_pose(ax, p, a)
plt.show()
