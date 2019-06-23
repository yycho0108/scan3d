#!/usr/bin/env python2
import numpy as np
import transformations as tx

def axa_to_q(axa):
    """ axis-angle to quaternion """
    # TODO : watch out for h == 0
    h = np.linalg.norm(axa, axis=-1, keepdims=True)
    # u = np.where(h_is_about_zero, axa/h, zero)
    u = axa / h
    w   = np.cos(0.5 * h)
    xyz = np.sin(0.5 * h) * u
    return np.concatenate([xyz, w], axis=-1)

def rpy_to_q(rpy):
    """ euler-angle (RzRyRx) to quaternion """
    h = 0.5 * rpy
    c = np.cos(h)
    s = np.sin(h)

    c1, c2, c3 = [c[...,i] for i in range(3)]
    s1, s2, s3 = [s[...,i] for i in range(3)]

    w = c1 * c2 * c3 + s1 * s2 * s3
    x = -c1 * s2 * s3 + s1 * c2 * c3
    y = c1 * s2 * c3 + s1 * c2 * s3
    z = -s1 * s2 * c3 + c1 * c2 * s3

    return np.stack([x,y,z,w], axis=-1)

def q_to_rpy(q):
    x, y, z, w = [q[..., i] for i in range(4)]
    tx = (2.0 * x)
    ty = (2.0 * y)
    tz = (2.0 * z)

    twx = tx*w
    twy = ty*w
    twz = tz*w
    txx = tx*x
    txy = ty*x
    txz = tz*x
    tyy = ty*y
    tyz = ty*z
    tzz = tz*z

    r00 = (1.0 - (tyy + tzz))
    r10 = (txy + twz)
    r21 = tyz + twx
    r22 = (1.0 - (txx + tyy))
    r20 = (txz - twy)
    r01 = (txy - twz)
    r02 = (txz + twy)

    # general_case
    hy = -np.arcsin(r20)
    schy = np.sign(np.cos(hy))
    hz   = np.arctan2(r10 * schy, r00 * schy)
    hx   = np.arctan2(r21 * schy, r22 * schy)

    is_gimbal = np.less(np.abs(np.abs(r20) - 1.0), 1.0e-6)
    print ('isg', is_gimbal.sum())

    return np.stack([hx,hy,hz], axis=-1)

def q_to_axa(q):
    q   = q+np.finfo(q.dtype).eps # prevent 0-norm
    xyz = q[..., :3]
    w   = q[..., 3:]
    n   = np.linalg.norm(xyz, axis=-1, keepdims=True)
    h   = 2.0 * np.arctan2(n, np.abs(w))
    # with np.errorflags ... or whatev
    with np.errstate(invalid='ignore'):
        ax = np.sign(w) * xyz / n
        ax = np.nan_to_num(ax)
    return h * ax

def axa_to_rpy(axa):
    return q_to_rpy(axa_to_q(axa))
def rpy_to_axa(rpy):
    return q_to_axa(rpy_to_q(rpy))

def main():
    q = np.random.normal(size=(2048,4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    axa = q_to_axa(q)
    rpy = q_to_rpy(q)

    q2 = axa_to_q(axa)
    q3 = rpy_to_q(rpy)

    R  = np.float32([tx.quaternion_matrix(qq) for qq in q])
    R2 = np.float32([tx.quaternion_matrix(qq) for qq in q2])
    R3 = np.float32([tx.quaternion_matrix(qq) for qq in q3])

    d2 = np.square(R-R2).sum(axis=(1,2))
    d3 = np.square(R-R3).sum(axis=(1,2))

    i2 = np.argmax(d2)
    i3 = np.argmax(d3)

    print (q[i2], q2[i2], d2[i2])
    print (q[i3], q2[i3], d2[i3])

if __name__ == '__main__':
    main()
