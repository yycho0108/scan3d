#!/usr/bin/env python2

import numpy as np
import cv2
from cho_util import vmath as vm
from tf import transformations as tx

def rotate_axa(rxn, lmk):
    # axis-angle (rodrigues) rotation
    v = lmk
    h = np.linalg.norm(rxn, axis=-1, keepdims=True)
    with np.errstate(invalid='ignore'):
        u = np.nan_to_num(rxn / h)
    c, s  = np.cos(h), np.sin(h)
    d = (u*v).sum(axis=-1, keepdims=True)
    return (v*c) + s*np.cross(u,v) + (1.-c)*(d) * u

class Projector(object):
    def __init__(self, txn, rxn, lmk, K):
        self.data_ = dict(
                txn=txn,
                rxn=rxn,
                lmk=lmk,
                K=K)

    def __getitem__(self, name):
        """ avoid repeating boilerplate @property code """
        if name not in self.data_:
            # compute cache
            fun = getattr(self, name, None)
            if fun is None:
                msg = 'Attempting to access an invalid compute path : {}'.format(name)
                raise ValueError(msg)
            self.data_[name] = fun()
        # return cache
        return self.data_[name]

    def fx(_):
        return _['K'][0,0]
    def fy(_):
        return _['K'][1,1]
    def cx(_):
        return _['K'][0,2]
    def cy(_):
        return _['K'][1,2]
    def x(_):
        return _['txn'][..., 0]
    def y(_):
        return _['txn'][..., 1]
    def z(_):
        return _['txn'][..., 2]
    def R(_):
        return _['rxn'][..., 0]
    def P(_):
        return _['rxn'][..., 1]
    def Y(_):
        return _['rxn'][..., 2]
    def px(_):
        return _['lmk'][..., 0]
    def py(_):
        return _['lmk'][..., 1]
    def pz(_):
        return _['lmk'][..., 2]
    def c0(_):
        return np.zeros_like(_['x'])
    def x0(_):
        return np.sin(_[ 'P' ])
    def x1(_):
        return _[ 'px' ]*_[ 'x0' ]
    def x2(_):
        return np.cos(_[ 'P' ])
    def x3(_):
        return np.cos(_[ 'R' ])
    def x4(_):
        return _[ 'pz' ]*_[ 'x3' ]
    def x5(_):
        return np.sin(_[ 'R' ])
    def x6(_):
        return _[ 'x2' ]*_[ 'x5' ]
    def x7(_):
        return _[ 'py' ]*_[ 'x6' ] - _[ 'x1' ] + _[ 'x2' ]*_[ 'x4' ] + _[ 'z' ]
    def x8(_):
        return 1./_[ 'x7' ]
    def x9(_):
        return np.cos(_[ 'Y' ])
    def x10(_):
        return _[ 'px' ]*_[ 'x2' ]
    def x11(_):
        return np.sin(_[ 'Y' ])
    def x12(_):
        return _[ 'x11' ]*_[ 'x5' ]
    def x13(_):
        return _[ 'x3' ]*_[ 'x9' ]
    def x14(_):
        return _[ 'x0' ]*_[ 'x13' ] + _[ 'x12' ]
    def x15(_):
        return _[ 'x11' ]*_[ 'x3' ]
    def x16(_):
        return _[ 'x5' ]*_[ 'x9' ]
    def x17(_):
        return _[ 'x0' ]*_[ 'x16' ]
    def x18(_):
        return -_[ 'x15' ] + _[ 'x17' ]
    def x19(_):
        return _[ 'py' ]*_[ 'x18' ] + _[ 'pz' ]*_[ 'x14' ] + _[ 'x10' ]*_[ 'x9' ]
    def x20(_):
        return _[ 'cx' ]*_[ 'x7' ] + _[ 'fx' ]*(_[ 'x' ] + _[ 'x19' ])
    def x21(_):
        return _[ 'fx' ]*_[ 'x8' ]
    def x22(_):
        return _[ 'x7' ]**(-2)
    def x23(_):
        return _[ 'x20' ]*_[ 'x22' ]
    def x24(_):
        return _[ 'pz' ]*_[ 'x6' ]
    def x25(_):
        return _[ 'x2' ]*_[ 'x3' ]
    def x26(_):
        return _[ 'py' ]*_[ 'x25' ]
    def x27(_):
        return _[ 'x24' ] - _[ 'x26' ]
    def x28(_):
        return -_[ 'x24' ] + _[ 'x26' ]
    def x29(_):
        return _[ 'x0' ]*_[ 'x4' ]
    def x30(_):
        return _[ 'py' ]*_[ 'x0' ]*_[ 'x5' ]
    def x31(_):
        return _[ 'x10' ] + _[ 'x29' ] + _[ 'x30' ]
    def x32(_):
        return _[ 'py' ]*_[ 'x2' ]
    def x33(_):
        return _[ 'pz' ]*_[ 'x2' ]
    def x34(_):
        return -_[ 'x10' ] - _[ 'x29' ] - _[ 'x30' ]
    def x35(_):
        return _[ 'x10' ]*_[ 'x11' ]
    def x36(_):
        return _[ 'x0' ]*_[ 'x12' ]
    def x37(_):
        return -_[ 'x13' ] - _[ 'x36' ]
    def x38(_):
        return _[ 'x0' ]*_[ 'x15' ]
    def x39(_):
        return _[ 'x13' ] + _[ 'x36' ]
    def x40(_):
        return -_[ 'x16' ] + _[ 'x38' ]
    def x41(_):
        return _[ 'cy' ]*_[ 'x7' ] + _[ 'fy' ]*(_[ 'py' ]*_[ 'x39' ] + _[ 'pz' ]*_[ 'x40' ] + _[ 'x35' ] + _[ 'y' ])
    def x42(_):
        return _[ 'fy' ]*_[ 'x8' ]
    def x43(_):
        return _[ 'x22' ]*_[ 'x41' ]
    def x44(_):
        return _[ 'cy' ]*_[ 'x2' ]
    def point (_):
        return [_['x20'] * _['x8'], _['x41'] * _['x8']]
    def j_txn (_):
        return ([[_[ 'x21' ], _[ 'c0' ], _[ 'cx' ]*_[ 'x8' ] - _[ 'x23' ]], [_[ 'c0' ], _[ 'x42' ], _[ 'cy' ]*_[ 'x8' ] - _[ 'x43' ]]])
    def j_rxn (_):
        return ([[_[ 'x23' ]*_[ 'x27' ] + _[ 'x8' ]*(_[ 'cx' ]*_[ 'x28' ] + _[ 'fx' ]*(_[ 'py' ]*_[ 'x14' ] + _[ 'pz' ]*(_[ 'x15' ] - _[ 'x17' ]))), _[ 'x23' ]*_[ 'x31' ] + _[ 'x8' ]*(_[ 'cx' ]*_[ 'x34' ] + _[ 'fx' ]*(-_[ 'x1' ]*_[ 'x9' ] + _[ 'x13' ]*_[ 'x33' ] + _[ 'x16' ]*_[ 'x32' ])), _[ 'x21' ]*(_[ 'py' ]*_[ 'x37' ] + _[ 'pz' ]*(_[ 'x16' ] - _[ 'x38' ]) - _[ 'x35' ])], [_[ 'x27' ]*_[ 'x43' ] + _[ 'x8' ]*(_[ 'cy' ]*_[ 'x28' ] + _[ 'fy' ]*(_[ 'py' ]*_[ 'x40' ] + _[ 'pz' ]*_[ 'x37' ])), _[ 'x31' ]*_[ 'x43' ] + _[ 'x8' ]*(_[ 'cy' ]*_[ 'x34' ] + _[ 'fy' ]*(-_[ 'x1' ]*_[ 'x11' ] + _[ 'x12' ]*_[ 'x32' ] + _[ 'x15' ]*_[ 'x33' ])), _[ 'x19' ]*_[ 'x42' ]]])
    def j_lmk (_):
        return ([[_[ 'x0' ]*_[ 'x23' ] + _[ 'x8' ]*(-_[ 'cx' ]*_[ 'x0' ] + _[ 'fx' ]*_[ 'x2' ]*_[ 'x9' ]), -_[ 'x23' ]*_[ 'x6' ] + _[ 'x8' ]*(_[ 'cx' ]*_[ 'x6' ] + _[ 'fx' ]*_[ 'x18' ]), -_[ 'x23' ]*_[ 'x25' ] + _[ 'x8' ]*(_[ 'cx' ]*_[ 'x25' ] + _[ 'fx' ]*_[ 'x14' ])], [_[ 'x0' ]*_[ 'x43' ] + _[ 'x8' ]*(-_[ 'cy' ]*_[ 'x0' ] + _[ 'fy' ]*_[ 'x11' ]*_[ 'x2' ]), -_[ 'x43' ]*_[ 'x6' ] + _[ 'x8' ]*(_[ 'fy' ]*_[ 'x39' ] + _[ 'x44' ]*_[ 'x5' ]), -_[ 'x25' ]*_[ 'x43' ] + _[ 'x8' ]*(_[ 'fy' ]*_[ 'x40' ] + _[ 'x3' ]*_[ 'x44' ])]])

    def compute(_, np=np, jac=False, data={}):
        if jac:
            res =  _['j_txn'], _['j_rxn'], _['j_lmk']
            data.update(_.data_)
            return res
        else:
            res = _['point']
            data.update(_.data_)
            return res

def project(txn, rxn, lmk, K, np=np, jac=False):
    # unroll
    fx,fy,cx,cy = K[(0,1,0,1),(0,1,2,2)]
    x,y,z=[txn[...,i] for i in range(3)]
    R,P,Y=[rxn[...,i] for i in range(3)]
    px,py,pz=[lmk[...,i] for i in range(3)]
    c0 = np.zeros_like(x)

    x0=np.sin(P)
    x1=np.cos(P)
    x2=np.cos(R)
    x3=x1*x2
    x4=np.sin(R)
    x5=py*x4
    x6=-px*x0 + pz*x3 + x1*x5
    x7=x6 + z
    x8=1/x7
    x9=np.cos(Y)
    x10=px*x1
    x11=np.sin(Y)
    x12=x11*x4
    x13=x2*x9
    x14=x0*x13 + x12
    x15=x11*x2
    x16=x4*x9
    x17=x0*x16 - x15
    x18=py*x17 + pz*x14 + x10*x9
    x19=x8*(cx*x7 + fx*(x + x18))
    x20=fx*x8
    x21=x1*(py*x2 - pz*x4)
    x22=pz*x0*x2 + x0*x5 + x10
    x23=fx*x9
    x24=x0*x12 + x13
    x25=x0*x15 - x16
    x26=py*x24 + pz*x25 + x10*x11
    x27=x1*x4
    x28=x8*(cy*x7 + fy*(x26 + y))
    x29=fy*x8
    x30=fy*x11

    if (not jac):
        point = [x19, x28]
        return point
    else:
        j_txn = [[x20, c0, x8*(cx - x19)], [c0, x29, x8*(cy - x28)]]
        j_rxn = [[x8*(cx*x21 + fx*(py*x14 - pz*x17) - x19*x21), x8*(-cx*x22 + x19*x22 + x23*x6), -x20*x26], [x8*(cy*x21 + fy*(py*x25 - pz*x24) - x21*x28), x8*(-cy*x22 + x22*x28 + x30*x6), x18*x29]]
        j_lmk = [[x8*(-cx*x0 + x0*x19 + x1*x23), x8*(cx*x27 + fx*x17 - x19*x27), x8*(cx*x3 + fx*x14 - x19*x3)], [x8*(-cy*x0 + x0*x28 + x1*x30), x8*(cy*x27 + fy*x24 - x27*x28), x8*(cy*x3 + fy*x25 - x28*x3)]]
        return j_txn, j_rxn, j_lmk

def project_axa(txn, rxn, lmk, K, np=np, jac=False):
    # unroll
    fx,fy,cx,cy = K[(0,1,0,1),(0,1,2,2)]
    x,y,z=[txn[...,i] for i in range(3)]
    R,P,Y=[rxn[...,i] for i in range(3)] # NOTE: not really Roll/Pitch/Yaw
    px,py,pz=[lmk[...,i] for i in range(3)]
    c0 = np.zeros_like(x)

    x0=P**2
    x1=R**2
    x2=Y**2
    x3=x0 + x1 + x2
    x4=np.sqrt(x3)
    x5=np.cos(x4)
    x6=1./x3
    x7=x5 - 1.0
    x8=Y*pz
    x9=P*py
    x10=R*px
    x11=x10 + x8 + x9
    x12=x11*x6*x7
    x13=P*px
    x14=R*py
    x15=x13 - x14
    x16=np.sin(x4)
    x17=x16/x4
    x18=Y*x12 - pz*x5 + x15*x17 - z
    x19=1./x18
    x20=P*pz
    x21=Y*py
    x22=x20 - x21
    x23=x19*(-cx*x18 + fx*(-R*x12 + px*x5 + x + x17*x22))
    x24=R*pz
    x25=R*x6
    x26=x25*x9
    x27=x1*x6
    x28=x25*x8
    x29=px*x27 - px + x26 + x28
    x30=Y*x6*x7
    x31=R*x5*x6
    x32=x13*x25
    x33=x11*x7/x3**2
    x34=R*Y
    x35=x3**(-3/2)
    x36=x11*x16*x35
    x37=x33*x34 + x34*x36
    x38=-x15*x31 - x17*x24 + x17*(-py*x27 + py + x32) + x29*x30 + x37
    x39=x16*x35
    x40=R*x6*x7
    x41=-x12
    x42=P*x5*x6
    x43=P*x6*x8
    x44=x0*x6
    x45=py*x44 - py + x32 + x43
    x46=P*Y
    x47=x33*x46 + x36*x46
    x48=-x15*x42 - x17*x20 - x17*(-px*x44 + px + x26) + x30*x45 + x47
    x49=Y*x6*x9
    x50=P*R
    x51=x33*x50 + x36*x50
    x52=x2*x6
    x53=Y*x5*x6
    x54=Y*px
    x55=x25*x54
    x56=pz*x52 - pz + x49 + x55
    x57=Y*x15*x39 - x15*x53 - x17*x8 + x2*x33 + x2*x36 + x30*x56 + x41
    x58=P*x17
    x59=Y*x40
    x60=x58 + x59
    x61=R*x17
    x62=P*x6*x7
    x63=Y*x62
    x64=-x61 + x63
    x65=Y*x17
    x66=P*x40
    x67=-x5
    x68=x52*x7 + x67
    x69=x24 - x54
    x70=x19*(cy*x18 + fy*(P*x12 - py*x5 + x17*x69 - y))

    if not jac:
        point=[-x23, x70]
        return point
    else:
        j_txn=[[-fx*x19, c0, -x19*(cx + x23)], [c0, -fy*x19, x19*(-cy + x70)]]
        j_rxn=[[-x19*(cx*x38 + fx*(-R*x22*x39 + x1*x33 + x1*x36 - x10*x17 + x22*x31 + x29*x40 + x41) + x23*x38), -x19*(cx*x48 + fx*(-x13*x17 + x17*(-pz*x44 + pz + x49) + x22*x42 + x40*x45 + x51) + x23*x48), -x19*(cx*x57 + fx*(-x17*x54 - x17*(-py*x52 + py + x43) + x22*x53 + x37 + x40*x56) + x23*x57)], [x19*(-cy*x38 - fy*(-x14*x17 - x17*(-pz*x27 + pz + x55) + x29*x62 - x31*x69 + x51) + x38*x70), x19*(-cy*x48 - fy*(P*x39*x69 + x0*x33 + x0*x36 - x17*x9 + x41 - x42*x69 + x45*x62) + x48*x70), x19*(-cy*x57 - fy*(-x17*x21 + x17*(-px*x52 + px + x28) + x47 - x53*x69 + x56*x62) + x57*x70)]]
        j_lmk=[[x19*(cx*x60 - fx*(-x27*x7 + x5) + x23*x60), x19*(cx*x64 + fx*(x65 + x66) + x23*x64), x19*(cx*x68 - fx*(x58 - x59) + x23*x68)], [x19*(cy*x60 - fy*(x65 - x66) - x60*x70), x19*(cy*x64 + fy*(x44*x7 + x67) - x64*x70), x19*(cy*x68 + fy*(x61 + x63) - x68*x70)]]

        return j_txn, j_rxn, j_lmk


def main():
    #np.random.seed(1)
    K = np.float32([
        500, 0, 320,
        0, 500, 240,
        0, 0, 1
        ]).reshape(3,3)

    txn = 0.1 * np.random.normal(size=(1,3))
    rxn = 0.1 * np.random.normal(size=(1,3))
    lmk = np.random.normal(size=(1,3))
    #lmk[..., 2] = np.abs(lmk[..., 2])

    point = project_axa(txn,rxn,lmk,K, jac=False)
    j_txn, j_rxn, j_lmk = project_axa(txn,rxn,lmk,K, jac=True)

    print 'point', np.float32(point).ravel()
    print 'j_txn', np.float32(j_txn).transpose(2,0,1)
    print 'j_rxn', np.float32(j_rxn).transpose(2,0,1)
    #print 'j_lmk', np.float32(j_lmk).transpose(2,0,1)

    #print np.transpose(project(txn,rxn,lmk, K)[0])
    #rvec = [cv2.Rodrigues(tx.euler_matrix(*r)[:3,:3])[0] for r in rxn]
    rvec = rxn
    tvec = txn

    #Tis = [tx.inverse_matrix(tx.compose_matrix(translate=t, angles=r)) for(t,r) in zip(txn, rxn)]
    #rvec = [cv2.Rodrigues(T[:3,:3])[0] for T in Tis]
    #tvec = [T[:3,3:].ravel() for T in Tis]

    cv_res = [cv2.projectPoints((l[None,:]), r, t, K, None) for (l,r,t) in zip(lmk, rvec, tvec)]
    cv_pt = np.float32([c[0] for c in cv_res])
    cv_j = np.float32([c[1] for c in cv_res])

    j_rxn = cv_j[:, :, 0:3]
    j_txn = cv_j[:, :, 3:6]
    j_cam = cv_j[:, :, 6:10]

    print 'point', np.float32(cv_pt)
    print 'j_txn', j_txn
    print 'j_rxn', j_rxn
    # print j_cam
    #j_lmk 

if __name__ == '__main__':
    main()
