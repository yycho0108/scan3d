#!/usr/bin/env python2

import numpy as np
import cv2
from cho_util import vmath as vm
from tf import transformations as tx

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
    x1=px*x0
    x2=np.cos(P)
    x3=np.cos(R)
    x4=pz*x3
    x5=np.sin(R)
    x6=x2*x5
    x7=py*x6 - x1 + x2*x4 + z
    x8=1./x7
    x9=np.cos(Y)
    x10=px*x2
    x11=np.sin(Y)
    x12=x11*x5
    x13=x3*x9
    x14=x0*x13 + x12
    x15=x11*x3
    x16=x5*x9
    x17=x0*x16
    x18=-x15 + x17
    x19=py*x18 + pz*x14 + x10*x9
    x20=cx*x7 + fx*(x + x19)
    x21=fx*x8
    x22=x7**(-2)
    x23=x20*x22
    x24=pz*x6
    x25=x2*x3
    x26=py*x25
    x27=x24 - x26
    x28=-x24 + x26
    x29=x0*x4
    x30=py*x0*x5
    x31=x10 + x29 + x30
    x32=py*x2
    x33=pz*x2
    x34=-x10 - x29 - x30
    x35=x10*x11
    x36=x0*x12
    x37=-x13 - x36
    x38=x0*x15
    x39=x13 + x36
    x40=-x16 + x38
    x41=cy*x7 + fy*(py*x39 + pz*x40 + x35 + y)
    x42=fy*x8
    x43=x22*x41
    x44=cy*x2
    if (not jac):
        point = ([x20*x8, x41*x8])
        return point
    else:
        j_txn = ([[x21, c0, cx*x8 - x23], [c0, x42, cy*x8 - x43]])
        j_rxn = ([[x23*x27 + x8*(cx*x28 + fx*(py*x14 + pz*(x15 - x17))), x23*x31 + x8*(cx*x34 + fx*(-x1*x9 + x13*x33 + x16*x32)), x21*(py*x37 + pz*(x16 - x38) - x35)], [x27*x43 + x8*(cy*x28 + fy*(py*x40 + pz*x37)), x31*x43 + x8*(cy*x34 + fy*(-x1*x11 + x12*x32 + x15*x33)), x19*x42]])
        j_lmk = ([[x0*x23 + x8*(-cx*x0 + fx*x2*x9), -x23*x6 + x8*(cx*x6 + fx*x18), -x23*x25 + x8*(cx*x25 + fx*x14)], [x0*x43 + x8*(-cy*x0 + fy*x11*x2), -x43*x6 + x8*(fy*x39 + x44*x5), -x25*x43 + x8*(fy*x40 + x3*x44)]])
        return j_txn, j_rxn, j_lmk

def main():
    K = np.float32([
        500, 0, 320,
        0, 500, 240,
        0, 0, 1
        ]).reshape(3,3)

    txn = np.random.normal(size=(5,3))
    rxn = np.random.normal(size=(5,3))
    lmk = np.random.normal(size=(5,3))
    #lmk[..., 2] = np.abs(lmk[..., 2])

    point, j_txn, j_rxn, j_lmk = project(txn,rxn,lmk,K)
    
    print 'j_txn', j_txn
    # print 'j_rxn', j_rxn

    #print np.transpose(project(txn,rxn,lmk, K)[0])
    rvec = [cv2.Rodrigues(tx.euler_matrix(*r)[:3,:3])[0] for r in rxn]
    tvec = txn

    #Tis = [tx.inverse_matrix(tx.compose_matrix(translate=t, angles=r)) for(t,r) in zip(txn, rxn)]
    #rvec = [cv2.Rodrigues(T[:3,:3])[0] for T in Tis]
    #tvec = [T[:3,3:].ravel() for T in Tis]

    cv_res = [cv2.projectPoints((l[None,:]), r, t, K, None) for (l,r,t) in zip(lmk, rvec, tvec)]
    cv_j = np.float32([c[1] for c in cv_res])

    j_rxn = cv_j[:, :, 0:3]
    j_txn = cv_j[:, :, 3:6]
    j_cam = cv_j[:, :, 6:10]

    print j_txn
    # print j_rxn
    # print j_cam
    #j_lmk 

if __name__ == '__main__':
    main()
