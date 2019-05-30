#!/usr/bin/env python2

import numpy as np
import cv2
from cho_util import vmath as vm
from tf import transformations as tx

def project(txn, rxn, lmk, K, np=np):
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

    point = ([x20*x8, x41*x8])
    j_txn = ([[x21, c0, cx*x8 - x23], [c0, x42, cy*x8 - x43]])
    j_rxn = ([[x23*x27 + x8*(cx*x28 + fx*(py*x14 + pz*(x15 - x17))), x23*x31 + x8*(cx*x34 + fx*(-x1*x9 + x13*x33 + x16*x32)), x21*(py*x37 + pz*(x16 - x38) - x35)], [x27*x43 + x8*(cy*x28 + fy*(py*x40 + pz*x37)), x31*x43 + x8*(cy*x34 + fy*(-x1*x11 + x12*x32 + x15*x33)), x19*x42]])
    j_lmk = ([[x0*x23 + x8*(-cx*x0 + fx*x2*x9), -x23*x6 + x8*(cx*x6 + fx*x18), -x23*x25 + x8*(cx*x25 + fx*x14)], [x0*x43 + x8*(-cy*x0 + fy*x11*x2), -x43*x6 + x8*(fy*x39 + x44*x5), -x25*x43 + x8*(fy*x40 + x3*x44)]])

    return point, j_txn, j_rxn, j_lmk

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
