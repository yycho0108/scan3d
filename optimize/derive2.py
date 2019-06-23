#!/usr/bin/env python2
import numpy as np
import sympy as sm
from sympy import Q, AppliedPredicate
from sympy.assumptions.assume import global_assumptions
import sys
from sympy.parsing.sympy_parser import parse_expr

def rotate_quat(rxn, lmk):
    # axis-angle (rodrigues) rotation
    global_assumptions.add(Q.real(rxn))
    global_assumptions.add(Q.real(lmk))

    u = rxn[:3, :]
    s = rxn[3, :]
    v = lmk

    a = u.dot(v) * u
    b = (s.dot(s) - u.dot(u)) * v
    c = 2 * s[0,0] * (u.cross(v))
    x =  a + b + c
    x = sm.simplify(x)
    return sm.Matrix(x)

def project(txn, rxn, pt, inv_d, K):
    pt3 = rotate_quat(rxn, lmk) + txn
    pth = K * pt3
    return pth[0:2, 0] / pth[2, 0]

qx,qy,qz,qw = sm.symbols('qx,qy,qz,qw')
x,y,z = sm.symbols('x,y,z')

q = sm.Matrix([qx,qy,qz,qw])
l = sm.Matrix([x,y,z])
lr = rotate_quat(q, l)

rdc, cex = sm.cse(lr, order='none', optimizations='basic')
for ex, v in rdc:
    s = '{}={}'.format(ex, v)
    print s.replace('cos', 'np.cos').replace('sin', 'np.sin')
print cex
