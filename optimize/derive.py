#!/usr/bin/env python2
import numpy as np
import sympy as sm
from sympy import Q, AppliedPredicate
from sympy.assumptions.assume import global_assumptions
import sys

def rotate(rxn, lmk):
    Rx = sm.rot_axis1(rxn[0]).T
    Ry = sm.rot_axis2(rxn[1]).T
    Rz = sm.rot_axis3(rxn[2]).T
    return Rz*Ry*Rx*lmk

def rotate_axa(rxn, lmk):
    # axis-angle (rodrigues) rotation
    global_assumptions.add(Q.real(rxn))
    global_assumptions.add(Q.real(lmk))

    v = lmk
    hsq = (rxn.T * rxn)[0,0]
    h = sm.sqrt(hsq)
    u = rxn / h
    c = sm.cos( h )
    s = sm.sin( h )
    d = sm.ones(1,3) * u.multiply_elementwise(v)
    p1 = (v*c)
    p2 = s * u.cross(v)
    p3 = (1.-c)*(d[0,0])*u
    s = p1+p2+p3
    return s

def test_conversions():
    x,y,z = sm.symbols('x,y,z')
    R,P,Y = RPY = sm.symbols('R,P,Y') # euler
    rx,ry,rz = AXA = sm.symbols('rx,ry,rz') # rodrigues

    M_xyz = sm.Matrix([x,y,z]).T
    M_rpy = sm.Matrix([R,P,Y]).T
    M_axa = sm.Matrix([rx,ry,rz]).T

    v1 = rotate([R,P,Y], M_xyz.T)
    v2 = rotate_axa(M_axa.T, M_xyz.T)
    v1 = sm.simplify(v1)
    v2 = sm.simplify(v2)

    sol = sm.solve(v1-v2, [rx,ry,rz])
    print sol
    sys.exit(0)


test_conversions()

def project(txn, rxn, lmk, K):
    #pt3 = rotate(rxn, lmk) + txn
    pt3 = rotate_axa(rxn, lmk) + txn
    fx,fy,cx,cy = np.array(K)[(0,1,0,1),(0,1,2,2)]
    pth = K * pt3
    return pth[0:2, 0] / pth[2, 0]

    px_x = cx + fx * (pt3[0] / pt3[2])
    px_y = cy + fy * (pt3[1] / pt3[2])
    p0 = px_x
    px_x = pth[0,0] / pth[2,0]
    p1 = px_x
    #print sm.simplify(p1 - p0)
    px_y = pth[1,0] / pth[2,0]
    return (sm.Matrix([px_x, px_y]))

def main():
    txn = sm.symbols('x,y,z')
    rxn = sm.symbols('R,P,Y')
    lmk = sm.symbols('px,py,pz')

    fx, fy, cx, cy = sm.symbols('fx,fy,cx,cy')

    ksym = [
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
            ]
    #for i in range(3):
    #    ksym.append([])
    #    for j in range(3):
    #        ksym[-1].append(sm.Symbol('K{}{}'.format(i,j)))
    K   = sm.Matrix(ksym)
    M_txn = sm.Matrix([txn]).T
    M_rxn = sm.Matrix([rxn]).T
    M_lmk = sm.Matrix([lmk]).T
    #res = project(txn, rxn, M_lmk)
    pt = project(M_txn, M_rxn, M_lmk, K)
    #print 'return {}'.format(pt).replace('Matrix','').replace('sin','np.sin').replace('cos','np.cos')

    print '======='
    jac = pt.jacobian(txn+rxn+lmk)
    #sm.pprint(jac)

    agg = pt.row_join(jac)
    ##sm.simplify(agg)
    rdc1, cex1 = sm.cse(agg, order='none', optimizations='basic')
    for ex,v in rdc1:
        s = '{}={}'.format(ex,v)
        print s.replace('cos','np.cos').replace('sin','np.sin')
    cex1 = cex1[0]

    names = 'point', 'j_txn', 'j_rxn', 'j_lmk'
    exs   = [cex1[:,0], cex1[:, 1:4], cex1[:, 4:7], cex1[:, 7:10]]

    for name, ex in zip(names, exs):
        s = '{}={}'.format(name, ex)
        print s.replace('Matrix', '')

if __name__ == '__main__':
    main()
