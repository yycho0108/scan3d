#!/usr/bin/env python2
import numpy as np
import sympy as sm

def rotate(rxn, lmk):
    Rx = sm.rot_axis1(rxn[0]).T
    print Rx
    Ry = sm.rot_axis2(rxn[1]).T
    Rz = sm.rot_axis3(rxn[2]).T
    return Rz*Ry*Rx*lmk

def project(txn, rxn, lmk, K):
    pt3 = rotate(rxn, lmk) + txn
    fx,fy,cx,cy = np.array(K)[(0,1,0,1),(0,1,2,2)]
    pth = K * pt3
    return pth[0:2, 0] / pth[2, 0]

    px_x = cx + fx * (pt3[0] / pt3[2])
    px_y = cy + fy * (pt3[1] / pt3[2])
    p0 = px_x
    px_x = pth[0,0] / pth[2,0]
    p1 = px_x
    print sm.simplify(p1 - p0)
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
    M_lmk = sm.Matrix([lmk]).T
    #res = project(txn, rxn, M_lmk)
    pt = project(M_txn, rxn, M_lmk, K)

    #print 'return {}'.format(pt).replace('Matrix','').replace('sin','np.sin').replace('cos','np.cos')

    print '======='
    jac = pt.jacobian(txn+rxn+lmk)
    #sm.pprint(jac)

    agg = pt.row_join(jac)
    sm.simplify(agg)
    rdc1, cex1 = sm.cse(sm.simplify(agg), order='none')
    for ex,v in rdc1:
        print '{}={}'.format(ex,v)
    cex1 = cex1[0]

    print cex1[:, 0]
    print cex1[:, 1:4]
    print cex1[:, 4:7]
    print cex1[:, 7:10]

if __name__ == '__main__':
    main()
