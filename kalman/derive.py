import sympy as sm
import numpy as np


iX,iY,iZ,iVx,iVy,iVz,iAx,iAy,iAz,iRx,iRy,iRz,iWx,iWy,iWz = range(15)

def fx(state, delta):
    roll     = state[ iRx ]
    pitch    = state[ iRy ]
    yaw      = state[ iRz ]

    xVel     = state[ iVx ]
    yVel     = state[ iVy ]
    zVel     = state[ iVz ]

    xAcc     = state[ iAx ]
    yAcc     = state[ iAy ]
    zAcc     = state[ iAz ]

    # cached variables
    sp = sm.sin(pitch)
    cp = sm.cos(pitch)
    cpi = 1.0 / cp
    tp = sp * cpi
    sr = sm.sin(roll)
    cr = sm.cos(roll)
    sy = sm.sin(yaw)
    cy = sm.cos(yaw)

    M = sm.Matrix(sm.eye(15))

    # Prepare the transfer function
    M[ iX, iVx ] = cy * cp * delta
    M[ iX, iVy ] = (cy * sp * sr - sy * cr) * delta
    M[ iX, iVz ] = (cy * sp * cr + sy * sr) * delta
    M[ iX, iAx ] = 0.5 * M[ iX, iVx ] * delta
    M[ iX, iAy ] = 0.5 * M[ iX, iVy ] * delta
    M[ iX, iAz ] = 0.5 * M[ iX, iVz ] * delta
    M[ iY, iVx ] = sy * cp * delta
    M[ iY, iVy ] = (sy * sp * sr + cy * cr) * delta
    M[ iY, iVz ] = (sy * sp * cr - cy * sr) * delta
    M[ iY, iAx ] = 0.5 * M[ iY, iVx ] * delta
    M[ iY, iAy ] = 0.5 * M[ iY, iVy ] * delta
    M[ iY, iAz ] = 0.5 * M[ iY, iVz ] * delta
    M[ iZ, iVx ] = -sp * delta
    M[ iZ, iVy ] = cp * sr * delta
    M[ iZ, iVz ] = cp * cr * delta
    M[ iZ, iAx ] = 0.5 * M[ iZ, iVx ] * delta
    M[ iZ, iAy ] = 0.5 * M[ iZ, iVy ] * delta
    M[ iZ, iAz ] = 0.5 * M[ iZ, iVz ] * delta
    M[ iRx, iWx ] = delta
    M[ iRx, iWy ] = sr * tp * delta
    M[ iRx, iWz ] = cr * tp * delta
    M[ iRy, iWy ] = cr * delta
    M[ iRy, iWz ] = -sr * delta
    M[ iRz, iWy ] = sr * cpi * delta
    M[ iRz, iWz ] = cr * cpi * delta
    M[ iVx, iAx ] = delta
    M[ iVy, iAy ] = delta
    M[ iVz, iAz ] = delta

    return M

sym_s = sm.symbols('x,y,z,vx,vy,vz,ax,ay,az,rx,ry,rz,wx,wy,wz')
delta = sm.Symbol('dt')
state = sm.Matrix([sym_s]).T
trans  = fx(state, delta)

print trans.shape
print state.shape
tjac = (trans * state).jacobian(sym_s)
#print 'trans'
#print trans
#print 'tjac'
#print tjac

rdc1, cex1 = sm.cse(trans.col_join(tjac - trans), order='none')
for ex,v in rdc1:
    print '{}={}'.format(ex,v)
cex1 = cex1[0]
trans_ex = cex1[:15,:]
tjac_ex =  cex1[15:,:]

print '==='
print sm.simplify(trans_ex)
print '==='
print sm.simplify(tjac_ex)
#print '==='
#print cex1[1]
