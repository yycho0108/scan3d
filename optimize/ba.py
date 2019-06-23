#!/usr/bin/env python2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix, csr_matrix
from cho_util import vmath as vm
from cho_util.math import transform as tx
from cho_util import math as cm
from autograd import jacobian
from autograd import numpy as anp
from .expr import project, project_axa, project_invd, project_axa_invd

from profilehooks import profile
from .rot import axa_to_q, q_to_axa, rpy_to_q, q_to_rpy

class BundleAdjustment(object):
    """
    Bundle Adjustment!

    """
    def __init__(self,
            i_src, i_lmk, p_obs,
            txn, rxn, lmk,
            K,
            pose_only=False,
            invd=False,
            axa=False
            ):
        self.crit_ = dict(
                ftol=1e-4,
                xtol=1e-4,
                loss='cauchy',
                max_nfev=1024,
                method='trf',
                verbose=2,
                tr_solver='lsmr',
                f_scale=np.sqrt(5.991)
                )
        self.pose_only_ = pose_only
        self.invd_      = False if pose_only else invd
        self.axa_       = axa

        # input - observation
        self.i_src_ = i_src
        self.i_lmk_ = i_lmk
        self.p_obs_ = p_obs

        # input - data (initial)
        self.txn_   = txn
        self.rxn_   = rxn
        self.lmk_   = lmk

        # input - parameters
        self.K_ = K
        self.d_lmk_ = (4 if self.invd_ else 3)
        self.d_txn_ = (3)
        self.d_rxn_ = (3) # TODO : maybe use quat?
        self.d_obs_ = (2)

        # derived data
        self.n_src_ = 1+np.max(i_src, initial=-1)
        self.n_lmk_ = 1+np.max(i_lmk, initial=-1)
        self.n_obs_ = len(self.p_obs_)

        # simple checks
        assert self.n_src_ == len(self.txn_), '{}!={}'.format(self.n_src_, len(self.txn_))
        assert(self.n_src_ == len(self.rxn_))
        assert(self.n_lmk_ == len(self.lmk_))

        # automatic differentiation?
        #self.jacobian = jacobian(lambda x : self.residual(x, np=anp))

    def invert(self, txn, rxn):
        Ti = tx.invert(tx.compose(r=rxn, t=txn, rtype=tx.rotation.euler))

        txn = tx.translation_from_matrix(Ti)
        rxn = tx.rotation.euler.from_matrix(Ti)

        txn = np.float32(txn)
        rxn = np.float32(rxn)
        return txn, rxn

    #@staticmethod
    def roll(self, txn, rxn, lmk):
        if self.pose_only_:
            return np.concatenate([txn.ravel(), rxn.ravel()])
        return np.concatenate([txn.ravel(), rxn.ravel(), lmk.ravel()])

    #@staticmethod
    def unroll(self, params, n_src, n_lmk):
        i0 = 0
        i1 = (i0 + self.d_txn_ * n_src)
        i2 = (i1 + self.d_rxn_ * n_src)
        txn = params[i0:i1].reshape(-1,self.d_txn_)
        rxn = params[i1:i2].reshape(-1,self.d_rxn_)

        if self.pose_only_:
            return txn, rxn, self.lmk_

        i3 = (i2 + self.d_lmk_ * n_lmk)
        lmk = params[i2:i3].reshape(-1,self.d_lmk_)
        return txn, rxn, lmk

    #@staticmethod
    def parametrize(self, txn, rxn, lmk):
        # RPY -> Rodrigues
        if self.axa_:
            rxn = q_to_axa(rpy_to_q(rxn))
        if self.invd_:
            # convert to inverse depth parametrization, subject to norm==1
            # but skip viewpoint thingy and see how it goes ... ?
            lmk = tx.to_homogeneous(lmk)
            lmk /= cm.norm(lmk, keepdims=True)
        return txn, rxn, lmk

    #@staticmethod
    def unparametrize(self, txn, rxn, lmk):
        # Rodrigues -> RPY
        if self.axa_:
            rxn = q_to_rpy(axa_to_q(rxn))
        if self.invd_:
            lmk = tx.from_homogeneous(lmk)
        return txn, rxn, lmk

    def jacobian(self, params):
        txn, rxn, lmk = self.unroll(params, self.n_src_, self.n_lmk_)
        jac_txn, jac_rxn, jac_lmk = self.project(params, jac=True)

        n_out = self.n_obs_*self.d_obs_
        n_in  = self.n_src_*(self.d_rxn_ + self.d_txn_)
        if not self.pose_only_:
            n_in += self.n_lmk_*(self.d_lmk_)
        
        j_shape = (n_out, n_in)

        jac_txn = np.transpose(jac_txn, (2,0,1))
        jac_rxn = np.transpose(jac_rxn, (2,0,1))
        jac_lmk = np.transpose(jac_lmk, (2,0,1))

        i_obs = np.arange(self.n_obs_)

        r_i = []
        c_i = []
        v   = [] 
        for i_o in range(self.d_obs_): # iterate over point (x,y)
            for i_c in range(self.d_txn_): # iterate over txn-3
                r_i.append(i_obs*self.d_obs_+i_o)
                c_i.append(self.i_src_*self.d_txn_+i_c)
                v  .append(jac_txn[:,i_o,i_c])
            for i_c in range(self.d_rxn_): # iterate over rxn-3
                r_i.append(i_obs*self.d_obs_+i_o)
                c_i.append((self.n_src_*self.d_txn_)+self.i_src_*self.d_rxn_+i_c)
                v  .append(jac_rxn[:,i_o,i_c])

            if self.pose_only_:
                continue

            for i_l in range(self.d_lmk_): # iterate over landmark-3
                r_i.append(i_obs*self.d_obs_+i_o)
                c_i.append((self.n_src_*self.d_txn_+self.n_src_*self.d_rxn_)+self.i_lmk_*self.d_lmk_+i_l)
                v  .append(jac_lmk[:,i_o,i_l])
        r_i = np.concatenate(r_i)
        c_i = np.concatenate(c_i)
        v   = np.concatenate(v)
        res = csr_matrix((v, (r_i, c_i)), shape=(n_out, n_in))

        return res

    def project(self, params, np=np, jac=False):
        # split data
        txn, rxn, lmk = self.unroll(params, self.n_src_, self.n_lmk_)
        # format data
        txn = txn[self.i_src_]
        rxn = rxn[self.i_src_]
        lmk = lmk[self.i_lmk_]

        # project
        if self.invd_:
            if self.axa_:
                res = project_axa_invd(txn,rxn,lmk,self.K_, np=np, jac=jac)
            else:
                res = project_invd(txn,rxn,lmk,self.K_, np=np, jac=jac)
        else:
            if self.axa_:
                res = project_axa(txn,rxn,lmk,self.K_, np=np, jac=jac)
            else:
                res = project(txn,rxn,lmk,self.K_, np=np, jac=jac)

        #data = {}
        #res = Projector(txn,rxn,lmk,self.K_).compute(np=np, jac=jac, data=data)
        #print data.keys()
        return res

    def residual(self, params, np=np):
        p_prj = self.project(params, np=np, jac=False)
        p_prj = np.stack(p_prj,axis=-1)
        d = (p_prj - self.p_obs_).ravel()
        #d[np.abs(d) > 10.0] *= 0 # ignore outliers ...
        return d

    #@profile
    def compute(self, crit={}, data={}):
        if (self.n_src_ <= 0) or (self.n_lmk_ <= 0):
            # unable to perform BA on empty data
            data['txn'] = self.txn_
            data['rxn'] = self.rxn_
            data['lmk'] = self.lmk_
            return False
        # criteria (local copy)
        tmp = self.crit_.copy()
        tmp.update(crit)
        crit = tmp

        # initial values
        txn, rxn = self.txn_, self.rxn_
        lmk      = self.lmk_

        # parametrize
        txn, rxn = self.invert(txn, rxn) # pose -> transform
        txn, rxn, lmk = self.parametrize(txn, rxn, lmk)
        x0 = self.roll(txn, rxn, lmk)

        res = least_squares(
                self.residual, x0,
                jac=self.jacobian,
                x_scale='jac',
                **crit
                )
        #print res.x.shape
        #print res.jac.shape # d(err / param)

        # un-parametrize
        txn, rxn, lmk = self.unroll(res.x, self.n_src_, self.n_lmk_)
        txn, rxn, lmk = self.unparametrize(txn, rxn, lmk)
        txn, rxn = self.invert(txn, rxn) # transform -> pose

        data['txn'] = txn
        data['rxn'] = rxn
        data['lmk'] = lmk

        return res.success

def main():
    np.random.seed(0)
    n_pt = 4096
    i_src = np.concatenate([np.full(n_pt, 0), np.full(n_pt, 1)])
    i_lmk = np.concatenate([np.arange(n_pt), np.arange(n_pt)])
    p_obs = np.random.uniform((0,0),(640,480),size=(2*n_pt,2))
    txn   = np.random.normal(scale=0.1, size=(2,3))
    rxn   = np.random.normal(scale=0.1, size=(2,3))
    lmk   = np.random.uniform(size=(n_pt,3))
    K     = np.float32([
        500, 0, 320,
        0, 500, 240,
        0, 0, 1]).reshape(3,3)

    BundleAdjustment(
            i_src, i_lmk, p_obs,
            txn, rxn, lmk,
            K
            ).compute()
if __name__ == '__main__':
    main()
