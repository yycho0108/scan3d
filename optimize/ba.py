#!/usr/bin/env python2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix, csr_matrix
from cho_util import vmath as vm
from autograd import jacobian
from autograd import numpy as anp
from expr import project

from profilehooks import profile

class BundleAdjustment(object):
    # static dimensions
    D_OBS = 2
    D_TXN = 3
    D_RXN = 3
    D_LMK = 3

    def __init__(self,
            i_src, i_lmk, p_obs,
            txn, rxn, lmk,
            K
            ):
        self.crit_ = dict(
                ftol=1e-4,
                xtol=1e-4,#np.finfo(float).eps,
                loss='huber',
                max_nfev=2,
                method='trf',
                verbose=2,
                tr_solver='lsmr',
                f_scale=1.0
                )
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

        # derived data
        self.n_src_ = 1+np.max(i_src)
        self.n_lmk_ = 1+np.max(i_lmk)
        self.n_obs_ = len(self.p_obs_)

        # simple checks
        assert self.n_src_ == len(self.txn_), '{}!={}'.format(self.n_src_, len(self.txn_))
        assert(self.n_src_ == len(self.rxn_))
        assert(self.n_lmk_ == len(self.lmk_))

        # automatic differentiation?
        #self.jacobian = jacobian(lambda x : self.residual(x, np=anp))

    def invert(self, txn, rxn):
        Ti = [vm.tx.inverse_matrix(vm.tx.compose_matrix(translate=t, angles=r)) for (t, r) in zip(txn, rxn)]
        txn, rxn = zip(*[( vm.tx.translation_from_matrix(T), vm.tx.euler_from_matrix(T) ) for T in Ti])
        txn = np.float32(txn)
        rxn = np.float32(rxn)
        return txn, rxn

    @staticmethod
    def roll(txn, rxn, lmk):
        return np.concatenate([txn.ravel(), rxn.ravel(), lmk.ravel()])

    @staticmethod
    def unroll(params, n_src, n_lmk):
        i0 = 0
        i1 = (i0 + BundleAdjustment.D_TXN * n_src)
        i2 = (i1 + BundleAdjustment.D_RXN * n_src)
        i3 = (i2 + BundleAdjustment.D_LMK * n_lmk)

        txn = params[i0:i1].reshape(-1,3)
        rxn = params[i1:i2].reshape(-1,3)
        lmk = params[i2:i3].reshape(-1,3)

        return txn, rxn, lmk

    #def project(self, txn, rxn, lmk):
    #    pt3 = txn + rotate(rxn, lmk)
    #    pt2 = vm.from_h(pt3.dot(self.K_.T)) # == [K.dot(p) for p in pt3]
    #    return pt2

    def jacobian(self, params):
        txn, rxn, lmk = BundleAdjustment.unroll(params, self.n_src_, self.n_lmk_)
        point, jac_txn, jac_rxn, jac_lmk = self.project(params, jac=True)

        n_out = self.n_obs_*BundleAdjustment.D_OBS
        n_in  = self.n_src_*(BundleAdjustment.D_RXN + BundleAdjustment.D_TXN)
        n_in += self.n_lmk_*(BundleAdjustment.D_LMK)
        
        j0_shape = (self.n_obs_, BundleAdjustment.D_OBS, n_in)
        j_shape = (n_out, n_in)

        jac_txn = np.transpose(jac_txn, (2,0,1))
        jac_rxn = np.transpose(jac_rxn, (2,0,1))
        jac_lmk = np.transpose(jac_lmk, (2,0,1))

        #J_res = lil_matrix((n_out, n_in))
        i_obs = np.arange(self.n_obs_)

        r_i = []
        c_i = []
        v   = [] 
        
        for i_o in range(self.D_OBS): # iterate over point (x,y)
            for i_c in range(self.D_TXN): # iterate over txn-3
                r_i = np.r_[r_i, i_obs*self.D_OBS+i_o]
                c_i = np.r_[c_i, self.i_src_*self.D_TXN+i_c]
                v   = np.r_[v  , jac_txn[:,i_o,i_c]]
                #J_res[i_obs*self.D_OBS+i_o, self.i_src_*self.D_TXN+i_c] = jac_txn[:,i_o,i_c]
            for i_c in range(self.D_RXN): # iterate over rxn-3
                r_i = np.r_[r_i, i_obs*self.D_OBS+i_o]
                c_i = np.r_[c_i, (self.n_src_*self.D_TXN)+self.i_src_*self.D_RXN+i_c]
                v   = np.r_[v  , jac_rxn[:,i_o,i_c]]
                #J_res[i_obs*self.D_OBS+i_o, (self.n_src_*self.D_TXN)+self.i_src_*self.D_RXN+i_c] = jac_rxn[:,i_o,i_c]
            for i_l in range(self.D_LMK): # iterate over landmark-3
                r_i = np.r_[r_i, i_obs*self.D_OBS+i_o]
                c_i = np.r_[c_i, (self.n_src_*self.D_TXN+self.n_src_*self.D_RXN)+self.i_lmk_*self.D_LMK+i_l]
                v   = np.r_[v  , jac_lmk[:,i_o,i_l]]
                #J_res[i_obs*self.D_OBS+i_o, (self.n_src_*self.D_TXN+self.n_src_*self.D_RXN)+self.i_lmk_*self.D_LMK+i_l] = jac_lmk[:,i_o,i_l]
        #res = J_res.tocsr()

        res = csr_matrix((v, (r_i, c_i)), shape=(n_out, n_in))

        # print('res', res.todense())
        # res1 = res.todense()
        # res = np.zeros(shape=j_shape, dtype=np.float32)
        # tmp = res.reshape(j0_shape)
        # txn_end = self.n_src_*BundleAdjustment.D_TXN
        # rxn_end = txn_end + self.n_src_ * BundleAdjustment.D_RXN
        # res_txn = tmp[..., :txn_end]
        # res_rxn = tmp[..., txn_end:rxn_end]
        # res_lmk = tmp[..., rxn_end:]
        # res_txn = res_txn.reshape(self.n_obs_, BundleAdjustment.D_OBS, self.n_src_, BundleAdjustment.D_TXN)
        # res_txn[i_obs,:,self.i_src_] += jac_txn
        # res_rxn = res_rxn.reshape(self.n_obs_, BundleAdjustment.D_OBS, self.n_src_, BundleAdjustment.D_RXN)
        # res_rxn[i_obs,:,self.i_src_] += jac_rxn
        # res_lmk = res_lmk.reshape(self.n_obs_, BundleAdjustment.D_OBS, self.n_lmk_, BundleAdjustment.D_LMK)
        # res_lmk[i_obs,:,self.i_lmk_] += jac_lmk
        # res2 = res
        # print res1[:3, :3]
        # print res2[:3, :3]
        # print 'jacobian error', np.square(res1 - res2).sum()

        return res

    def project(self, params, np=np, jac=False):
        txn, rxn, lmk = BundleAdjustment.unroll(params, self.n_src_, self.n_lmk_)
        txn = txn[self.i_src_]
        rxn = rxn[self.i_src_]
        lmk = lmk[self.i_lmk_]
        res = project(txn,rxn,lmk,self.K_, np=np, jac=jac)
        return res

    def residual(self, params, np=np):
        p_prj = self.project(params, np=np, jac=False)
        p_prj = np.stack(p_prj,axis=-1)
        return (p_prj - self.p_obs_).ravel()

    @profile
    def compute(self, crit={}, data={}):
        # criteria (local copy)
        tmp = self.crit_.copy()
        tmp.update(crit)
        crit = tmp

        # initial values
        txn, rxn = self.txn_, self.rxn_
        lmk      = self.lmk_

        # parametrize
        txn, rxn = self.invert(txn, rxn) # pose -> transform
        x0 = self.roll(txn, rxn, lmk)

        res = least_squares(
                self.residual, x0,
                jac=self.jacobian,
                #jac='2-point',
                #jac_sparsity=A,
                x_scale='jac',
                **crit
                )
        #print res.x.shape
        #print res.jac.shape # d(err / param)

        # un-parametrize
        txn, rxn, lmk = self.unroll(res.x, self.n_src_, self.n_lmk_)
        txn, rxn = self.invert(txn, rxn) # transform -> pose

        data['txn'] = txn
        data['rxn'] = rxn
        data['lmk'] = lmk

        return res.success

def main():
    np.random.seed(0)
    n_pt = 47
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
