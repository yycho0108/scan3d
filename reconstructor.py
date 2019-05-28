#!/usr/bin/env python2
import cv2
import numpy as np
from cho_util import vmath as vm
from collections import namedtuple
import cv_util as cvu

class Reconstructor(object):
    """
    Two-Frame Reconstructor.
    Somewhat obsessive attempt to never repeat computation.

    IMPORTANT :: all transforms will reference
    the camera Matrix of **pt1** as the map coordinate frame.
    """

    # hack to circumvent multiple output from E()
    E_t = namedtuple('E_t', ['mat', 'msk'])

    def __init__(self, pt0, pt1, K):
        # initialize cache
        self.data_ = dict(
                pt0=pt0,
                pt1=pt1,
                K=K
                )
        self.crit_ = dict(
                z_min = np.finfo(np.float64).eps, # minimum z-depth
                z_max = np.inf, # max z-depth
                e_max = 1.0, # max reprojection error
                f_max = 0.99998, # max **cosine** parallax to be considered finite
                t_min = 64, # minimum number of triangulated points
                p_min = np.deg2rad(1.0), # minimum parallax to accept solution
                u_max = 0.7 # min ratio by which `unique` xfm must surpass alternatives
                )

    def E(_):
        if (_['pt1'].shape[0] <= 5):
            print('Impossible to compute error')
            return Reconstructor.E_t(None, None)
        mat, msk = cvu.E(_['pt1'], _['pt0'], _['K'], method=cv2.FM_RANSAC)
        return Reconstructor.E_t(mat=mat, msk=msk)
    def P0(_):
        return np.concatenate([_['R'], _['t'].reshape(3,1)], axis=1)
    def P1(_):
        return np.eye(3,4)
    def KP0(_):
        return _['K'].dot(_['P0'])
    def KP1(_):
        return _['K'].dot(_['P1'])
    def cld1(_):
        return cvu.triangulate_points(
                _['KP1'], _['KP0'],
                _['pt1'], _['pt0'],
                )
    def cld0(_):
        return vm.tx3(_['P0'], _['cld1'])
    def pt0r(_):
        #R, t = (_['P0'][:3,:3], _['P0'][:3,3:])
        #rvec, tvec = cv2.Rodrigues(R)[0], t
        return cvu.project_points(_['cld0'], np.zeros(3), np.zeros(3), _['K'], np.zeros(5))
    def pt1r(_):
        rvec = cv2.Rodrigues(_['P1'][:3, :3])[0]
        tvec = _['P1'][:3, 3:]
        return cvu.project_points(_['cld1'], rvec, tvec, _['K'], np.zeros(5))
    def n0(_):
        return _['cld0']# - vm.rx3(_['R'].T, -_['t'].ravel())
    def n1(_):
        return _['cld1']
    def d0(_):
        return np.linalg.norm(_['n0'], axis=-1)
    def d1(_):
        return np.linalg.norm(_['n1'], axis=-1)

    def cospar(_):
        """ cosine of parallax """
        return np.einsum('...a,...a->...', _['n0'], _['n1']) / (_['d0'] * _['d1'])

    def parallax(_):
        """ angular parallax, in radians. """
        return np.arccos(_['cospar'])

    def __getitem__(self, name):
        """ avoid repeating boilerplate @property code """
        if name not in self.data_:
            # compute cache
            fun = getattr(self, name, None)
            if fun is None:
                raise ValueError('attempting to access an invalid compute path')
            self.data_[name] = fun()
        # return cache
        return self.data_[name]

    def evaluate(_, crit):
        # unroll evaluation criteria
        z_min = crit['z_min']
        z_max = crit['z_max']
        e_max = crit['e_max']
        f_max = crit['f_max']
        p_min = crit['p_min']

        # unroll data
        cld0, cld1 = _['cld0'], _['cld1']
        msk_E = _['E'].msk

        # reprojection error
        r_err0 = np.linalg.norm(_['pt0r'] - _['pt0'], axis=-1)
        r_err1 = np.linalg.norm(_['pt1r'] - _['pt1'], axis=-1)

        # compute final mask
        op_or = np.logical_or # quick alias
        msk_inf = ( _['cospar'] >= f_max ) # == points at `infinity`
        z0, z1 = cld0[:,2], cld1[:,2]

        # compute parallax cache
        # _['parallax']

        # collect conditions
        msks = [
                # essential matrix consistency check
                _['E'].msk,

                # depth validity check
                op_or(msk_inf, z_min <= z0),
                z0 < z_max,
                op_or(msk_inf, z_min <= z1),
                z1 < z_max,

                # reprojection error check
                r_err0 < e_max,
                r_err1 < e_max
                ]
        #print('msks', [m.sum() for m in msks])

        msk = np.logical_and.reduce(msks)
        # "valid" triangulation mask
        msk_cld = np.logical_and(msk, ~msk_inf)
        # count of "good" points including infinity
        n_good   = msk.sum()
        return n_good, msk_cld

    def compute(self, crit={}, data={}):
        """ compute reconstruction. """

        # criteria
        tmp = self.crit_.copy()
        tmp.update(crit)
        crit = tmp

        if (self['E'].mat is None) or (len(self['E'].mat) > 3):
            print('non-unique or invalid essential matrix')
            return False

        # correct matches?
        #F = vm.E2F(self['E'].mat, self['K'])
        #self.data_['pt1'], self.data_['pt0'] = cvu.correct_matches(F,
        #        self.data_['pt1'],
        #        self.data_['pt0'])

        R1, R2, t = cv2.decomposeEssentialMat(self['E'].mat)
        T_perm = [(R1, t), (R2, t), (R1, -t), (R2, -t)]
        d_null = dict(self.data_)
        d_perm = [] # data for each permutation

        for (R, t) in T_perm:
            # reset data
            self.data_ = dict(d_null)
            self.data_['R'] = R
            self.data_['t'] = t

            # evaluate permutation
            n_good, msk_cld = self.evaluate(crit)

            # save evaluation results
            self.data_['n_good'] = n_good
            self.data_['msk_cld'] = msk_cld

            # save current cache
            d_perm.append( self.data_ )

        # determine best solution
        n_good   = [d['n_good'] for d in d_perm]
        best_idx = np.argmax(n_good)
        n_best   = n_good[best_idx]

        # update intermediate values
        data.update(d_perm[best_idx])

        # determine whether to accept the result
        n_pt = len(data['pt1'])
        n_similar = np.sum(np.greater(n_good, n_best*crit['u_max']))
        min_good = np.maximum(0.5*n_pt, crit['t_min']) # << TODO : MAGIC ratio

        # p_det guarantees
        # at least `t_min` parallax values
        # are greater than `p_min`

        # below is a somewhat less efficient version 
        # but better readibility
        n_pgood = np.sum(data['cospar'] <= np.cos(crit['p_min']))

        p_det  = (n_pgood >= crit['t_min'])
        # below is an efficient(?) version
        # but with worse readibility (from ORB-SLAM)
        #if n_best <= crit['t_min']:
        #    p_det = data['parallax'].min() >= crit['p_min']
        #else:
        #    p_det = np.sort(data['parallax'])[-crit['t_min']] >= crit['p_min']

        #print '=== dbg ==='
        #print 'n_pt', n_pt
        #print 'n_best', n_best
        #print 'n_similar', n_similar
        #print 'n_pgood', n_pgood, crit['t_min']
        #print 'min_good', min_good

        #print '=== conclusion ==='
        #print 'points', n_best >= min_good # sufficient points
        #print 'unique', (n_similar == 1) # unique solution
        #print 'parallax', p_det # sufficient parallax

        det = dict(
                num_pts = (n_best >= min_good),
                uniq_xfm = (n_similar == 1),
                parallax = p_det
                )
        suc = np.logical_and.reduce(det.values())
        return suc, det
