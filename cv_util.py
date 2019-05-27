#!/usr/bin/env python2
import cv2
import numpy as np
from cho_util import vmath as vm
from collections import namedtuple

def H(*args, **kwargs):
    h, msk = cv2.findHomography(
            *args, **kwargs)
    if msk is not None:
        msk = msk[:,0].astype(np.bool)
    return h, msk

def E(*args, **kwargs):
    """ wrap cv2.findEssentialMat() wrapper """
    # WARNING : cv2.findEssentialMat()
    # sometimes fails and raises and Error.
    e, msk = cv2.findEssentialMat(
            *args, **kwargs)
    if msk is not None:
        msk = msk[:,0].astype(np.bool)
    return e, msk

def F(*args, **kwargs):
    f, msk = cv2.findFundamentalMat(
            *args, **kwargs)
    if msk is not None:
        msk = msk[:,0].astype(np.bool)
    return f, msk

def project_points(*args, **kwargs):
    pt2, jac = cv2.projectPoints(*args, **kwargs)
    return pt2[:, 0]

def correct_matches(F, pta, ptb):
    pta_f, ptb_f = cv2.correctMatches(F, pta[None,...], ptb[None,...])
    pta_f = pta_f[0]
    ptb_f = ptb_f[0]
    return pta_f, ptb_f

def triangulate_points(Pa, Pb, pta, ptb,
        *args, **kwargs):
    pt_h = cv2.triangulatePoints(
            Pa, Pb,
            pta[None,...],
            ptb[None,...])
    return vm.from_h(pt_h.T)

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
                z_min = np.finfo(np.float64).eps,
                z_max = np.inf,
                e_max = 1.0,
                f_max = 0.99998, # max **cosine** parallax to be considered finite
                t_min = 64, # minimum number of triangulated points
                p_min = np.deg2rad(1.0) # minimum parallax to accept solution
                )

    def E(_):
        if (_['pt1'].shape[0] <= 5):
            print('Impossible to compute error')
            return Reconstructor.E_t(None, None)
        mat, msk = E(_['pt1'], _['pt0'], _['K'], method=cv2.FM_RANSAC)
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
        return triangulate_points(
                _['KP1'], _['KP0'],
                _['pt1'], _['pt0'],
                )
    def cld0(_):
        return vm.tx3(_['P0'], _['cld1'])
    def pt0r(_):
        #R, t = (_['P0'][:3,:3], _['P0'][:3,3:])
        #rvec, tvec = cv2.Rodrigues(R)[0], t
        return project_points(_['cld0'], np.zeros(3), np.zeros(3), _['K'], np.zeros(5))
    def pt1r(_):
        rvec = cv2.Rodrigues(_['P1'][:3, :3])[0]
        tvec = _['P1'][:3, 3:]
        return project_points(_['cld1'], rvec, tvec, _['K'], np.zeros(5))
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
        n_similar = np.sum(np.greater(n_good, n_best*0.7)) # << TODO : MAGIC 0.7 ratio
        min_good = np.maximum(0.7*n_pt, crit['t_min']) # << TODO : MAGIC 0.5 ratio

        # p_det guarantees
        # at least `t_min` parallax values
        # are greater than `p_min`

        # below is a somewhat less efficient version 
        # but better readibility
        n_pgood = np.sum(data['cospar'] <= np.cos(crit['p_min']))
        p_det  = (n_pgood >= crit['t_min'])

        # below is an efficient(?) version
        # but with worse readibility (from ORB-SLAM)
        # if n_best <= crit['t_min']:
        #     p_det = data['parallax'].min() >= crit['p_min']
        # else:
        #     p_det = np.sort(data['parallax'])[-crit['t_min']] >= crit['p_min']

        print '=== dbg ==='
        print 'n_pt', n_pt
        print 'n_best', n_best
        print 'n_similar', n_similar
        print 'n_pgood', n_pgood
        print 'min_good', min_good

        print '=== conclusion ==='
        print 'points', n_best >= min_good # sufficient points
        print 'unique', (n_similar == 1) # unique solution
        print 'parallax', p_det # sufficient parallax

        suc = np.logical_and.reduce([
            n_best >= min_good, # sufficient points
            n_similar == 1, # unique solution
            p_det # sufficient parallax
            ])
        return suc

#def resolve_perm(perm, K,
#        pt1, pt2,
#        threshold=0.8,
#        z_range=None,
#        guess=None
#        ):
#    P1 = np.eye(3,4)
#    P2 = np.eye(3,4)
#
#    sel   = 0
#    scores = [0.0 for _ in perm]
#    msks = [None for _ in perm]
#    pt3s = [None for _ in perm]
#    ctest = -np.inf
#
#    for i, (R, t) in enumerate(perm):
#        # Compute Projection Matrix
#        P2[:3,:3] = R
#        P2[:3,3:] = t.reshape(3,1)
#        KP1 = K.dot(P1) # NOTE : this could be unnecessary, idk.
#        KP2 = K.dot(P2)
#
#        # Triangulate Points
#        pt3 = triangulate_points(KP1, KP2, pt1, pt2)
#        pt3_a = pt3
#        pt3_b = M.tx3(P2, pt3)
#
#        # apply z-value (depth) filter
#        za, zb = pt3_a[:,2], pt3_b[:,2]
#        msk_i = np.logical_and.reduce([
#            z_min < za,
#            za < z_max,
#            z_min < zb,
#            zb < z_max
#            ])
#        c = msk_i.sum()
#
#        # store data
#        pt3s[i] = pt3_a # NOTE: a, not b
#        msks[i] = msk_i
#        scores[i] = ( float(msk_i.sum()) / msk_i.size)
#
#        if log:
#            print('[{}] {}/{}'.format(i, c, msk_i.size))
#            print_Rt(R, t)
#
#    # option one: compare best/next-best
#    sel = np.argmax(scores)
#
#    if guess is not None:
#        # -- option 1 : multiple "good" estimates by score metric
#        # here, threshold = score
#        # soft_sel = np.greater(scores, threshold)
#        # soft_idx = np.where(soft_sel)[0]
#        # do_guess = (soft_sel.sum() >= 2)
#        # -- option 1 end --
#
#        # -- option 2 : alternative next estimate is also "good" by ratio metric
#        # here, threshold = ratio
#        next_idx, best_idx = np.argsort(scores)[-2:]
#        soft_idx = [next_idx, best_idx]
#        if scores[best_idx] >= np.finfo(np.float32).eps:
#            do_guess = (scores[next_idx] / scores[best_idx]) > threshold
#        else:
#            # zero-division protection
#            do_guess = False
#        # -- option 2 end --
#
#        soft_scores = []
#        if do_guess:
#            # TODO : currently, R-guess is not supported.
#            R_g, t_g = guess
#            t_g_u = M.uvec(t_g.ravel()) # convert guess to uvec
#            
#            for i in soft_idx:
#                # filter by alignment with current guess-translational vector
#                R_i, t_i = perm[i]
#                t_i_u = M.uvec(t_i.ravel())
#                score_i = t_g_u.dot(t_i_u)
#                soft_scores.append(score_i)
#
#            # finalize selection
#            sel = soft_idx[ np.argmax(soft_scores) ]
#            unsel = soft_idx[ np.argmin(soft_scores) ] # NOTE: log-only
#
#    R, t = perm[sel]
#    msk = msks[sel]
#    pt3 = pt3s[sel][msk]
#    n_in = msk.sum()
#
#    if return_index:
#        return n_in, R, t, msk, pt3, sel
#    else:
#        return n_in, R, t, msk, pt3
#
#def recover_pose(E, K,
#        pt1, pt2,
#        threshold=0.8,
#        z_range=None,
#        guess=None,
#        ):
#
#    if z_range is None:
#        z_range = (np.finfo(np.float32).eps, np.inf)
#
#    R1, R2, t = cv2.decomposeEssentialMat(E)
#    perm = [(R1, t), (R2, t), (R1, -t), (R2, -t)]
#    return resolve_perm(perm, K,
#            pt1, pt2,
#            z_range,
#            threshold=threshold,
#            guess=guess,
#            )
