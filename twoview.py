#!/usr/bin/env python2
import cv2
import numpy as np
from cho_util import vmath as vm
from collections import namedtuple
import cv_util as cvu

def score_H(pt1, pt2, H, sigma=1.0):
    """ Homography model symmetric transfer error. """
    score = 0.0
    th = 5.991 # ??? TODO : magic number
    iss = (1.0 / (sigma*sigma))

    Hi = vm.inv(H)
    pt2_r = vm.from_h(vm.to_h(pt1).dot(H.T))
    pt1_r = vm.from_h(vm.to_h(pt2).dot(Hi.T))
    e1 = np.square(pt1 - pt1_r).sum(axis=-1)
    e2 = np.square(pt2 - pt2_r).sum(axis=-1)

    #score = 1.0 / (e1.mean() + e2.mean())
    chi_sq1 = e1 * iss
    msk1 = (chi_sq1 <= th)
    score += ((th - chi_sq1) * msk1).sum()

    chi_sq2 = e2 * iss
    msk2 = (chi_sq2 <= th)
    score += ((th - chi_sq2) * msk2).sum()
    return score, (msk1 & msk2)

def score_F(pt1, pt2, F, sigma=1.0):
    """
    Fundamental Matrix symmetric transfer error.
    reference:
        https://github.com/opencv/opencv/blob/master/modules/calib3d/src/fundam.cpp#L728
    """
    score = 0.0
    th = 3.841 # ??
    th_score = 5.991 # ?? TODO : magic number
    iss = (1.0 / (sigma*sigma))

    pt1_h = vm.to_h(pt1)
    pt2_h = vm.to_h(pt2)

    x1, y1 = pt1.T
    x2, y2 = pt2.T

    a, b, c = pt1_h.dot(F.T).T # Nx3
    s2 = 1./(a*a + b*b);
    d2 = a * x2 + b * y2 + c
    e2 = d2*d2*s2

    a, b, c = pt2_h.dot(F).T
    s1 = 1./(a*a + b*b);
    d1 = a * x1 + b * y1 + c
    e1 = d1*d1*s1

    #score = 1.0 / (e1.mean() + e2.mean())
    chi_sq2 = e2 * iss
    msk2 = (chi_sq2 <= th)
    score += ((th_score - chi_sq2) * msk2).sum()

    chi_sq1 = e1* iss
    msk1 = (chi_sq1 <= th)
    score += ((th_score - chi_sq1) * msk1).sum()

    return score, (msk1 & msk2)

class TwoView(object):
    """
    Two-Frame Reconstructor.
    Somewhat obsessive attempt to never repeat computation.

    IMPORTANT :: all transforms will reference
    the camera Matrix of **pt1** as the map coordinate frame.
    """

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
                p_min = np.deg2rad(2.0), # minimum parallax to accept solution
                u_max = 0.7 # min ratio by which `unique` xfm must surpass alternatives
                )
    def _H(_):
        """ internal function for invoking H() to allow multiple returns """
        if (_['pt1'].shape[0] <= 5):
            print('Impossible to compute Homography Matrix')
            return None, None
        return cvu.H(_['pt1'], _['pt0'],
                method=cv2.FM_RANSAC,
                confidence=0.99,
                ransacReprojThreshold=0.5
                #prob=0.99,
                #threshold=0.5
                )
    def H(_):
        """ Homography Matrix """
        return _['_H'][0]
    def msk_H(_):
        """ Homography Matrix Inlier Mask """
        return _['_H'][1]

    def _E(_):
        """ internal function for invoking E() to allow multiple returns """
        if (_['pt1'].shape[0] <= 5):
            print('Impossible to compute Essential Matrix')
            return None, None
        return cvu.E(_['pt1'], _['pt0'], _['K'],
                method=cv2.FM_RANSAC,
                prob=0.99,
                threshold=0.5
                )
    def E(_):
        """ Essential Matrix """
        return _['_E'][0]
    def msk_E(_):
        """ Essential Matrix Inlier Mask"""
        return _['_E'][1]

    def F(_):
        """ Fundamental Matrix (converted from E()) """
        return vm.E2F(_['E'],K=_['K'])
    def P0(_):
        """ Projection Matrix (view 0) """
        return np.concatenate([_['R'], _['t'].reshape(3,1)], axis=1)
    def P1(_):
        """ Projection Matrix (view 1) """
        return np.eye(3,4)
    def KP0(_):
        """ Camera Projection Matrix (view 0) """
        return _['K'].dot(_['P0'])
    def KP1(_):
        """ Camera Projection Matrix (view 1) """
        return _['K'].dot(_['P1'])
    def cld1(_):
        """ Triangulated Point Cloud (view 1) """
        return cvu.triangulate_points(
                _['KP1'], _['KP0'],
                _['pt1'], _['pt0'],
                )
    def cld0(_):
        """ Triangulated PointCloud (view 2) """
        return vm.tx3(_['P0'], _['cld1'])
    def pt0r(_):
        """ Reconstructed pixel-coordinate point (view 0) """
        #R, t = (_['P0'][:3,:3], _['P0'][:3,3:])
        #rvec, tvec = cv2.Rodrigues(R)[0], t
        return cvu.project_points(_['cld0'], np.zeros(3), np.zeros(3), _['K'], np.zeros(5))
    def pt1r(_):
        """ Reconstructed pixel-coordinate point (view 1) """
        rvec = cv2.Rodrigues(_['P1'][:3, :3])[0]
        tvec = _['P1'][:3, 3:]
        return cvu.project_points(_['cld1'], rvec, tvec, _['K'], np.zeros(5))
    def n0(_):
        """ cld0 `normal direction` vector (unnormalized) """
        # only apply translational component to cld1.
        #return _['cld0']
        return _['cld1'] + _['t'].reshape(1,3)
        #return _['cld1'] - vm.rx3(_['R'].T, -_['t'].ravel())
    def n1(_):
        """ cld1 `normal direction` vector (unnormalized) """
        return _['cld1']
    def d0(_):
        """ euclidean (l2)-length of n0 """
        return np.linalg.norm(_['n0'], axis=-1)
    def d1(_):
        """ euclidean (l2)-length of n1 """
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
        """ evaluate the score of a particular permutation """
        # unroll evaluation criteria
        z_min = crit['z_min']
        z_max = crit['z_max']
        e_max = crit['e_max']
        f_max = crit['f_max']
        p_min = crit['p_min']

        # unroll data
        cld0, cld1 = _['cld0'], _['cld1']

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
                # model inlier check
                _['msk_H'] if (_['model'] == 'H') else _['msk_E'],

                # depth validity check
                op_or(msk_inf, z_min <= z0),
                z0 < z_max,
                op_or(msk_inf, z_min <= z1),
                z1 < z_max,

                # reprojection error check
                r_err0 < e_max,
                r_err1 < e_max
                ]

        msk = np.logical_and.reduce(msks)
        # "valid" triangulation mask
        msk_cld = np.logical_and(msk, ~msk_inf)
        # count of "good" points including infinity
        n_good   = msk.sum()
        return n_good, msk_cld

    def r_H(_):
        """ r_H (ratio to determine Homography/Essential Matrix model """
        sF = score_F(_['pt1'], _['pt0'], _['F'])[0]
        sH = score_H(_['pt1'], _['pt0'], _['H'])[0]
        return (sH / (sH + sF)) # ratio

    def model(_):
        """ Model Identifier """
        return ('H' if (_['r_H'] > 0.45) else 'E')

    def select_model(_):
        """ Select model and return possible transform permutations """
        if (_['model'] == 'H'):
            # decompose_H()
            res_h, Hr, Ht, Hn = cv2.decomposeHomographyMat(_['H'],_['K'])
            Ht = np.float32(Ht)
            Ht /= np.linalg.norm(Ht, axis=(1,2), keepdims=True)
            T_perm = zip(Hr, Ht)
        else: 
            # decompose_E()
            R1, R2, t = cv2.decomposeEssentialMat(_['E'])
            T_perm = [(R1, t), (R2, t), (R1, -t), (R2, -t)]
        return T_perm

    def compute(self, crit={}, data={}):
        """
        Compute reconstruction.

        Arguments:
            crit(dict): specify criteria to override.
            data(dict): this will be updated with intermediate values.

        Returns:
            suc(bool): whether reconstruction was successful.
            det(dict): dictionary of determinants for `suc` based on criteria.
        """

        # criteria
        tmp = self.crit_.copy()
        tmp.update(crit)
        crit = tmp

        T_perm = self.select_model()
        if (T_perm is None):
            data['dbg-tv'] = 'Unable to obtain valid transform permutations'
            return False

        # correct matches?
        if self['model'] != 'H':
            F = self['F']
            self.data_['pt1'], self.data_['pt0'] = cvu.correct_matches(F,
                    self.data_['pt1'],
                    self.data_['pt0'])

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

        det = dict(
                num_pts = (n_best >= min_good),
                uniq_xfm = (n_similar == 1),
                parallax = p_det
                )
        # override parallax flag
        # det['parallax'] = True

        # debug log
        data['dbg-tv'] = """
        [TwoView Log]
        model     : {} ({})
        num_pts   : {} ({} / {}),
        uniq_xfm  : {} ({} / {}),
        parallax  : {} ({} / {})
        """.format(
                data['model'], data['r_H'],
                det['num_pts'], n_best,  min_good,
                det['uniq_xfm'], n_similar, 1,
                det['parallax'], n_pgood, crit['t_min']
                )
        suc = np.logical_and.reduce(det.values())
        return suc, det
