#!/usr/bin/env python

import numpy as np

from filterpy.kalman import ExtendedKalmanFilter as EKF
from filterpy.kalman import MerweScaledSigmaPoints, JulierSigmaPoints

from expr import fx, anorm

def hx(s):
    return np.r_[s[0:3], s[9:12]]

def HJ(_):
    J = np.zeros(shape=(6,15), dtype=np.float32)
    J[(0,1,2),(0,1,2)] = 1.
    J[(3,4,5),(9,10,11)] = 1.
    return J

def residual(a, b):
    """
    Runs circular residual for angular states, which is critical to preventing issues related to linear assumptions.
    WARNING : do not replace with the default residual function.
    """
    d = np.subtract(a, b)
    d[3:6] = anorm(d[3:6])
    return d

class EKFWrapper(EKF):
    def __init__(self, *args, **kwargs):
        super(EKFWrapper, self).__init__(*args, **kwargs)

    def predict(self, dt):
        self.x, F = fx(self.x, dt) # uses the same transition function
        self.F = F
        self.P = np.linalg.multi_dot([
            F, self.P, F.T]) + self.Q

        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)
        return

    def update(self, z):
        return super(EKFWrapper,self).update(z,
                HJ,
                hx,
                residual=residual)

def gen_drive(dt=0.1):
    data = []
    s = np.random.normal(size=15)
    for i in range(100):
        data.append( s.copy() )
        s, _ = fx(s, dt=dt)
    return data

def build_ekf(x0=None, P0=None,
        Q=None, R=None
        ):

    # build ukf
    if x0 is None:
        x0 = np.zeros(15)

    if P0 is None:
        # initial pose is very accurate, but velocity is unknown.
        # considering vehicle dynamics, it's most likely constrained
        # within +-0.5m/s in x direction, +-0.1m/s in y direction,
        # and 0.5rad/s in angular velocity.
        P0 = 1e-3 * np.eye(15)

    if Q is None:
        # treat Q as a "tuning parameter"
        # low Q = high confidence in general state estimation
        # high Q = high confidence in measurement
        # from https://www.researchgate.net/post/How_can_I_find_process_noise_and_measurement_noise_in_a_Kalman_filter_if_I_have_a_set_of_RSSI_readings
        # Higher Q, the higher gain, more weight to the noisy measurements
        # and the estimation accuracy is compromised; In case of lower Q, the
        # better estimation accuracy is achieved and time lag may be introduced in the estimated value.
        # process noise
        Q = 1e-3 * np.eye(15)

    if R is None:
        # in general anticipate much lower heading error than positional error
        # 1e-2 ~ 0.57 deg
        R = 1e-2 * np.eye(6)

    ekf = EKFWrapper(15, 6)
    ekf.x = x0.copy()
    ekf.P = P0.copy()
    ekf.Q = Q
    ekf.R = R
    return ekf

def main():
    data = gen_drive()
    ekf = EKFWrapper(15, 6)
    ekf.x = np.random.normal(loc=data[0], scale=0.5)
    ekf.P = 1e-2 * np.eye(15)
    ekf.Q = 1e-3 * np.eye(15)
    ekf.R = 1e-3 * np.eye(6)

    for i in range(1, len(data)):
        print 'compare', ekf.x[3:6], data[i-1][3:6]
        ekf.predict( 0.1 )
        ekf.update( hx(data[i]) )


if __name__ == '__main__':
    main()
