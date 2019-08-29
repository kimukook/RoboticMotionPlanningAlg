'''
This is a simple script that implements the Kalman filter.
    - The KF algorithm is written with author own effort.
    - The dynamics are referred by the following link:
        https://github.com/balzer82/Kalman/blob/master/Kalman-Filter-CA-Ball.ipynb?create=1
    - If you are looking for a good tutorial of Kalman filter, you can take a look at this book
        <Optimal State Estimation, Kalman Hinfinity and Nonlinear Approaches> by Dan Simon

=========================
Author  :  Muhan Zhao
Date    :  Aug. 28, 2019
Location:  West Hill, LA, CA
=========================
'''
import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter:
    def __init__(self, F=None, G=None, H=None, Q=None, R=None, P=None):
        self.n = F.shape[0]  # dimension of states xk
        self.m = H.shape[0]  # dimension of measurements yk
        self.nc = 0 if G is None else G.shape[0]  # dimension of input

        self.F = F  # state dynamic matrix F
        self.H = H  # state dynamic matrix H
        self.G = np.zeros((self.nc, 1)) if G is None else G  # measurement dynamic
        self.Q = np.identity(self.n) if Q is None else Q  # wk noise covariance
        self.R = np.identity(self.m) if R is None else R  # vk noise covariance

        self.K = np.zeros((self.n, self.m))  # Kalman Gain matrix
        self.P = np.zeros((self.n, self.n)) if P is None else P  # state covariance matrix

    def predict(self, x, u):
        '''
        Time update
        :param x:   (Measurement update from last moment) Filtered values of x
        :param u:   Control input u
        :return:
        '''
        # time update
        x = x.reshape(-1, 1)
        x = np.dot(self.F, x) + np.dot(self.G, u)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return x

    def update(self, x, y):
        '''
        Measurement update
        :param x:  (Time update) Predicted values of x
        :param y:  Measurements y_k, self.m by 1
        :return:
        '''
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        # compute the Kalman gain
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.pinv(S)))

        x = x + np.dot(K, (y - np.dot(self.H, x)))
        I = np.identity(self.n)
        # self.P = np.dot(I - np.dot(K, self.H), np.dot(self.P, (I - np.dot(K, self.H)).T)) + np.dot(K, np.dot(self.R, K.T))
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        return x


if __name__ == "__main__":
    # problem setup
    # initialize the frequency and the time interval
    hz = 100  # frequency of vision system
    dt = 1.0 / hz
    T = 3  # time length

    # setup the initial position
    x = 0
    y = 0
    z = 1

    # setup the initial speed
    vx = 10
    vy = 0
    vz = 0

    # setup the drag resistance coefficient
    c = .1  # drag
    d = .9  # damping
    g = 9.8

    # setup the truth trajectory
    length = int(T / dt)
    xr, yr, zr = np.zeros(length), np.zeros(length), np.zeros(length)

    # ==========  Compute the truth trajectory  ==========
    for i in range(length):
        accx = -c * vx ** 2
        vx += accx * dt
        x += vx * dt

        accz = -g + c * vz ** 2
        vz += accz * dt
        z += vz * dt

        # ball bumping on the groud
        if z < 0.025:
            vz = -vz * d
            z += 0.02

        if vx < 0.1:
            accx = 0
            accz = 0

        xr[i] = x
        yr[i] = y
        zr[i] = z

    # ==========  Determine measurements by Adding Gaussian noise  ==========
    sigma = 0.1
    xm = xr + sigma * np.random.randn(length)
    ym = yr + sigma * np.random.randn(length)
    zm = zr + sigma * np.random.randn(length)

    measurements = np.vstack((xm, ym, zm))

    plt.figure(figsize=(16, 9))
    plt.scatter(xm, zm, label='measurements')
    plt.plot(xr, zr, label='truth', c='r')
    plt.xlabel('x')
    plt.ylabel('z')

    # ==========  Initialize A, B, H, Q, R matrices for the dynamic system ==========
    # x = Ax + Bu + w
    # y = Hx + v
    P = 10 * np.identity(9)
    A = np.matrix([[1.0, 0.0, 0.0, dt, 0.0, 0.0, 1/2.0*dt**2, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0, 1/2.0*dt**2, 0.0],
                  [0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0, 1/2.0*dt**2],
                  [0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  dt],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    H = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    r_sigma = 1.0 ** 2  # Noise of Position Measurement
    R = np.matrix([[r_sigma, 0.0, 0.0],
                   [0.0, r_sigma, 0.0],
                   [0.0, 0.0, r_sigma]])

    sa = 0.1
    G = np.matrix([[1 / 2.0 * dt ** 2],
                   [1 / 2.0 * dt ** 2],
                   [1 / 2.0 * dt ** 2],
                   [dt],
                   [dt],
                   [dt],
                   [1.0],
                   [1.0],
                   [1.0]])
    Q = G * G.T * sa ** 2

    u = 0
    B = np.matrix([[0.0],
                   [0.0],
                   [0.0],
                   [0.0],
                   [0.0],
                   [0.0],
                   [0.0],
                   [0.0],
                   [0.0]])

    # ==========  Implement Kalman Filter algorithm  ==========
    x0 = np.matrix([0.0, 0.0, 1.0, 10.0, 0.0, 0.0, 0.0, 0.0, -9.81]).T
    kf = KalmanFilter(F=A, G=B, H=H, Q=Q, R=R, P=P)

    # xf - filtered states
    xf = np.zeros((9, length))

    hitplate = False
    for i in range(length):
        # about to hit the ground
        if x0[2] < 0.025 and not hitplate:
            x0[5] = -x0[5]
            hitplate = True

        # after hitting the groud
        if x0[2] > 0.025 and x0[5] < 0 and hitplate:
            hitplate = False

        x0 = kf.predict(x0, u)
        x0 = kf.update(x0, measurements[:, i])
        xf[:, i] = np.copy(x0.T[0])

    plt.axhline(0, color='k')
    plt.plot(xf[0, :], xf[2, :], label='Kalman filter predictions', c='g')
    plt.legend()
    print('z min = ', np.min(xf[2, :]))
    plt.show()


