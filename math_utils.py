import numpy as np
from lstm_model.learning_utils import MyConfig
from numpy import linalg as la

eps = np.finfo(float).eps


class ConstVelModel:
    def __init__(self, conf=MyConfig()):
        self.my_conf = conf

    def predict(self, inp): # use config for
        #inp.ndim = 2
        avg_vel = np.array([0, 0])
        if inp.ndim > 1 and inp.shape[0] > 1:
            for i in range(1, inp.shape[0]):
                avg_vel = avg_vel + inp[i, :]-inp[i-1, :]
            avg_vel = avg_vel / (inp.shape[0]-1)

        cur_pos = inp[-1, :]
        out = np.empty((0, 2))
        for i in range(0, self.my_conf.n_next):
            out = np.vstack((out, cur_pos + avg_vel * (i+1)))

        return out


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return np.array([rho, phi]).transpose()


def norm(x):
    return la.norm(x)


def unit(x):
    return x / (norm(x) + eps)
