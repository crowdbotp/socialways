import numpy as np
from numpy import linalg as la

eps = np.finfo(float).eps


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return np.array([rho, phi]).transpose()


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return np.array([x, y]).transpose()


def norm(x, axis=-1):
    if axis < 0:
        return la.norm(x)
    else:
        return la.norm(x, axis=axis)


def unit(x):
    return x / (norm(x) + eps)
