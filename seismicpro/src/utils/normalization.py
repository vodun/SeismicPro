import numpy as np
from numba import njit

from . import general_utils


@njit(nogil=True)
def scale_standard(data, mean, std, eps):
    data = (data - mean) / (std + eps)
    return data


@njit(nogil=True)
def scale_maxabs(data, min_value, max_value, clip, eps):
    max_abs = np.maximum(np.abs(min_value), np.abs(max_value))
    # Use np.atleast_2d(array).T to make the array 2-dimentional by adding dummy trailing axes
    # for further broadcasting to work tracewise
    data /= np.atleast_2d(np.array(max_abs)).T + eps
    if clip:
        data = general_utils.clip(data, -1, 1)
    return data


@njit(nogil=True)
def scale_minmax(data, min_value, max_value, clip, eps):
    # Use np.atleast_2d(array).T to make the array 2-dimentional by adding dummy trailing axes
    # for further broadcasting to work tracewise
    min_value = np.atleast_2d(np.array(min_value)).T
    max_value = np.atleast_2d(np.array(max_value)).T
    data = (data - min_value) / (max_value - min_value + eps)
    if clip:
        data = general_utils.clip(data, 0, 1)
    return data
