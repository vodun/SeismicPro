import numpy as np
from numba import njit


@njit(nogil=True)
def clip_numba(data, data_min, data_max):
    data_shape = data.shape
    data = data.reshape(-1)
    for i in range(len(data)):
        data[i] = min(max(data[i], data_min), data_max)
    return data.reshape(data_shape)


@njit(nogil=True)
def scale_standard_numba(data, mean, std, eps):
    data = (data - mean) / (std + eps)
    return data


@njit(nogil=True)
def scale_maxabs_numba(data, min_value, max_value, clip, eps):
    max_abs = np.maximum(np.abs(min_value), np.abs(max_value))
    # Use np.atleast_2d(array).T to make the array 2-dimentional by adding dummy trailing axes
    # for further broadcasting to work tracewise
    data /= np.atleast_2d(max_abs).T + eps
    if clip:
        data = clip_numba(data, -1, 1)
    return data


@njit(nogil=True)
def scale_minmax_numba(data, min_value, max_value, clip, eps):
    # Use np.atleast_2d(array).T to make the array 2-dimentional by adding dummy trailing axes
    # for further broadcasting to work tracewise
    min_value = np.atleast_2d(np.array(min_value)).T
    max_value = np.atleast_2d(np.array(max_value)).T
    data = (data - min_value) / (max_value - min_value + eps)
    if clip:
        data = clip_numba(data, 0, 1)
    return data
