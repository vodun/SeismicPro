"""Implements optimized functions for various gather normalizations"""

import numpy as np
from numba import njit

from . import general_utils


@njit(nogil=True)
def scale_standard(data, mean, std, eps):
    r"""Scale `data` using the following formula:

    :math:`S = \frac{data - mean}{std + eps}`

    Parameters
    ----------
    data : np.ndarray
        Data to scale.
    mean : float or np.ndarray
        Mean value. Must be broadcastable to `data.shape`.
    std : float or np.ndarray
        Standard deviation. Must be broadcastable to `data.shape`.
    eps : float
        A constant to be added to the denominator to avoid division by zero.

    Returns
    -------
    data : np.ndarray
        Scaled data with unchanged shape.
    """
    data = (data - mean) / (std + eps)
    return data


@njit(nogil=True)
def scale_maxabs(data, min_value, max_value, clip, eps):
    r"""Scale `data` using the following formula:

    :math:`S = \frac{data}{max(|min_value|, |max_value|) + eps}`

    Parameters
    ----------
    data : 2d np.ndarray
        Data to scale.
    min_value : int, float, 1d or 2d array-like
        Minimum value. Dummy trailing axes are added to the array to have at least 2 dimensions, the result must be
        broadcastable to `data.shape`.
    max_value : int, float, 1d or 2d array-like
        Maximum value. Dummy trailing axes are added to the array to have at least 2 dimensions, the result must be
        broadcastable to `data.shape`.
    clip : bool
        Whether to clip scaled data to the [-1, 1] range.
    eps : float
        A constant to be added to the denominator to avoid division by zero.

    Returns
    -------
    data : np.ndarray
        Scaled data with unchanged shape.
    """
    max_abs = np.maximum(np.abs(min_value), np.abs(max_value))
    # Use np.atleast_2d(array).T to make the array 2-dimentional by adding dummy trailing axes
    # for further broadcasting to work tracewise
    data /= np.atleast_2d(np.asarray(max_abs)).T + eps
    if clip:
        data = general_utils.clip(data, np.float32(-1), np.float32(1))
    return data


@njit(nogil=True)
def scale_minmax(data, min_value, max_value, clip, eps):
    r"""Scale `data` using the following formula:

    :math:`S = \frac{data - min_value}{max_value - min_value + eps}`

    Parameters
    ----------
    data : 2d np.ndarray
        Data to scale.
    min_value : int, float, 1d or 2d array-like
        Minimum value. Dummy trailing axes are added to the array to have at least 2 dimensions, the result must be
        broadcastable to `data.shape`.
    max_value : int, float, 1d or 2d array-like
        Maximum value. Dummy trailing axes are added to the array to have at least 2 dimensions, the result must be
        broadcastable to `data.shape`.
    clip : bool
        Whether to clip scaled data to the [0, 1] range.
    eps : float
        A constant to be added to the denominator to avoid division by zero.

    Returns
    -------
    data : np.ndarray
        Scaled data with unchanged shape.
    """
    # Use np.atleast_2d(array).T to make the array 2-dimentional by adding dummy trailing axes
    # for further broadcasting to work tracewise
    min_value = np.atleast_2d(np.asarray(min_value)).T
    max_value = np.atleast_2d(np.asarray(max_value)).T
    data = (data - min_value) / (max_value - min_value + eps)
    if clip:
        data = general_utils.clip(data, np.float32(0), np.float32(1))
    return data
