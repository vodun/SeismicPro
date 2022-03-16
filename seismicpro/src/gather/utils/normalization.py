"""Implements optimized functions for various gather normalizations"""

import numpy as np
from numba import njit


@njit(nogil=True)
def clip_inplace(data, data_min, data_max):
    """Limit the `data` values. May change `data` inplace.

    `data` values outside [`data_min`, `data_max`] interval are clipped to the interval edges.

    Parameters
    ----------
    data : np.ndarray
        Data to clip.
    data_min : int, float
        Minimum value of the interval.
    data_max : int, float
        Maximum value of the interval.

    Returns
    -------
    data : np.ndarray
        Clipped data with the same shape.
    """
    data_shape = data.shape
    data = data.reshape(-1)  # may return a copy but usually a view
    for i in range(len(data)):  # pylint: disable=consider-using-enumerate
        data[i] = min(max(data[i], data_min), data_max)
    return data.reshape(data_shape)


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
    return (data - mean) / (std + eps)


@njit(nogil=True)
def scale_maxabs(data, min_value, max_value, clip, eps):
    r"""Scale `data` inplace using the following formula:

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
    max_abs += eps
    # Use np.atleast_2d(array).T to make the array 2-dimensional by adding dummy trailing axes
    # for further broadcasting to work tracewise
    data /= np.atleast_2d(max_abs).T
    if clip:
        data = clip_inplace(data, np.float32(-1), np.float32(1))
    return data


@njit(nogil=True)
def scale_minmax(data, min_value, max_value, clip, eps):
    r"""Scale `data` inplace using the following formula:

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
    # Use np.atleast_2d(array).T to make the array 2-dimensional by adding dummy trailing axes
    # for further broadcasting to work tracewise
    min_value = np.atleast_2d(np.asarray(min_value)).T
    max_value = np.atleast_2d(np.asarray(max_value) + eps).T
    data -= min_value
    data /= max_value - min_value
    if clip:
        data = clip_inplace(data, np.float32(0), np.float32(1))
    return data
