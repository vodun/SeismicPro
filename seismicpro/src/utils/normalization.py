"""Implements functions for various gather normalization"""
import numpy as np
from numba import njit

from . import general_utils


@njit(nogil=True)
def scale_standard(data, mean, std, eps):
    r"""Standardize `data` by given `mean` and `std`.

    The standard score of a data `x` is calculated as:

    :math:`z = \frac{x - u}{s + eps}`,

    where `u` is the mean and `s` is the standard deviation of the gather. The `eps` is the constant that is added to
    the denominator to avoid division by zero.

    The standardization does not change the type and size of the `data`, thus input statistics `mean` and `std`
    must be broadcastable to `data.shape`.

    Parameters
    ----------
    data : np.ndarray
        Data to standardize.
    mean : float or np.ndarray
        Mean value.
    std : float or np.ndarray
        Standard deviation.
    eps : float
        A constant to be added to the denominator to avoid division by zero.

    Returns
    -------
    data : np.ndarray
        Standardized data with unchanged shape.
    """
    data = (data - mean) / (std + eps)
    return data


@njit(nogil=True)
def scale_maxabs(data, min_value, max_value, clip, eps):
    r"""Scale `data` by the following formula:

    :math: `S = \frac{data}{max(|min_value|, |max_value|) + eps}`

    Parameters
    ----------
    data : 2d np.ndarray
        Data to scale.
    min_value : int, float, 1d or 2d array-like
        Minimum value.
    max_value : int, float, 1d or 2d array-like
        Maximum value.
    clip : bool
        Wether to clip scaled data by [-1, 1] range.
    eps : float
        A constant to be added to the denominator to avoid division by zero.

    Returns
    -------
    data : np.ndarray
        Scaled data with unchaned shape.
    """
    max_abs = np.maximum(np.abs(min_value), np.abs(max_value))
    # Use np.atleast_2d(array).T to make the array 2-dimentional by adding dummy trailing axes
    # for further broadcasting to work tracewise
    data /= np.atleast_2d(np.array(max_abs)).T + eps
    if clip:
        data = general_utils.clip(data, -1, 1)
    return data


@njit(nogil=True)
def scale_minmax(data, min_value, max_value, clip, eps):
    r"""Scale `data` by the following formula:

    :math: `S = \frac{data - min_value}{max_value - min_value + eps}`

    Parameters
    ----------
    data : 2d np.ndarray
        Data to scale.
    min_value : int, float, 1d or 2d array-like
        Minimum value.
    max_value : int, float, 1d or 2d array-like
        Maximum value.
    clip : bool
        Wether to clip scaled data by [0, 1] range.
    eps : float
        A constant to be added to the denominator to avoid division by zero.

    Returns
    -------
    data : np.ndarray
        Scaled data with unchaned shape.
    """
    # Use np.atleast_2d(array).T to make the array 2-dimentional by adding dummy trailing axes
    # for further broadcasting to work tracewise
    min_value = np.atleast_2d(np.array(min_value)).T
    max_value = np.atleast_2d(np.array(max_value)).T
    data = (data - min_value) / (max_value - min_value + eps)
    if clip:
        data = general_utils.clip(data, 0, 1)
    return data
