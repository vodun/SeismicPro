"""General survey processing utils"""

import numpy as np
from numba import njit, prange


@njit(nogil=True)
def calculate_trace_stats(trace):
    """Calculate min, max, mean and var of trace amplitudes."""
    trace_min = trace_max = trace[0]

    # Traces are generally centered around zero so variance is calculated in a single pass by accumulating sum and
    # sum of squares of trace amplitudes as float64 for numerical stability
    trace_sum = np.float64(trace[0])
    trace_sum_sq = trace_sum**2

    for sample in trace[1:]:
        trace_min = min(sample, trace_min)
        trace_max = max(sample, trace_max)
        sample64 = np.float64(sample)
        trace_sum += sample64
        trace_sum_sq += sample64**2
    trace_mean = trace_sum / len(trace)
    trace_var = trace_sum_sq / len(trace) - trace_mean**2
    return trace_min, trace_max, trace_mean, trace_var


@njit(nogil=True, parallel=True)
def ibm_to_ieee(hh, hl, lh, ll):
    """Convert 4 arrays representing individual bytes of IBM 4-byte floats into a single array of floats. Input arrays
    are ordered from most to least significant bytes and have `np.uint8` dtypes. The result is returned as an
    `np.float32` array."""
    res = np.empty_like(hh, dtype=np.float32)
    for i in prange(res.shape[0]):  # pylint: disable=not-an-iterable
        for j in prange(res.shape[1]):  # pylint: disable=not-an-iterable
            mant = (((np.int32(hl[i, j]) << 8) | lh[i, j]) << 8) | ll[i, j]
            if hh[i, j] & 0x80:
                mant = -mant
            exp16 = (np.int8(hh[i, j]) & np.int8(0x7f)) - 70
            res[i, j] = mant * 16.0**exp16
    return res


@njit(nogil=True)
def create_supergather_index(centers, size):
    """Create a mapping from supergather centers to coordinates of gathers in them.

    Examples
    --------
    >>> centers = np.array([[5, 5], [8, 9]])
    >>> size = (3, 3)
    >>> create_supergather_index(centers, size)
    array([[ 5,  5,  4,  4],
           [ 5,  5,  4,  5],
           [ 5,  5,  4,  6],
           [ 5,  5,  5,  4],
           [ 5,  5,  5,  5],
           [ 5,  5,  5,  6],
           [ 5,  5,  6,  4],
           [ 5,  5,  6,  5],
           [ 5,  5,  6,  6],
           [ 8,  9,  7,  8],
           [ 8,  9,  7,  9],
           [ 8,  9,  7, 10],
           [ 8,  9,  8,  8],
           [ 8,  9,  8,  9],
           [ 8,  9,  8, 10],
           [ 8,  9,  9,  8],
           [ 8,  9,  9,  9],
           [ 8,  9,  9, 10]])

    Parameters
    ----------
    centers : 2d np.ndarray with 2 columns
        Coordinates of supergather centers.
    size : tuple with 2 elements
        Supergather size along inline and crossline axes. Measured in lines.

    Returns
    -------
    mapping : 2d np.ndarray with 4 columns
        Coordinates of supergather centers in the first 2 columns and coordinates of the included gathers in the last
        two columns.
    """
    area_size = size[0] * size[1]
    shifts_i = np.arange(size[0]) - size[0] // 2
    shifts_x = np.arange(size[1]) - size[1] // 2
    mapping = np.empty((len(centers) * area_size, 4), dtype=centers.dtype)
    for ix, (i, x) in enumerate(centers):
        for ix_i, shift_i in enumerate(shifts_i):
            for ix_x, shift_x in enumerate(shifts_x):
                row = np.array([i, x, i + shift_i, x + shift_x])
                mapping[ix * area_size + ix_i * size[1] + ix_x] = row
    return mapping
