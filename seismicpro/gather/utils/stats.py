import numpy as np
from numba import njit, prange


@njit(nogil=True, parallel=True)
def numba_rms(traces):
    temp = np.empty(len(traces), dtype=np.float32)
    for i in prange(len(traces)):
        temp[i] = np.nanmean(traces[i]**2)**.5
    return temp


@njit(nogil=True, parallel=True)
def numba_abs(traces):
    temp = np.empty(len(traces), dtype=np.float32)
    for i in prange(len(traces)):
        temp[i] = np.nanmean(np.abs(traces[i]))
    return temp
