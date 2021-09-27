import numpy as np
from numba import njit, prange


@njit(nogil=True)
def is_decreasing(window):
    for i in range(window.shape[1]):
        if window[0, i] > window[0, i + 1]:
            return True
    return False


@njit(nogil=True)
def max_std(window):
    max_std = 0
    for i in range(window.shape[1]):
        current_std = window[:, i].std()
        max_std = max(max_std, current_std)
    return max_std


@njit(nogil=True)
def max_mean_variation(window):
    max_mean_var = 0
    for i in range(window.shape[1]):
        current_mean_var = (np.mean(window[:, i]) - window[0, i]) / window[0, i]
        max_mean_var = max(max_mean_var, current_mean_var)
    return max_mean_var


VELOCITY_QC_METRICS = {
    "is_decreasing": is_decreasing,
    "max_std": max_std,
    "max_mean_variation": max_mean_variation,
}
