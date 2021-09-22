import numpy as np
from numba import njit, prange


VELOCITY_QC_METRICS = ["is_decreasing", "max_standard_deviation", "max_relative_variation"]


@njit(nogil=True)
def is_decreasing(window):
    for i in range(window.shape[1]):
        if window[0, i] > window[0, i + 1]:
            return True
    return False


@njit(nogil=True)
def max_standard_deviation(window):
    max_std = 0
    for i in range(window.shape[1]):
        current_std = window[:, i].std()
        max_std = max(max_std, current_std)
    return max_std


@njit(nogil=True)
def max_relative_variation(window):
    max_mean_var = 0
    for i in range(window.shape[1]):
        current_mean_var = abs(np.mean(window[1:, i]) - window[0, i]) / window[0, i]
        max_mean_var = max(max_mean_var, current_mean_var)
    return max_mean_var
