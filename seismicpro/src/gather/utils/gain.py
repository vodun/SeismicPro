"""Implements functions for gather gain amplifications"""

import numpy as np
from numba import njit, prange

@njit(nogil=True, parallel=True)
def apply_agc(data, window_size=125, mode='abs'):
    """ TODO """
    n_traces, trace_len = data.shape
    win_left, win_right = window_size // 2, window_size - window_size // 2
    # start is the first trace index that fits the full window, end is the last.
    # AGC coefficients before start and after end are extrapolated.
    start, end = win_left, trace_len - win_right

    for i in prange(n_traces):  # pylint: disable=not-an-iterable
        trace = data[i]
        trace = np.power(trace, 2) if mode=='rms' else np.abs(trace)

        amplitudes_cumsum = np.cumsum(trace)
        nonzero_counts_cumsum = np.cumsum(trace!=0)

        coefs = np.empty_like(trace)
        coefs[start:end] = ((nonzero_counts_cumsum[:-window_size] - nonzero_counts_cumsum[window_size:])
                            / (amplitudes_cumsum[:-window_size] - amplitudes_cumsum[window_size:] + 1e-15))
        # Extrapolate AGC coefs for trace indices that don't fit the full window
        coefs[:start] = coefs[start]
        coefs[end:] = coefs[end-1]

        coefs = np.sqrt(coefs) if mode=='rms' else coefs
        data[i] *= coefs
    return data


@njit(nogil=True, parallel=True)
def calculate_sdc_coefficient(v_pow, velocities, t_pow, times):
    """ TODO """
    sdc_coefficient = velocities**v_pow * times**t_pow
    # Scale sdc_coefficient to be 1 at maximum time
    sdc_coefficient /= sdc_coefficient[-1]
    return sdc_coefficient


@njit(nogil=True, parallel=True)
def apply_sdc(data, v_pow, velocities, t_pow, times):
    """ TODO """
    sdc_coefficient = calculate_sdc_coefficient(v_pow, velocities, t_pow, times)
    for i in prange(len(data)):  # pylint: disable=not-an-iterable
        data[i] *= sdc_coefficient
    return data

@njit(nogil=True, parallel=True)
def undo_sdc(data, v_pow, velocities, t_pow, times):
    """ TODO """
    sdc_coefficient = calculate_sdc_coefficient(v_pow, velocities, t_pow, times)
    for i in prange(len(data)):  # pylint: disable=not-an-iterable
        data[i] /= sdc_coefficient
    return data
