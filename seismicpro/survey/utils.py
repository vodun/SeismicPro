"""General survey processing utils"""

import numpy as np
from numba import njit, prange

from ..const import EPS, HDR_FIRST_BREAK


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


def mute_and_norm(gather, muter_col=HDR_FIRST_BREAK, rv_params=None):
    """Mute direct wave using `first_breaks_col` and normalise"""
    if muter_col not in gather.headers:
        raise RuntimeError(f"{muter_col} not in headers")

    if rv_params is None:
        rv_params = dict(n_refractors=1)

    muter = gather.calculate_refractor_velocity(first_breaks_col=muter_col, **rv_params).create_muter()
    return gather.copy().mute(muter=muter, fill_value=np.nan).scale_standard()


@njit
def rms_2_windows_ratio(data, n_begs, s_begs, win_size):
    """Compute RMS ratio for 2 windows defined by their starting samples and window size."""
    res = np.full(data.shape[0], fill_value=np.nan)

    for i, (trace, n_beg, s_beg) in enumerate(zip(data, n_begs, s_begs)):
        if n_beg > 0 and s_beg > 0:
            signal = trace[s_beg:s_beg + win_size]
            noise = trace[n_beg:n_beg + win_size]
            res[i] = np.sqrt(np.mean(signal**2)) / (np.sqrt(np.mean(noise**2)) + EPS)

    return res
