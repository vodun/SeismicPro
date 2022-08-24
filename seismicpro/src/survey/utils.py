"""General survey processing utils"""

import numpy as np
from scipy.stats import ttest_ind
from numba import njit, prange
from tqdm import tqdm
import matplotlib.pyplot as plt


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

def plot_n_refractors_losses(survey, max_refractor, n_samples): # strange naming here
    loss = np.empty((n_samples, max_refractor))
    stats = np.zeros((max_refractor, 2))
    for i in tqdm(range(n_samples)):
        g = survey.sample_gather()  # TODO: add different sampling politics
        for j in range(1, max_refractor + 1):
            rv = g.calculate_refractor_velocity(n_refractors=j)
            loss[i][j-1] = rv.fit_result.fun
    for i in range(max_refractor - 1):
        stats[i + 1] = ttest_ind(loss[:, i], loss[:, i+1], nan_policy='propagate', alternative="greater")
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    ax1.plot(range(1, max_refractor + 1), np.nanmean(loss, axis=0), color="black")
    ax1.set_title(survey.name)
    ax1.set_ylabel("Mean loss")
    ax1.set_xticks(range(1, max_refractor + 1))
    ax1.set_ylim((0, np.nanmean(loss, axis=0).max() + 1))

    ax2 = ax1.twinx()
    ax2.plot(range(1, max_refractor + 1), np.clip(stats[:, 1], 0, .5), color="red")
    ax2.tick_params(axis ='y', labelcolor = "red")
    ax2.set_ylabel("Pvalue", color="red")
    ax2.set_ylim((0, .51))
