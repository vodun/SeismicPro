"""General survey processing utils"""

import numpy as np
from numba import njit, prange

from ..refractor_velocity import RefractorVelocity


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

def binarization_offsets(offsets, times, step=20):
    bins = np.arange(0, offsets.max() + step, step=step)
    mean_offsets = np.arange(bins.shape[0]) * step + step / 2
    mean_time = np.full(shape=bins.shape[0], fill_value=np.nan)
    indices = np.digitize(offsets, bins, right=True)
    for idx in np.unique(indices):
        mean_time[idx] = times[idx == indices].mean()
    nan_mask = np.isnan(mean_time)
    return mean_offsets[~nan_mask], mean_time[~nan_mask]

def _is_refractors_valid(rv, min_offsets_diff, min_velocity_diff, min_points_percentile):
    cross_offsets_diff = np.diff(rv.piecewise_offsets[1:]) # except 0
    velocity_diff = np.diff(list(rv.params.values())[rv.n_refractors:])
    points, _ = np.histogram(rv.offsets, rv.piecewise_offsets)
    points_percentile = points / points.sum()
    if (np.all(cross_offsets_diff > min_offsets_diff) and np.all(velocity_diff > min_velocity_diff) and \
            np.all(points_percentile > min_points_percentile)):
        return True
    return False

def calc_max_refractors_params(offsets, times, max_refractors, min_offsets_diff, min_velocity_diff,
                               min_points_percentile, weathering, max_weathering_offset=300, name=None,
                               plot_last=False): # name and plot_last is debug features
    for refractor in range(1, max_refractors + 1):
        rv_last = RefractorVelocity.from_first_breaks(offsets, times, n_refractors=refractor)
        if _is_refractors_valid(rv_last, min_offsets_diff, min_velocity_diff, min_points_percentile):
            rv = rv_last
        elif weathering:
            rv_last = RefractorVelocity.from_first_breaks(offsets, times, n_refractors=refractor,
                                                          bounds={'x1': [1, max_weathering_offset]})
            if _is_refractors_valid(rv_last, min_offsets_diff, min_velocity_diff, min_points_percentile):
                rv = rv_last
                name = str(name) + '\nweathering layer detected'   # debug feature
        else:
            break
    name = str(name) + '\n' # debug feature
    if plot_last and rv is not None:   # debug feature
        rv.plot(title=name)
    return rv.params
