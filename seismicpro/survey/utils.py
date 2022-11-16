"""General survey processing utils"""

import warnings

import numpy as np
from numba import njit, prange

from matplotlib import pyplot as plt
from matplotlib import colors, cm

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
def fill_leading_nulls(arr):
    """"Fill leading null values of array's row with the first non null value in a row."""

    n_samples = arr.shape[1]

    for i in range(arr.shape[0]):
        nan_indices = np.nonzero(np.isnan(arr[i]))[0]
        if len(nan_indices) > 0:
            j = nan_indices[-1] + 1
            if j < n_samples:
                arr[i, :j] = arr[i, j]

@njit(nogil=True)
def get_val_subseq(traces, cmpval):
    """Indicator of constant subsequences equal to given value."""

    old_shape = traces.shape
    traces = np.atleast_2d(traces)

    indicators = (traces == cmpval).astype(np.int16)

    for t, indicator in enumerate(indicators):
        counter = 0
        for i, sample in enumerate(indicator):
            if sample == 1:
                counter += 1
            else:
                if counter > 1:
                    indicators[t, i - counter: i] = counter
                counter = 0

        if counter > 1:
            indicators[t, -counter:] = counter

    return indicators.reshape(*old_shape)

@njit(nogil=True)
def get_const_subseq(traces):
    """Indicator of constant subsequences."""

    old_shape = traces.shape
    traces = np.atleast_2d(traces)

    indicators = np.full_like(traces, fill_value=0, dtype=np.int16)
    indicators[:, 1:] = (traces[:, 1:] == traces[:, :-1]).astype(np.int16)
    for t, indicator in enumerate(indicators):
        counter = 0
        for i, sample in enumerate(indicator):
            if sample == 1:
                counter += 1
            else:
                if counter > 0:
                    indicators[t, i - counter - 1:i] = counter + 1
                counter = 0

        if counter > 0:
            indicators[t, -counter - 1:] = counter + 1

    return indicators.reshape(*old_shape)


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


def deb_wiggle_plot(sur_std, ax, arr, labels, norm_tracewize, std=0.1, **kwargs):
    """ Wiggle plot with labels for each trace. The color of each trace corresponds to its std. Used for debugging. """

    y_coords = np.arange(arr.shape[1])

    norm = colors.LogNorm(vmin=min(arr.std(axis=1)/sur_std), vmax=max(arr.std(axis=1)/sur_std))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        axis = 1 if norm_tracewize else None
        traces = std * ((arr - np.nanmean(arr, axis=axis, keepdims=True)) /
                        (np.nanstd(arr, axis=axis, keepdims=True) + 1e-10))

    for i, (trace, label) in enumerate(zip(traces, labels)):
        ax.plot(i + trace, y_coords, 'k', alpha=0.1, **kwargs)

        rgba_color = cm.viridis(norm(arr[i].std()/sur_std))
        ax.fill_betweenx(y_coords, i, i + trace, where=(trace > 0), color=rgba_color, alpha=0.5, **kwargs)
        ax.text(i, 0, label, size='x-small')

    ax.invert_yaxis()
    ax.set_title('Trace-wise norm' if norm_tracewize else 'Batch-wise norm')
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)


def deb_indices(sur, indices, size, mode='wiggle', select_mode='sample', title=None, figsize=(15, 7), std=0.1):
    """" Plot `size` traces from `sur` with given `indices`. Used for debuging.  """

    if len(indices) == 0:
        warnings.warn('empty subset!')
        return

    size = min(size, len(indices))

    if size == len(indices):
        ind = indices
    elif select_mode == 'sample':
        ind = sorted(np.random.choice(indices, size=size, replace=False))
    elif select_mode == 'subseq':
        start_ind = np.random.choice(len(indices) - size + 1)
        ind = indices[start_ind: start_ind + size]
    else:
        raise ValueError(f"mode can be only `sample` or `subset`, but {mode} recieved")

    gathers = [sur.get_gather(i) for i in ind]
    traces = np.concatenate([g.data for g in gathers], axis=0)

    if mode == 'wiggle':
        labels = ['\n'.join([str(i),
                             str(g.headers.FieldRecord.values[-1]),
                             str(g.headers.TraceNumber.values[-1])])
                  for g, i in zip(gathers, ind)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        deb_wiggle_plot(sur.std, ax1, traces, labels, norm_tracewize=False, std=std)
        deb_wiggle_plot(sur.std, ax2, traces, labels, norm_tracewize=True, std=std)
    elif mode == 'imshow':
        fig, ax = plt.subplots(figsize=figsize)
        cv = max(np.abs(np.quantile(traces, (0.1, 0.9))))
        ax.imshow(traces.T, vmin=-cv, vmax=cv, cmap='gray')

    if title:
        title += '\n'
    else:
        title = ''

    fig.suptitle(title + f"{size} of {len(indices)}")
