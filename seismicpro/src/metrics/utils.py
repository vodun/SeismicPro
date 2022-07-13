"""Utility functions for coordinates and metric values processing"""

import warnings

import numpy as np
import scipy as sp
import pandas as pd

from numba import njit

from matplotlib import pyplot as plt
from matplotlib import colors, cm

from ..utils import to_list, get_first_defined, Coordinates


def parse_coords(coords, coords_cols=None):
    """Cast given `coords` to a 2d `np.ndarray` with shape [n_coords, 2] and try inferring names of both coordinates if
    `coords_cols` is not passed."""
    if isinstance(coords, pd.DataFrame):
        data_coords_cols = coords.columns.tolist()
        coords = coords.values
    elif isinstance(coords, pd.Index):
        data_coords_cols = coords.names
        if None in data_coords_cols:  # Undefined index names, fallback to a default
            data_coords_cols = None
        coords = coords.to_frame().values
    elif isinstance(coords, (list, tuple, np.ndarray)):
        data_coords_cols = None

        # Try inferring coordinates columns if passed coords is an iterable of Coordinates
        if all(isinstance(coord, Coordinates) for coord in coords):
            data_coords_cols_set = {coord.names for coord in coords}
            if len(data_coords_cols_set) != 1:
                raise ValueError("Coordinates from different header columns were passed")
            data_coords_cols = data_coords_cols_set.pop()

        # Cast coords to an array. If coords is an array of arrays, convert it to an array with numeric dtype.
        coords = np.asarray(coords)
        coords = np.array(coords.tolist()) if coords.ndim == 1 else coords
    else:
        raise ValueError(f"Unsupported type of coords {type(coords)}")

    coords_cols = get_first_defined(coords_cols, data_coords_cols, ("X", "Y"))
    coords_cols = to_list(coords_cols)
    if len(coords_cols) != 2:
        raise ValueError(f"List of coordinates names must have length 2 but {len(coords_cols)} was given.")
    if coords.ndim != 2:
        raise ValueError("Coordinates array must be 2-dimensional.")
    if coords.shape[1] != 2:
        raise ValueError(f"Each item of coords must have length 2 but {coords.shape[1]} was given.")
    return coords, coords_cols


def parse_metric_values(metric_values, metric_name=None, metric_type=None):
    """Cast given `metric_values` to a 1d `np.ndarray` and try inferring metric name from `metric_values` and
    `metric_type` if `metric_name` is not given."""
    err_msg = "Metric values must be a 1-dimensional array-like."
    if isinstance(metric_values, pd.DataFrame):
        columns = metric_values.columns
        if len(columns) != 1:
            raise ValueError(err_msg)
        data_metric_name = columns[0]
        metric_values = metric_values.values[:, 0]
    elif isinstance(metric_values, pd.Series):
        data_metric_name = metric_values.name
        metric_values = metric_values.values
    else:
        data_metric_name = None
        metric_values = np.array(metric_values)
        if metric_values.ndim != 1:
            raise ValueError(err_msg)

    metric_name = get_first_defined(metric_name, data_metric_name, getattr(metric_type, "name"), "metric")
    return metric_values, metric_name


def calc_spikes(arr):
    """Calculate spikes indicator."""
    with sp.fft.set_workers(25):
        running_mean = sp.signal.fftconvolve(arr, [[1,1,1]], mode='valid', axes=1)/3

    running_mean = arr.copy()
    running_mean[:, :-1] += arr[:, 1:]
    running_mean[:, 1:] += arr[:, :-1]
    running_mean /= 3


    ### check strided.get_window 

    return np.abs(arr[...,1:-1] - running_mean)

@njit
def fill_nulls(arr):
    """"Fill leading null values of array's row with the first non null value in a row."""

    n_samples = arr.shape[1]

    for i in range(arr.shape[0]):
        nan_indices = np.nonzero(np.isnan(arr[i]))[0]
        if len(nan_indices) > 0:
            j = nan_indices[-1]+1
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
    indicators[:, 1:] = (traces[..., 1:] == traces[..., :-1]).astype(np.int16)
    for t, indicator in enumerate(indicators):
        counter = 0
        for i, sample in enumerate(indicator):
            if sample == 1:
                counter += 1
            else:
                if counter > 0:
                    indicators[t, i - counter - 1: i] = counter + 1
                counter = 0

        if counter > 0:
            indicators[t, -counter - 1:] = counter + 1

    return indicators.reshape(*old_shape)


def deb_wiggle_plot(sur, ax, arr, labels, norm_tracewize, std=0.1, **kwargs):

    n_traces, n_samples = arr.shape

    y_coords = np.arange(n_samples)


    norm = colors.LogNorm(vmin=min(arr.std(axis=1))/sur.std, vmax=max(arr.std(axis=1))/sur.std)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        axis = 1 if norm_tracewize else None
        traces = std * ((arr - np.nanmean(arr, axis=axis, keepdims=True)) /
                        (np.nanstd(arr, axis=axis, keepdims=True) + 1e-10))

    for i, (trace, label) in enumerate(zip(traces, labels)):
        ax.plot(i + trace, y_coords, 'k', alpha=0.1, **kwargs)

        rgba_color, alpha = (cm.viridis(norm(arr[i].std()/sur.std)), 0.5)
        cbar = ax.fill_betweenx(y_coords, i, i + trace, where=(trace > 0), color=rgba_color, alpha=alpha, **kwargs)
        ax.text(i, 0, label, size='x-small')
        # ax.text(i, y_coords[-1]+10, f"", size='x-small')

    ax.invert_yaxis()
    ax.set_title('Trace-wise norm' if norm_tracewize else 'Batch-wise norm')
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)


def deb_indices(sur, indices, size, mode='wiggle', select_mode='sample', title=None, figsize=(15, 7), std=0.1):

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
        deb_wiggle_plot(sur, ax1, traces, labels, norm_tracewize=False, std=std)
        deb_wiggle_plot(sur, ax2, traces, labels, norm_tracewize=True, std=std)
    elif mode == 'imshow':
        fig, ax = plt.subplots(figsize=figsize)
        cv = max(np.abs(np.quantile(traces, (0.1, 0.9))))
        ax.imshow(traces.T, vmin=-cv, vmax=cv, cmap='gray')

    if title:
        title += '\n'
    else:
        title = ''

    fig.suptitle(title + f"{size} of {len(indices)}")