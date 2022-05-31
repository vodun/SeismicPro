"""Utility functions for coordinates and metric values processing"""

import numpy as np
import pandas as pd

from numba import njit
from scipy import signal, fft

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
    with fft.set_workers(25):
        running_mean = signal.fftconvolve(arr, [[1,1,1]], mode='valid', axes=1)/3
    return (np.abs(arr[...,1:-1] - running_mean))

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
def get_const_indicator(traces, cmpval=None):
    """Indicator of constant subsequences.

    Parameters
    ----------
    traces : np.array
        traces to analyse
    cmpval : float or None, optional
        If not None, only subsequences of this value are considered,
        otherwize, of any constant value, by default None

    Returns
    -------
    np.array
        ???????????????
    """

    if cmpval is None:
        indicator = (traces[..., 1:] == traces[..., :-1])
    else:
        indicator = (traces[..., 1:] == cmpval)

    brdr_zeros = np.zeros(traces.shape[:-1]+(1,), dtype=np.bool8)
    indicator = np.concatenate((brdr_zeros, indicator), axis=-1)

    return indicator.astype(np.int32)

@njit(nogil=True)
def get_constlen_indicator(traces, cmpval=None):
    """?????????????????????????????"""

    old_shape = traces.shape

    traces = np.atleast_2d(traces)

    indicator = get_const_indicator(traces, cmpval)


    for i in range(1, indicator.shape[-1]):
        indicator[..., i] += indicator[..., i-1] * (indicator[..., i] == 1)


    for j in range(0, indicator.shape[-2]):
        for i in range(indicator.shape[-1] - 1, 0, -1):
            if indicator[..., j, i] != 0 and indicator[..., j, i - 1] != 0:
                indicator[..., j, i - 1] = indicator[..., j, i]

    if cmpval is None:
        indicator += (indicator > 0).astype(np.int32)

    return indicator.reshape(*old_shape)
