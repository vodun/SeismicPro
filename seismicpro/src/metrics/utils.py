"""Utility functions for coordinates and metric values processing"""

import numpy as np
import pandas as pd

from ..utils import to_list


def parse_coords(coords, coords_cols=None):
    """Cast given `coords` to a 2d `np.ndarray` with shape [n_coords, 2] and try inferring names of both coordinates if
    `coords_cols` is not passed."""
    if isinstance(coords, pd.DataFrame):
        data_coords_cols = coords.columns
        coords = coords.values
    elif isinstance(coords, pd.Index):
        data_coords_cols = coords.names
        if None in data_coords_cols:  # Undefined index names, fallback to a default
            data_coords_cols = None
        coords = coords.to_frame().values
    elif isinstance(coords, (list, tuple, np.ndarray)):
        # Try inferring coordinates columns if passed coords is an iterable of namedtuples
        data_coords_cols_set = {getattr(coord, "_fields", None) for coord in coords}
        if len(data_coords_cols_set) != 1:
            raise ValueError("Coordinates from different header columns were passed")
        data_coords_cols = data_coords_cols_set.pop()
        coords = np.asarray(coords)
        # If coords is an array of arrays, convert it to an array with numeric dtype
        coords = np.array(coords.tolist()) if coords.ndim == 1 else coords
    else:
        raise ValueError(f"Unsupported type of coords {type(coords)}")

    coords_cols = coords_cols or data_coords_cols or ("X", "Y")
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

    metric_name = metric_name or data_metric_name or getattr(metric_type, "name") or "metric"
    return metric_values, metric_name
