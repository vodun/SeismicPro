import numpy as np
import pandas as pd

from ..utils import to_list


def parse_coords(coords, coords_cols=None):
    default_coords_cols = ("X", "Y") if coords_cols is None else coords_cols
    if isinstance(coords, pd.DataFrame):
        coords_cols = coords.columns
        coords = coords.values
    elif isinstance(coords, pd.Index):
        coords_cols = coords.names
        if None in coords_cols:  # Undefined index names, fallback to a default
            coords_cols = default_coords_cols
        coords = coords.to_frame().values
    elif isinstance(coords, (list, tuple, np.ndarray)):
        # Try inferring coordinates columns if passed coords is an iterable of namedtuples
        coords_cols_set = {getattr(coord, "_fields", tuple(default_coords_cols)) for coord in coords}
        if len(coords_cols_set) != 1:
            raise ValueError("Coordinates from different header columns were passed")
        coords_cols = coords_cols_set.pop()
        coords = np.asarray(coords)
        # If coords is an array of arrays, convert it to an array with numeric dtype and check its shape
        coords = np.array(coords.tolist()) if coords.ndim == 1 else coords
    else:
        raise ValueError(f"Unsupported type of coords {type(coords)}")
    coords_cols = to_list(coords_cols)

    if coords.ndim != 2:
        raise ValueError("Coordinates array must be 2-dimensional.")
    if coords.shape[1] != 2:
        raise ValueError("Coordinates array must have shape (N, 2), where N is the number of elements"
                         f" but an array with shape {coords.shape} was given")
    return coords, coords_cols


def parse_metric_values(metric_values, metric_name=None, metric_type=None):
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
