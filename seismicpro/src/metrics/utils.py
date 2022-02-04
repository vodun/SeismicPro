import numpy as np
import pandas as pd

from ..utils import to_list


def parse_accumulator_inputs(coords, metrics_dict, coords_cols=None):
    if not metrics_dict:
        raise ValueError("At least one metric should be passed.")

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

    # Create a dict with coordinates and passed metrics values
    res_metrics = dict(zip(coords_cols, coords.T))
    for metric_name, metric_values in metrics_dict.items():
        if isinstance(metric_values, pd.Series):
            metric_values = metric_values.values
        elif isinstance(metric_values, (list, tuple, np.ndarray)):
            metric_values = np.asarray(metric_values)
        else:
            raise TypeError(f"{metric_name} metric value must be array-like but {type(metric_values)} received")

        if len(metric_values) != len(coords):
            raise ValueError(f"The length of {metric_name} metric array must match the length of coordinates "
                                f"array ({len(coords)}) but equals {len(metric_values)}")
        res_metrics[metric_name] = metric_values

    metrics_cols = sorted(metrics_dict.keys())
    metrics = pd.DataFrame(res_metrics)[coords_cols + metrics_cols]
    return metrics, coords_cols, metrics_cols
