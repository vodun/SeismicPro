"""Miscellaneous utility functions for refractor velocity estimation"""

import numpy as np
import pandas as pd

from ..utils import Coordinates, to_list


def get_param_names(n_refractors):
    """Return names of parameters of a near-surface velocity model describing given number of refractors."""
    return ["t0"] + [f"x{i}" for i in range(1, n_refractors)] + [f"v{i}" for i in range(1, n_refractors + 1)]


def postprocess_params(params):
    """Postprocess array of parameters of a near-surface velocity model so that the following constraints are
    satisfied:
    - Intercept time is non-negative,
    - Crossover offsets are non-negative and increasing,
    - Velocities of refractors are non-negative and increasing.
    """
    is_1d = (params.ndim == 1)
    params = np.atleast_2d(params).copy()
    n_refractors = params.shape[1] // 2

    # Ensure that t0 is non-negative
    np.clip(params[:, 0], 0, None, out=params[:, 0])

    # Ensure that velocities of refractors are non-negative and increasing
    velocities = params[:, n_refractors:]
    np.clip(velocities[:, 0], 0, None, out=velocities[:, 0])
    np.maximum.accumulate(velocities, axis=1, out=velocities)

    # Ensure that crossover offsets are non-negative and increasing
    if n_refractors > 1:
        cross_offsets = params[:, 1:n_refractors]
        np.clip(cross_offsets[:, 0], 0, None, out=cross_offsets[:, 0])
        np.maximum.accumulate(cross_offsets, axis=1, out=cross_offsets)

    if is_1d:
        return params[0]
    return params

def calc_df_to_dump(rv_list):
    """Calculate a DataFrame with coordinates and parameters of the passed RefractorVelocity.

    Parameters
    ----------
    rv_list : iterable of :class:`~refractor_velocity.RefractorVelocity`
        List of RefractorVelocity instances.

    Returns
    -------
    df : DataFrame
        DataFrame with coordinates and parameters of passed RefractorVelocity instances.
    """
    df_list = []
    for rv in to_list(rv_list):
        columns = ['name_x', 'name_y', 'coord_x', 'coord_y'] + list(rv.params.keys()) + ["max_offset"]
        data = [*rv.coords.names] + [*rv.coords.coords] + list(rv.params.values()) + [rv.max_offset]
        df_list.append(pd.DataFrame.from_dict({col: [data] for col, data in zip(columns, data)}))
    return pd.concat(df_list)

def dump_df(df, path, encoding, min_col_size):
    """Dump DataFrames to a file.

    Parameters
    ----------
    df : DataFrame
        DataFrame to dump.
    path : str
        Path to the created file.
    encoding : str
        File encoding.
    min_col_size : int
        Minimum size of each columns in the resulting file.
    """
    with open(path, 'w', encoding=encoding) as f:
        df.to_string(buf=f, col_space=min_col_size, float_format="%.2f", index=False)

def load_rv(path, encoding):
    """Load the coordinates and parameters of RefractorVelocity from a file.

    Parameters
    ----------
    path : str
        Path to the file.
    encoding : str
        File encoding.

    Returns
    -------
    params_list : list of dict,
        List of dict. Each dict contains parameters and coords sufficient to define near-surface velocity model at one
        or more locations.
    """
    df = pd.read_csv(path, sep=r'\s+', encoding=encoding)
    params_names = df.columns[4:]
    params_list = []
    for row in df.to_numpy():
        if np.isnan(row[-1]):
            raise ValueError(f"Unsufficient parameters in the row {row}.")
        params = dict(zip(params_names, row[4:]))
        params['coords'] = Coordinates(names=tuple(row[:2]), coords=tuple(row[2:4].astype(int)))
        params_list.append(params)
    return params_list
