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

    # Ensure that all params are non-negative
    params = np.clip(np.atleast_2d(params), 0, None)

    # Ensure that velocities of refractors and crossover offsets are non-decreasing
    n_refractors = params.shape[1] // 2
    np.maximum.accumulate(params[:, n_refractors:], axis=1, out=params[:, n_refractors:])
    np.maximum.accumulate(params[:, 1:n_refractors], axis=1, out=params[:, 1:n_refractors])

    if is_1d:
        return params[0]
    return params

def dump_refractor_velocities(refractor_velocities, path, encoding="UTF-8"):
    """Dump the parameters of passed velocity models to a file.

    Parameters
    ----------
    refractor_velocities : RefractorVelocity or iterable of RefractorVelocity.
        Refractor Velocity instances to dump to the file.
    path : str
        Path to the created file.
    encoding : str, optional, defaults to "UTF-8"
        File encoding.
    """
    rv_list = to_list(refractor_velocities)
    columns = ['name_x', 'name_y', 'coord_x', 'coord_y'] + list(rv_list[0].params.keys())
    data = np.empty((len(rv_list), len(columns)), dtype=object)
    for i, rv in enumerate(rv_list):
        data[i] = [*rv.coords.names] + [*rv.coords.coords] + list(rv.params.values())
    df =  pd.DataFrame(data, columns=columns)
    df = df.infer_objects()
    df.to_string(buf=path, float_format="%.2f", index=False, encoding=encoding)

def load_refractor_velocities(path, encoding="UTF-8"):
    """Load the coordinates and parameters of the velocity models from a file.

    Parameters
    ----------
    path : str
        Path to the file.
    encoding : str, optional, defaults to "UTF-8"
        File encoding.

    Returns
    -------
    rv_list : list of RefractorVelocity
        List of the near-surface velocity models that are created from the parameters and coords loaded from the file.
    """
    #pylint: disable-next=import-outside-toplevel
    from .refractor_velocity import RefractorVelocity  # import inside to avoid the circular import
    df = pd.read_csv(path, sep=r'\s+', encoding=encoding)
    params_names = df.columns[4:]
    params_values = df[df.columns[4:]].to_numpy()

    coords_names = df[df.columns[:2]].to_numpy()
    coords_values = df[df.columns[2:4]].to_numpy()
    rv_list = []
    for coords_name, coords_value, params_value in zip(coords_names, coords_values, params_values):
        params = dict(zip(params_names, params_value))
        coords = Coordinates(names=coords_name, coords=coords_value)
        rv_list.append(RefractorVelocity(coords=coords, **params))
    return rv_list
