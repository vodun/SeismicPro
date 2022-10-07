"""Miscellaneous utility functions for refractor velocity estimation"""

import numpy as np
import pandas as pd

from ..utils import Coordinates, to_list
from ..const import HDR_FIRST_BREAK


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

def dump_refractor_velocity(rv_list, path, encoding="UTF-8"):
    """Dump the parameters of passed velocity models to a file.

    Parameters
    ----------
    rv_list : RefractorVelocity or iterable of RefractorVelocity.
        Refractor Velocity instances to dump to the file.
    path : str
        Path to the created file.
    encoding : str, optional, defaults to "UTF-8"
        File encoding.
    """
    rv_list = to_list(rv_list)
    columns = ['name_x', 'name_y', 'coord_x', 'coord_y'] + list(rv_list[0].params.keys())
    coords_names = np.empty((len(rv_list), 2), dtype=object)
    coords_values = np.empty((len(rv_list), 2), dtype=np.int32)
    params_values = np.empty((len(rv_list), len(list(rv_list[0].params.keys()))), dtype=np.float32)
    for i, rv in enumerate(rv_list):
        coords_names[i] = rv.coords.names
        coords_values[i] = rv.coords.coords
        params_values[i] = list(rv.params.values())
    df = pd.concat([pd.DataFrame(coords_names), pd.DataFrame(coords_values), pd.DataFrame(params_values)], axis=1)
    df.columns = columns
    df.to_string(buf=path, float_format="%.2f", index=False, encoding=encoding)

def load_refractor_velocity(path, encoding="UTF-8"):
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
    coords_names = df[df.columns[:2]].to_numpy()
    coords_values = df[df.columns[2:4]].to_numpy()
    params_values = df[df.columns[4:]].to_numpy()
    params_names = df.columns[4:]
    rv_list = []
    for i in range(df.shape[0]):
        if np.isnan(params_values[i, -1]):
            raise ValueError(f"Unsufficient parameters in the row {i}.")
        params = dict(zip(params_names, params_values[i]))
        params['coords'] = Coordinates(names=coords_names[i], coords=coords_values[i])
        rv_list.append(RefractorVelocity(**params))
    return rv_list

# calculate number of refractors

def binarize_df(df, first_breaks_col, step=20):
    df['bins'] = df['offset'] // step
    res = df[['bins', first_breaks_col]].groupby(by='bins').mean()
    res['offset'] = res.index * step + step / 2
    return res['offset'].to_numpy(), res[first_breaks_col].to_numpy()

def calc_max_refractors_rv(offsets, times, min_refractor_size, min_velocity_step, start_refractor=1,
                           max_refractors=10, init=None, bounds=None, weathering=False):
    """Calculate RefractorVelocity which have maximum number of refractor based on given constraints.
    """
    #pylint: disable-next=import-outside-toplevel
    from .refractor_velocity import RefractorVelocity
    rv = None
    for refractor in range(start_refractor, max_refractors + 1):
        min_refractor_size_ = np.full(min_refractor_size, refractor)
        if weathering:
            min_refractor_size_[0] = 1
        if offsets.max() < min_refractor_size_[-1] * refractor:
            break
        rv_last = RefractorVelocity.from_first_breaks(offsets, times, n_refractors=refractor, init=init, tol=1e-6,
                          bounds=bounds, min_velocity_step=min_velocity_step, min_refractor_size=min_refractor_size_)
        rv_last.plot(title=rv_last.fit_result.fun)  # debug
        n_points, _ = np.histogram(rv_last.offsets, bins=rv_last.piecewise_offsets)
        if not ((n_points > 1).all() and (rv is None or rv_last.fit_result.fun < rv.fit_result.fun)):
            break
        rv = rv_last
    return rv

def calc_optimal_velocity(survey, min_offsets_diff=300, min_velocity_diff=300, first_breaks_col=HDR_FIRST_BREAK,
                          weathering=False):
    """Calculate one velocity model describe passed survey."""
    if survey.n_gathers < 1:  # need if the func calls separately from `RefractorVelocityField.from_survey`
        raise ValueError("Survey is empty.")
    # # reduce points
    offsets, times = binarize_df(survey.headers[['offset', first_breaks_col]], first_breaks_col=first_breaks_col)
    rv = calc_max_refractors_rv(offsets, times, min_offsets_diff, min_velocity_diff)
    if weathering:  # try to find the weathering layer
        init = {'x1': 150, 'v1': rv.v1 / 2}
        bounds = {'x1': [1, 300], 'v1': [1, rv.v1]}
        start_refractor = max(rv.n_refractors, 2)
        rv_weathering = calc_max_refractors_rv(offsets, times, min_offsets_diff, min_velocity_diff,
                                start_refractor=start_refractor, init=init, bounds=bounds, weathering=True)
        if rv_weathering is not None and rv_weathering.fit_result.fun < rv.fit_result.fun:
            rv = rv_weathering
    return rv
