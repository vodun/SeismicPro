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

def dump_refractor_velocity(rv_list, path, encoding="UTF-8"):
    """Dump DataFrames to a file.

    Parameters
    ----------
    rv_list : iterable of RefractorVelocity or single RefractorVelocity.
        List of :class:`~refractor_velocity.RefractorVelocity` instances.
    path : str
        Path to the created file.
    encoding : str, defaults to "UTF-8"
        File encoding.
    """
    df_list = []
    for rv in to_list(rv_list):
        columns = ['name_x', 'name_y', 'coord_x', 'coord_y'] + list(rv.params.keys()) + ["max_offset"]
        data = [*rv.coords.names] + [*rv.coords.coords] + list(rv.params.values()) + [rv.max_offset]
        df_list.append(pd.DataFrame.from_dict({col: [data] for col, data in zip(columns, data)}))
    with open(path, 'w', encoding=encoding) as f:
        pd.concat(df_list).to_string(buf=f, float_format="%.2f", index=False)

def load_refractor_velocity_params(path, encoding="UTF-8"):
    """Load the coordinates and parameters of RefractorVelocity from a file.

    Parameters
    ----------
    path : str
        Path to the file.
    encoding : str, defaults to "UTF-8"
        File encoding.

    Returns
    -------
    params_list : list of dict
        Each dict in the returned list contains parameters and coords sufficient to define near-surface velocity model
        at a given locations.
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

# calculate number of refractors

def binarization_offsets(offsets, times, step=20):
    """Binarize offsets-times points."""
    bins = np.arange(0, offsets.max() + step, step=step)
    mean_offsets = np.arange(bins.shape[0]) * step + step / 2
    mean_time = np.full(shape=bins.shape[0], fill_value=np.nan)
    indices = np.digitize(offsets, bins, right=True)
    for idx in np.unique(indices):
        mean_time[idx] = times[idx == indices].mean()
    nan_mask = np.isnan(mean_time)
    return mean_offsets[~nan_mask], mean_time[~nan_mask]

def calc_max_refractors_rv(offsets, times, min_offsets_diff, min_velocity_diff, start_refractor=1,
                           max_refractors=10, init=None, bounds=None, weathering=False,
                           name=None, plot_last=False):  # name and plot_last is debug features
    """Calculate RefractorVelocity which have maximum number of refractor based on given constraints.
    """
    #pylint: disable-next=import-outside-toplevel
    from .refractor_velocity import RefractorVelocity
    name = str(name)   # debug feature
    rv = None
    for refractor in range(start_refractor, max_refractors + 1):
        min_refractor_size = np.repeat(min_offsets_diff, refractor)
        if weathering:
            min_refractor_size[0] = 1
        if offsets.max() > min_refractor_size[-1] * refractor:
            rv_last = RefractorVelocity.from_first_breaks(offsets, times, n_refractors=refractor, init=init, tol=1e-6,
                          bounds=bounds, min_velocity_step=min_velocity_diff, min_refractor_size=min_refractor_size)
            # rv_last.plot(title=rv_last.fit_result.fun)
        else:
            break
        n_points, _ = np.histogram(rv_last.offsets, bins=rv_last.piecewise_offsets)
        if (n_points > 1).all() and (rv is None or rv_last.fit_result.fun < rv.fit_result.fun):
            rv = rv_last
        else:
            break
    if plot_last and rv is not None:  # debug feature
        rv.plot(title=name)
    return rv

def calc_optimal_init(survey, min_offsets_diff=300, min_velocity_diff=300, fb_col=HDR_FIRST_BREAK, weathering=True,
                     name=None, plot_last=False):
    if len(survey.indices) < 1:
        raise ValueError("Object is empty")
    offsets = survey.headers['offset'].ravel()
    times = survey.headers[fb_col].ravel()
    # reduce points
    offsets, times = binarization_offsets(offsets, times)

    rv = calc_max_refractors_rv(offsets, times, min_offsets_diff,
                                min_velocity_diff, name=name, plot_last=plot_last)
    if weathering:  # try to find the weathering layer
        init = {'x1': 150, 'v1': rv.v1 / 2}
        bounds = {'x1': [1, 300], 'v1': [1, rv.v1]}
        start_refractor = max(rv.n_refractors, 2)
        weathering_rv = calc_max_refractors_rv(offsets, times, min_offsets_diff,
                                min_velocity_diff, start_refractor=start_refractor,
                                init=init, bounds=bounds, weathering=True,
                                name=name, plot_last=plot_last) # debug
        if weathering_rv is not None and weathering_rv.fit_result.fun < rv.fit_result.fun:
            rv = weathering_rv
    return rv.params
