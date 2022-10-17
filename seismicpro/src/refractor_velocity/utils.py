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

def dump_refractor_velocities(refractor_velocities, path, encoding="UTF-8"):
    """Dump parameters of passed velocity models to a file.

    Parameters
    ----------
    refractor_velocities : RefractorVelocity or iterable of RefractorVelocities.
        RefractorVelocity instances to dump to the file.
    path : str
        Path to the created file.
    encoding : str, optional, defaults to "UTF-8"
        File encoding.
    """
    rv_list = to_list(refractor_velocities)
    columns = ['name_x', 'name_y', 'x', 'y'] + list(rv_list[0].params.keys())
    data = np.empty((len(rv_list), len(columns)), dtype=object)
    for i, rv in enumerate(rv_list):
        data[i] = [*rv.coords.names] + [*rv.coords.coords] + list(rv.params.values())
    df = pd.DataFrame(data, columns=columns).convert_dtypes()
    df.to_string(buf=path, float_format=lambda x: f"{x:.2f}", index=False, encoding=encoding)

def load_refractor_velocities(path, encoding="UTF-8"):
    """Load coordinates and parameters of the velocity models from a file.

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
    df = pd.read_csv(path, sep=r'\s+', encoding=encoding).convert_dtypes()
    params_names = df.columns[4:]
    return [RefractorVelocity(**dict(zip(params_names, row[4:])), coords=Coordinates(row[2:4], row[:2]))
            for row in df.itertuples(index=False)]

# calculate optimal near-surface velocity model

def reduce_mean_df(df, x='offset', y=HDR_FIRST_BREAK, step=20):
    """Reduce DataFrame columns `x` and `y`."""
    df['bins'] = df[x] // step
    res = df.groupby(by='bins').mean()
    return res[x].to_numpy(), res[y].to_numpy()

def calc_optimal_velocity(offsets, times, init=None, bounds=None,  min_velocity_step=300, min_refractor_size=300,
                          loss="L1", huber_coef=20, min_refractors=1, max_refractors=10, find_weathering=False,
                          debug=False):
    """Calculate a near-surface velocity model with a number of refractors that give minimal loss.

    Parameters
    ----------
    offsets : 1d ndarray
        Offsets of traces. Measured in meters.
    times : 1d ndarray
        Time of first break for each trace. Measured in milliseconds.
    init : dict, optional
        Initial values of model parameters.
    bounds : dict, optional
        Lower and upper bounds of model parameters.
    min_velocity_step : int, optional, defaults to 300
        Minimum difference between velocities of two adjacent refractors.
    min_refractor_size : int, optional, defaults to 300
        Minimum offset range covered by each refractor.
    loss : str, optional, defaults to "L1"
        Loss function to be minimized. Should be one of "MSE", "huber", "L1", "soft_L1", or "cauchy".
    huber_coef : float, optional, default to 20
        Coefficient for Huber loss function.
    min_refractors : int, optional, defaults to 1
        Minimum number of refractors for the expected velocity model.
    max_refractors : int, optional, defaults to 10
        Maximum number of refractors for the expected velocity model.
    find_weathering : bool, optional, defaults to False.
        If True set `min_refractor_size` for the first refractor to 1.

    Returns
    -------
    rv : RefractorVelocity or None
        A near-surface velocity model with optimal number of refractors. Returns None if no velocity model is possible
        for the given parameters.
    """
    #pylint: disable-next=import-outside-toplevel
    from .refractor_velocity import RefractorVelocity  # avoid circulat import
    rv = None
    for refractor in range(min_refractors, max_refractors + 1):
        min_refractor_size_vec = np.full(refractor, min_refractor_size)
        if find_weathering:
            min_refractor_size_vec[0] = 1
        if offsets.max() < min_refractor_size * refractor:
            break
        rv_last = RefractorVelocity.from_first_breaks(offsets, times, init=init, bounds=bounds, n_refractors=refractor,
                                                      loss=loss, huber_coef=huber_coef,
                                                      min_velocity_step=min_velocity_step,
                                                      min_refractor_size=min_refractor_size_vec)
        # TODO: remove debug
        if debug:
            rv_last.plot(title=f'{rv_last.fit_result.fun}\nfind_weathering={find_weathering}')
        n_points, _ = np.histogram(rv_last.offsets, bins=rv_last.piecewise_offsets)
        if not ((n_points > 1).all() and (rv is None or rv_last.fit_result.fun < rv.fit_result.fun)):
            break
        rv = rv_last
    return rv

def calc_mean_velocity(survey, min_velocity_step=300, min_refractor_size=300, loss="L1", huber_coef=20,
                       first_breaks_col=HDR_FIRST_BREAK, find_weathering=True, debug=False):
    """Calculate mean near-surface velocity model describing the survey.

    Parameters
    ----------
    survey : Survey
        Survey with preloaded offsets, times of first breaks, and coords.
    min_velocity_step : int, optional, defaults to 300
        Minimum difference between velocities of two adjacent refractors. Default value ensures that velocities are
        strictly increasing.
    min_refractor_size : int, optional, defaults to 300
        Minimum offset range covered by each refractor. Default value ensures that refractors do not degenerate
        into single points.
    loss : str, optional, defaults to "L1"
        Loss function to be minimized. Should be one of "MSE", "huber", "L1", "soft_L1", or "cauchy".
    huber_coef : float, optional, default to 20
        Coefficient for Huber loss function.
    first_breaks_col : str, optional, defaults to :const:`~const.HDR_FIRST_BREAK`
        Column name from `survey.headers` where times of first break are stored.
    find_weathering : bool, optional, defaults to True
        Try to find a weathering layer.

    Returns
    -------
    rv : RefractorVelocity
        Mean near-surface velocity model.

    Raises
    ------
    ValueError
        If survey does not contain any indices.
    """
    if survey.n_gathers < 1:  # need if the func calls separately from `RefractorVelocityField.from_survey`
        raise ValueError("Survey is empty.")
    offsets, times = reduce_mean_df(survey.headers[['offset', first_breaks_col]])
    rv = calc_optimal_velocity(offsets, times, min_refractor_size=min_refractor_size,
                               min_velocity_step=min_velocity_step, debug=debug)
    if find_weathering:  # try to find the weathering layer
        init = {'x1': 150, 'v1': rv.v1 / 2}
        bounds = {'x1': [1, 300], 'v1': [1, rv.v1]}
        min_refractors = max(rv.n_refractors, 2)
        rv_weathering = calc_optimal_velocity(offsets, times, init, bounds, min_velocity_step, min_refractor_size,
                                              loss, huber_coef, min_refractors, find_weathering=True, debug=debug)
        if rv_weathering is not None and rv_weathering.fit_result.fun < rv.fit_result.fun:
            rv = rv_weathering
    return rv
