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


def load_refractor_velocities(path, encoding="UTF-8"):
    """Load near-surface velocity models from a file.

    The file should define near-surface velocity models at given field locations and have the following structure:
    - The first row contains names of the coordinates parameters ("name_x", "name_y", "x", "y") and names of parameters
      of near-surface velocity models ("t0", "x1"..."x{n-1}", "v1"..."v{n}"). Each velocity model must describe the
      same number of refractors.
    - Each next row contains the corresponding parameters of a single near-surface velocity model.

    File example:
     name_x     name_y          x          y        t0        x1        v1        v2
    SourceX    SourceY    1111100    2222220     50.25   1000.10   1500.25   2000.10
    ...
    SourceX    SourceY    1111100    2222220     50.50   1000.20   1500.50   2000.20

    Parameters
    ----------
    path : str
        Path to a file.
    encoding : str, optional, defaults to "UTF-8"
        File encoding.

    Returns
    -------
    rv_list : list of RefractorVelocity
        A list of loaded near-surface velocity models.
    """
    #pylint: disable-next=import-outside-toplevel
    from .refractor_velocity import RefractorVelocity  # import inside to avoid the circular import
    df = pd.read_csv(path, sep=r'\s+', encoding=encoding).convert_dtypes()
    params_names = df.columns[4:]
    return [RefractorVelocity(**dict(zip(params_names, row[4:])), coords=Coordinates(row[2:4], row[:2]))
            for row in df.itertuples(index=False)]


def dump_refractor_velocities(refractor_velocities, path, encoding="UTF-8"):
    """Dump parameters of passed near-surface velocity models to a file.

    Notes
    -----
    See more about the file format in :func:`~load_refractor_velocities`.

    Parameters
    ----------
    refractor_velocities : RefractorVelocity or iterable of RefractorVelocity
        Near-surface velocity models to be dumped to a file.
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


def reduce_offsets_and_times(survey, first_breaks_col=HDR_FIRST_BREAK, reduce_step=20):
    """Reduce the offsets and times of first breaks stored in the survey headers.

    Method splits the survey headers by bins with the same size define by `reduce_step` using offsets values and
    calculate the mean separately by bins. The first breaks times uses same bins and also calculate the mean
    separately by bins.
    """
    headers = survey.headers[['offset', first_breaks_col]]
    headers['bins'] = (headers['offset'].to_numpy() / reduce_step).astype(np.uint16)  # faster than integer division
    reduced_headers = headers.groupby(by='bins', sort=False).mean()
    return reduced_headers['offset'].to_numpy(), reduced_headers[first_breaks_col].to_numpy()


# pylint: disable-next=too-many-arguments
def calc_optimal_velocity(offsets, times, init=None, bounds=None, min_velocity_step=400, min_refractor_size=400,
                          loss="L1", huber_coef=20, min_refractors=1, max_refractors=10, find_weathering=False):
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
    min_velocity_step : int, optional, defaults to 400
        Minimum difference between velocities of two adjacent refractors.
    min_refractor_size : int, optional, defaults to 400
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
        If True the minimum refractor size contraint for the expected weathering layer is removed.

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
        max_offset = max(offsets.max(), min_refractor_size * refractor)
        rv_last = RefractorVelocity.from_first_breaks(offsets, times, init, bounds, refractor, max_offset,
                                                      min_velocity_step, min_refractor_size_vec, loss, huber_coef)
        n_points, _ = np.histogram(rv_last.offsets, bins=rv_last.piecewise_offsets)
        if not ((n_points > 1).all() and (rv is None or rv_last.fit_result.fun < rv.fit_result.fun)):
            break
        rv = rv_last
    return rv


def calc_mean_velocity(survey, min_velocity_step=400, min_refractor_size=400, loss="L1", huber_coef=20,
                       first_breaks_col=HDR_FIRST_BREAK, find_weathering=False, reduce_step=20):
    """Calculate mean near-surface velocity model describing the survey.

    Parameters
    ----------
    survey : Survey
        Survey with preloaded offsets, times of first breaks.
    min_velocity_step : int, optional, defaults to 400
        Minimum difference between velocities of two adjacent refractors. Default value ensures that velocities are
        strictly increasing.
    min_refractor_size : int, optional, defaults to 400
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
    reduce_step : float, defaults to 20
        Size of data chunks when splitting data by offset to reduce the data.

    Returns
    -------
    rv : RefractorVelocity
        Mean near-surface velocity model.

    Raises
    ------
    ValueError
        If the reduced survey data contains less than two points.
    """
    offsets, times = reduce_offsets_and_times(survey, first_breaks_col, reduce_step)
    if offsets.shape[0] < 2:
        raise ValueError("Offsets contains less than two points after reducing. Decrease the value of `reduce_step`.")
    rv = calc_optimal_velocity(offsets, times, min_velocity_step=min_velocity_step,
                               min_refractor_size=min_refractor_size, loss=loss, huber_coef=huber_coef)
    if find_weathering:
        init = {'x1': 150, 'v1': rv.v1 / 2}
        bounds = {'x1': [1, 300], 'v1': [1, rv.v1]}
        min_refractors = max(rv.n_refractors, 2)
        rv_weathering = calc_optimal_velocity(offsets, times, init, bounds, min_velocity_step, min_refractor_size,
                                              loss, huber_coef, min_refractors, find_weathering=True)
        if rv_weathering is not None and rv_weathering.fit_result.fun < rv.fit_result.fun:
            rv = rv_weathering
    return rv
