"""Implements RefractorVelocity class for estimating the velocity model of an upper part of the section."""

import re
from functools import partial

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import SGDRegressor

from .utils import get_param_names, postprocess_params, load_rv, dump_rv, calc_df_to_dump
from ..muter import Muter
from ..decorators import batch_method, plotter
from ..utils import set_ticks, set_text_formatting, Coordinates
from ..utils.interpolation import interp1d


def _scale_standard(data):
    """Scale data to zero mean and unit variance."""
    if len(data) == 0:
        return data, 0, 0
    mean, std = np.mean(data), np.std(data)
    data_scaled = (data - mean) / (std + 1e-10)
    return data_scaled, mean, std


def fit_refractor_velocity(offsets, times, refractor_bounds):
    refractor_mask = (offsets > refractor_bounds[0]) & (offsets <= refractor_bounds[1])
    scaled_offsets, mean_offset, std_offset = _scale_standard(offsets[refractor_mask])
    scaled_times, mean_time, std_time = _scale_standard(times[refractor_mask])
    if np.isclose(min(std_offset, std_time), 0):
        return np.nan, np.nan

    lin_reg = SGDRegressor(loss='huber', penalty=None, shuffle=True, epsilon=.1, eta0=0.1, alpha=0.01,
                           tol=1e-6, max_iter=1000, learning_rate='optimal')
    lin_reg.fit(scaled_offsets.reshape(-1, 1), scaled_times, coef_init=1, intercept_init=0)
    velocity = std_offset / (lin_reg.coef_[0] * std_time)
    t0 = max(0, mean_time + lin_reg.intercept_[0] * std_time - mean_offset / velocity)
    return 1000 * velocity, t0


def refine_velocities(velocities, fixed_indices, min_velocity_increase):
    nan_velocities = np.isnan(velocities)
    if nan_velocities.all():
        return 1600 + min_velocity_increase * np.arange(len(velocities))
    if len(fixed_indices) == 0:
        fixed_indices = np.where(~nan_velocities)[0][:1]

    # Refine velocities between each two adjacent velocities obtained from init
    for start, stop in zip(fixed_indices[:-1], fixed_indices[1:]):
        for pos in range(start + 1, stop):
            velocities[pos] = np.nanmax([velocities[pos], velocities[pos - 1] + min_velocity_increase])
        for pos in range(stop - 1, start, -1):
            velocities[pos] = np.nanmin([velocities[pos], velocities[pos + 1] - min_velocity_increase])

    # Refine velocities of refractors outside those defined in init
    for pos in range(fixed_indices[-1] + 1, len(velocities)):
        velocities[pos] = np.nanmax([velocities[pos], velocities[pos - 1] + min_velocity_increase])
    for pos in range(fixed_indices[0] - 1, -1, -1):
        velocities[pos] = np.nanmin([velocities[pos], velocities[pos + 1] - min_velocity_increase])

    return velocities


# pylint: disable=too-many-instance-attributes
class RefractorVelocity:
    """The class stores and fits parameters of a velocity model of an upper part of the section.

    An instance can be created using one of the following `classmethod`s:
        * `from_first_breaks` - fits a near-surface velocity model by offsets and times of first breaks.
        * `from_params` - creates a `RefractorVelocity` instance from given parameters without model fitting.
        * `from_constant` - creates a single-layer `RefractorVelocity` with zero intercept time and given velocity of
                            the refractor.

    Parameters of the constructed velocity model can be obtained by accessing the following attributes:
        `t0`: two-way travel time from a shot to a receiver just above it on the surface for uphole surveys. Measured
              in milliseconds.
        `x{i}`: crossover offset: the offset where a wave refracted from the i-th layer arrives at the same time as
                a wave refracted from the next underlying layer. Measured in meters.
        `v{i}`: velocity of the i-th layer. Measured in meters per second.

    The same names are used as keys in `init` and `bounds` dicts passed to `RefractorVelocity.from_first_breaks`
    constructor. Some keys may be omitted in one dict if they are passed in another, e.g. one can pass only bounds for
    `v1` without an initial value, which will be inferred automatically. Both `init` and `bounds` dicts may not be
    passed at all if `n_refractors` is given.

    Examples
    --------
    Create a `RefractorVelocity` instance from known parameters and avoid the fitting procedure:
    >>> refractor_velocity = RefractorVelocity.from_params(params={'t0': 100, 'x1': 1500, 'v1': 2000, 'v2': 3000})

    `RefractorVelocity` can be estimated automatically by offsets of traces and times of first arrivals. First, let's
    load them for a randomly selected common source gather:
    >>> survey = Survey(survey_path, header_index="FieldRecord", header_cols=["offset", "TraceNumber"])
    >>> survey = survey.load_first_breaks(first_breaks_path)
    >>> gather = survey.sample_gather()
    >>> offsets = gather.offsets
    >>> fb_times = gather['FirstBreak'].ravel()

    Now an instance of `RefractorVelocity` can be created using `from_first_breaks` method:
    >>> refractor_velocity = RefractorVelocity.from_first_breaks(offsets, fb_times, n_refractors=2)

    The same can be done by calling `calculate_refractor_velocity` method of the gather:
    >>> refractor_velocity = gather.calculate_refractor_velocity(n_refractors=2)

    Fit a two-layer refractor velocity model using initial values of its parameters:
    >>> initial_params = {'t0': 100, 'x1': 1500, 'v1': 2000, 'v2': 3000}
    >>> refractor_velocity = RefractorVelocity.from_first_breaks(offsets, fb_times, init=initial_params)

    Fit a single-layer model with constrained bounds:
    >>> refractor_velocity = RefractorVelocity.from_first_breaks(offsets, fb_times,
                                                                 bounds={'t0': [0, 200], 'v1': [1000, 3000]})

    Some keys in `init` or `bounds` may be omitted if they are defined in another `dict` or `n_refractors` is given:
    >>> refractor_velocity = RefractorVelocity.from_first_breaks(offsets, fb_times, init={'x1': 200, 'v1': 1000},
                                                                 bounds={'t0': [0, 50]}, n_refractors=3)

    Attributes
    ----------
    offsets : 1d ndarray
        Offsets of traces. Measured in meters.
    fb_times : 1d ndarray
        First breaks times of traces. Measured in milliseconds.
    max_offset : float
        Maximum offset value.
    coords : Coordinates or None
        Spatial coordinates at which refractor velocity is estimated.
    init : dict
        The initial values used to fit the parameters of the velocity model. Includes the calculated values for
        parameters that were not passed.
    bounds : dict
        Lower and upper bounds used to fit the parameters of the velocity model. Includes the calculated values for
        parameters that were not passed.
    n_refractors : int
        The number of layers used to fit the parameters of the velocity model.
    piecewise_offsets : 1d ndarray
        Offsets of knots of the offset-traveltime curve. Measured in meters.
    piecewise_times : 1d ndarray
        Times of knots of the offset-traveltime curve. Measured in milliseconds.
    params : dict
        Parameters of the fitted velocity model.
    interpolator : callable
        An interpolator returning expected arrival times for given offsets.
    """
    def __init__(self, max_offset=None, coords=None, **params):
        self._validate_params(params, max_offset)
        self.n_refractors = len(params) // 2

        # Store params in the order defined by param_names
        self.params = {name: params[name] for name in self.param_names}
        knots = self._calc_knots_by_params(np.array(list(self.params.values())), max_offset)
        self.piecewise_offsets, self.piecewise_times = knots
        self.interpolator = interp1d(self.piecewise_offsets, self.piecewise_times)
        self.coords = coords

        # Fit-related attributes, set only when from_first_breaks is called
        self.is_fit = False
        self.fit_result = None
        self.init = None
        self.bounds = None
        self.offsets = None
        self.times = None

    @classmethod
    def from_first_breaks(cls, offsets, times, init=None, bounds=None, n_refractors=None, max_offset=None,
                          min_velocity_increase=0, min_crossover_increase=0, loss="L1", huber_coef=20, tol=1e-5,
                          coords=None, **kwargs):
        """Create a `RefractorVelocity` instance from offsets and times of first breaks. At least one of `init`,
        `bounds` or `n_refractors` must be passed.

        Parameters
        ----------
        offsets : 1d ndarray
            Offsets of the traces. Measured in meters.
        times : 1d ndarray
            First break times. Measured in milliseconds.
        init : dict, defaults to None
            Initial parameters of a velocity model.
        bounds : dict, defaults to None
            Lower and upper bounds of the velocity model parameters.
        n_refractors : int, defaults to None
            Number of layers of the velocity model.
        coords : Coordinates or None, optional
            Spatial coordinates of the created refractor velocity.
        loss : str, defaults to "L1"
            Loss function for `scipy.optimize.minimize`. Should be one of "MSE", "huber", "L1", "soft_L1", or "cauchy".
        huber_coef : float, default to 20
            Coefficient for Huber loss function.
        kwargs : misc, optional
            Additional keyword arguments to `scipy.optimize.minimize`.

        Raises
        ------
        ValueError
            If all `init`, `bounds`, and `n_refractors` are `None`.
            If any `init` values are negative.
            If any `bounds` values are negative.
            If left bound is greater than the right bound for any of model parameters.
            If initial value of a parameter is out of defined bounds.
            If `n_refractors` is less than 1.
            If passed `init` and/or `bounds` keys are insufficient or excessive.
        """
        offsets = np.array(offsets)
        times = np.array(times)

        if all(param is None for param in (init, bounds, n_refractors)):
            raise ValueError("At least one of `init`, `bounds` or `n_refractors` must be defined")
        init = {} if init is None else init
        bounds = {} if bounds is None else bounds

        init_by_bounds = {key: (val1 + val2) / 2 for key, (val1, val2) in bounds.items()}
        init = {**init_by_bounds, **init}

        # Check whether init dict contains only valid param names
        pattern = re.compile("(t0)|([xv][1-9]\d*)")  # t0 or x{i}/v{i} for i >= 1
        bad_names = [param_name for param_name in init.keys() if pattern.fullmatch(param_name) is None]
        if bad_names:
            raise ValueError(f"Wrong param names passed to init or bounds: {bad_names}")

        # Estimate max_offset if it is not given and check whether it is greater than all user-defined inits and bounds
        # for crossover offsets
        max_data_offset = offsets.max()
        max_crossover_offset_init = max((val for key, val in init.items() if key.startswith("x")), default=0)
        max_crossover_offset_bound = max((max(val) for key, val in bounds.items() if key.startswith("x")), default=0)
        if max_offset is None:
            max_offset = max_data_offset
        if max_offset < max(max_data_offset, max_crossover_offset_init, max_crossover_offset_bound):
            raise ValueError("max_offset must be greater than maximum data offset and all user-defined "
                             "inits and bounds for crossover offsets")

        if n_refractors is not None:
            # Automatically estimate all params that were not passed in init or bounds
            param_names = get_param_names(n_refractors)
            if init.keys() - set(param_names):
                raise ValueError("Parameters defined by init and bounds describe more refractors "
                                 "than defined by n_refractors")

            # Linearly interpolate unknown crossover offsets
            crossover_offsets = np.array([0] + [init.get(f"x{i}", np.nan) for i in range(1, n_refractors)] + [max_offset])
            undefined_mask = np.isnan(crossover_offsets)
            crossover_indices = np.arange(n_refractors + 1)
            crossover_offsets = np.interp(crossover_indices, crossover_indices[~undefined_mask], crossover_offsets[~undefined_mask])

            # Fit linear regressions to estimate unknown refractor velocities
            velocities = np.array([init.get(f"v{i}", np.nan) for i in range(1, n_refractors + 1)])
            undefined_mask = np.isnan(velocities)
            if undefined_mask[0] or ("t0" not in init):
                vel, t0 = fit_refractor_velocity(offsets, times, crossover_offsets[:2])
                if undefined_mask[0]:
                    velocities[0] = vel
                init_t0 = init.get("t0", np.nan_to_num(t0))
            for i in np.where(undefined_mask[1:])[0] + 1:
                velocities[i] = fit_refractor_velocity(offsets, times, crossover_offsets[i:i+2])[0]
            velocities = refine_velocities(velocities, np.where(~undefined_mask)[0], min_velocity_increase)
            init = dict(zip(param_names, [init_t0, *crossover_offsets[1:-1], *velocities]))

        cls._validate_params(init, max_offset, min_velocity_increase, min_crossover_increase)
        n_refractors = len(init) // 2
        param_names = get_param_names(n_refractors)

        default_params_bounds = np.array([[0, np.inf]] * 2 * n_refractors)
        default_params_bounds[1:n_refractors, 1] = max_offset  # clip crossover offsets with max offset
        bounds = {**dict(zip(param_names, default_params_bounds)), **bounds}
        cls._validate_params_bounds(init, bounds)

        # Store init and bounds in the order defined by param_names
        init = {name: init[name] for name in param_names}
        bounds = {name: bounds[name] for name in param_names}

        # Calculate arrays of initial params and their bounds to be passed to minimize
        init_array = cls._scale_params(np.array(list(init.values()), dtype=np.float32))
        bounds_array = cls._scale_params(np.array(list(bounds.values()), dtype=np.float32))

        # Define model constraints, appropriately scale minimum velocity and crossover offset increase
        crossover_offsets_ascend = {
            "type": "ineq",
            "fun": lambda x: (np.diff(x[1:n_refractors], prepend=0, append=max_offset / 1000) -
                              min_crossover_increase / 1000)
        }
        velocities_ascend = {
            "type": "ineq",
            "fun": lambda x: np.diff(x[n_refractors:]) - min_velocity_increase / 1000
        }
        constraints = [crossover_offsets_ascend, velocities_ascend]

        # Fit a piecewise-linear velocity model
        loss_fn = partial(cls.calculate_loss, loss=loss, huber_coef=huber_coef)
        fit_result = minimize(loss_fn, args=(offsets, times, max_offset), method="SLSQP", tol=tol, options=kwargs,
                              x0=init_array, bounds=bounds_array, constraints=constraints)
        param_values = postprocess_params(cls._unscale_params(fit_result.x.copy()))
        params = dict(zip(param_names, param_values))

        # Construct a refractor velocity instance
        self = cls(coords=coords, max_offset=max_offset, **params)
        self.is_fit = True
        self.fit_result = fit_result
        self.init = init
        self.bounds = bounds
        self.offsets = offsets
        self.times = times
        return self

    @classmethod
    def from_file(cls, path, encoding="UTF-8"):
        """Load parameters from a file and create a RefractorVelocity instance from the loaded parameters.

        File example:
        SourceX   SourceY        t0        x1        v1        v2 max_offset
        1111100   2222220     50.00   1000.00   1500.00   2000.00    2000.00

        Parameters
        ----------
        path : str,
            path to the file with parameters.

        Returns
        -------
        self : RefractorVelocity
            RefractorVelocity instance created from a file.
        """
        coords_list, params_list, max_offset_list = load_rv(path, encoding)
        if len(coords_list) > 1:
            raise ValueError("The loaded file contains more than one set of RefractorVelocity parameters.")
        # TODO: select one of the options when max_offset support will be determined
        return cls(coords=coords_list[0], **params_list[0])
        # return cls(max_offset=max_offset_list[0], coords=coords_list[0], **params_list[0])

    @classmethod
    def from_constant_velocity(cls, velocity, max_offset=None, coords=None):
        """Define a 1-layer near-surface velocity model with given velocity of the first layer and zero intercept time.

        Parameters
        ----------
        velocity : float
            Velocity of the first layer.
        coords : Coordinates or None, optional
            Spatial coordinates of the created object.

        Returns
        -------
        RefractorVelocity
            RefractorVelocity instance based on given velocity.

        Raises
        ------
        ValueError
            If passed `velocity` is negative.
        """
        return cls(t0=0, v1=velocity, max_offset=max_offset, coords=coords)

    @property
    def param_names(self):
        return get_param_names(self.n_refractors)

    @property
    def has_coords(self):
        """bool: Whether refractor velocity coordinates are not-None."""
        return self.coords is not None

    def __getattr__(self, key):
        """Get requested parameter of the velocity model."""
        return self.params[key]

    def __call__(self, offsets):
        """Return the expected times of first breaks for the given offsets."""
        return self.interpolator(offsets)

    # Methods to validate model parameters and their bounds for correctness

    @staticmethod
    def _validate_params_names(params):
        n_refractors = len(params) // 2
        if n_refractors < 1:
            raise ValueError("At least t0 and v1 parameters must be specified.")
        wrong_keys = set(get_param_names(n_refractors)) ^ params.keys()
        if wrong_keys:
            raise ValueError("The model is underdetermined. The following parameters should be passed: "
                             "t0, v1, ..., v{n}, x1, ..., x{n-1}")

    @classmethod
    def _validate_params(cls, params, max_offset=None, min_velocity_increase=0, min_crossover_increase=0):
        cls._validate_params_names(params)
        n_refractors = len(params) // 2
        param_names = get_param_names(n_refractors)
        param_values = np.array([params[name] for name in param_names])
        if max_offset is None:
            max_offset = np.inf

        negative_param = {key: val for key, val in params.items() if val < 0}
        if negative_param:
            raise ValueError(f"The following parameters contain negative values: {negative_param}")

        if (np.diff(param_values[1:n_refractors], prepend=0, append=max_offset) < min_crossover_increase).any():
            raise ValueError(f"Crossover offsets must ascend by no less than {min_crossover_increase}")

        if (np.diff(param_values[n_refractors:]) < min_velocity_increase).any():
            raise ValueError(f"Refractor velocities must ascend by no less than {min_velocity_increase}")

    @classmethod
    def _validate_params_bounds(cls, params, bounds):
        cls._validate_params_names(bounds)
        if params.keys() != bounds.keys():
            raise ValueError("params and bounds must contain the same keys")

        negative_bounds = {key: val for key, val in bounds.items() if min(val) < 0}
        if negative_bounds:
            raise ValueError(f"The following parameters contain negative bounds: {negative_bounds}")

        reversed_bounds = {key: [left, right] for key, [left, right] in bounds.items() if left > right}
        if reversed_bounds:
            raise ValueError(f"The following parameters contain reversed bounds: {reversed_bounds}")

        out_of_bounds = {name for name in params.keys()
                              if params[name] < bounds[name][0] or params[name] > bounds[name][1]}
        if out_of_bounds:
            raise ValueError(f"Values of the following parameters are out of their bounds: {out_of_bounds}")

    # Loss definition

    @staticmethod
    def _scale_params(params):
        params[0] /= 100
        params[1:] /= 1000
        return params

    @staticmethod
    def _unscale_params(params):
        params[0] *= 100
        params[1:] *= 1000
        return params

    @staticmethod
    def _calc_knots_by_params(params, max_offset=None):
        """Calculate the coordinates of the knots of a piecewise linear function based on the given `params` and
        `max_offset`."""
        n_refractors = len(params) // 2
        params_max_offset = params[n_refractors - 1] if n_refractors > 1 else 0
        if max_offset is None or max_offset < params_max_offset:
            max_offset = params_max_offset + 1000  # Artificial setting of max offset to properly define interpolator

        piecewise_offsets = np.concatenate([[0], params[1:n_refractors], [max_offset]])
        piecewise_times = np.empty(n_refractors + 1)
        piecewise_times[0] = params[0]
        params_zip = zip(piecewise_offsets[1:], piecewise_offsets[:-1], params[n_refractors:])
        for i, (cross, prev_cross, vel) in enumerate(params_zip):
            piecewise_times[i + 1] = piecewise_times[i]
            if not np.isclose(vel, 0):
                piecewise_times[i + 1] += 1000 * (cross - prev_cross) / vel  # m/s to km/s
        return piecewise_offsets, piecewise_times

    @classmethod
    def calculate_loss(cls, params, offsets, times, max_offset, loss='L1', huber_coef=20):
        """Calculate the result of the loss function based on the passed args.

        Method calls `calc_knots_by_params` to calculate piecewise linear attributes of a RefractorVelocity instance.
        After that, the method calculates the loss function between the true first breaks times stored in the
        `self.times` and predicted piecewise linear function. The loss function is calculated at the offsets points.

        Piecewise linear function is defined by the given `args`. `args` should be list-like and have the following
        structure:
            args[0] : intercept time in milliseconds.
            args[1:n_refractors] : cross offsets points in meters.
            args[n_refractors:] : velocities of each layer in kilometers/seconds.
            Total length of args should be n_refractors * 2.

        Notes:
            * 'init', 'bounds' and 'params' store velocity in m/s unlike args for `loss_piecewise_linear`.
            * The list-like `args` is due to the `scipy.optimize.minimize`.

        Parameters
        ----------
        args : tuple, list, or 1d ndarray
            Parameters of the piecewise linear function.
        loss : str, optional, defaults to "L1".
            The loss function type. Should be one of "MSE", "huber", "L1", "soft_L1", or "cauchy".
            All implemented loss functions have a mean reduction.
        huber_coef : float, default to 20
            Coefficient for Huber loss.

        Returns
        -------
        loss : float
            Loss function result between true first breaks times and a predicted piecewise linear function.

        Raises
        ------
        ValueError
            If given `loss` does not exist.
        """
        piecewise_offsets, piecewise_times = cls._calc_knots_by_params(cls._unscale_params(params), max_offset)
        abs_diff = np.abs(np.interp(offsets, piecewise_offsets, piecewise_times) - times)
        if loss == 'MSE':
            return (abs_diff ** 2).mean()
        if loss == 'huber':
            loss = np.empty_like(abs_diff)
            mask = abs_diff <= huber_coef
            loss[mask] = .5 * (abs_diff[mask] ** 2)
            loss[~mask] = huber_coef * abs_diff[~mask] - .5 * (huber_coef ** 2)
            return loss.mean()
        if loss == 'L1':
            return abs_diff.mean()
        if loss == 'soft_L1':
            return 2 * ((1 + abs_diff) ** .5 - 1).mean()
        if loss == 'cauchy':
            return np.log(abs_diff + 1).mean()
        raise ValueError("Unknown loss function")

    # General processing methods

    @batch_method(target="for", copy_src=False)
    def create_muter(self, delay=0, velocity_reduction=0):
        return Muter.from_refractor_velocity(self, delay=delay, velocity_reduction=velocity_reduction)

    def dump(self, path, encoding="UTF-8", min_col_size=11):
        """Dump the RefractorVelocity instance to a file. Coords should be preloaded.

        File example:
        SourceX   SourceY        t0        x1        v1        v2 max_offset
        1111100   2222220     50.00   1000.00   1500.00   2000.00    2000.00

        Parameters
        ----------
        path : str
            Path to a file.
        encoding : str, optional, defaults to "UTF-8"
            File encoding.
        min_col_size : int, defaults to 11
            Minimum size of each columns in file.

        Returns
        -------
        self : RefractorVelocity
            RefractorVelocity unchanged.

        Raises
        ------
        ValueError
            If coords attributes is None.
        """
        if self.coords is None:
            raise ValueError("`coords` attribute should be defined.")
        dump_rv([calc_df_to_dump(self)], path=path, encoding=encoding, min_col_size=min_col_size)
        return self

    @plotter(figsize=(10, 5), args_to_unpack="compare_to")
    def plot(self, *, ax=None, title=None, x_ticker=None, y_ticker=None, show_params=True, threshold_times=None,
             compare_to=None, text_kwargs=None, **kwargs):
        """Plot the RefractorVelocity data, fitted curve, cross offsets, and additional information.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional, defaults to None
            An axis of the figure to plot on. If not given, it will be created automatically.
        title : str, optional, defaults to None
            Plot title.
        x_ticker : dict, optional, defaults to None
            Parameters for ticks and ticklabels formatting for the x-axis; see :func:`~utils.set_ticks`
            for more details.
        y_ticker : dict, optional, defaults to None
            Parameters for ticks and ticklabels formatting for the y-axis; see :func:`~utils.set_ticks`
            for more details.
        show_params : bool, optional, defaults to True
            If `True` shows the velocity model parameters on the plot.
        threshold_times : float or None, optional. Defaults to None
            Neighborhood margins of the fitted curve to fill in the area inside. If None the area don't show.
        compare_to : RefractorVelocity, str or None, optional
            RefractorVelocity instance. Used to plot an additional RefractorVelocity on the same axis.
            May be `str` if plotted in a pipeline: in this case it defines a component with refractor velocities to
            compare to.
        text_kwargs : dict, optional
            Additional arguments to the :func:`~matplotlib.pyplot.text`. This function plot velocity model parameters
            on the plot.
        kwargs : dict, optional
            Additional keyword arguments to :func:`~utils.set_text_formatting`. Used to the modify the text and titles
            formatting.

        Returns
        -------
        self : RefractorVelocity
            RefractorVelocity without changes.
        """
        (title, x_ticker, y_ticker, text_kwargs), kwargs = set_text_formatting(title, x_ticker, y_ticker, text_kwargs,
                                                                               **kwargs)
        if kwargs:
            raise ValueError(f'kwargs contains unknown keys {kwargs.keys()}')
        set_ticks(ax, "x", tick_labels=None, label="offset, m", **x_ticker)
        set_ticks(ax, "y", tick_labels=None, label="time, ms", **y_ticker)

        ax.scatter(self.offsets, self.times, s=1, color='black', label='first breaks')
        self._plot_lines(ax, curve_label='offset-traveltime curve', curve_color='red',
                         crossoffset_label='crossover point', crossover_color='blue')

        if show_params:
            params = [self.params[name] for name in self.param_names]
            text_info = f"t0: {params[0]:.2f} ms"
            if self.n_refractors > 1:
                text_info += f"\ncrossover offsets: {', '.join(str(round(x)) for x in params[1:self.n_refractors])} m"
            text_info += f"\nvelocities: {', '.join(f'{v:.0f}' for v in params[self.n_refractors:])} m/s"
            text_kwargs = {'fontsize': 12, 'va': 'top', **text_kwargs}
            text_ident = text_kwargs.pop('x', .03), text_kwargs.pop('y', .94)
            ax.text(*text_ident, text_info, transform=ax.transAxes, **text_kwargs)

        if threshold_times is not None:
            ax.fill_between(self.piecewise_offsets, self.piecewise_times - threshold_times,
                            self.piecewise_times + threshold_times, color='red',
                            label=f'+/- {threshold_times}ms threshold area', alpha=.2)

        if compare_to is not None:
            # pylint: disable-next=protected-access
            compare_to._plot_lines(ax, curve_label='compared offset-traveltime curve', curve_color='#ff7900',
                                   crossoffset_label='compared crossover point', crossover_color='green')

        ax.set_xlim(0)
        ax.set_ylim(0)
        ax.legend(loc='lower right')
        ax.set_title(**{"label": None, **title})
        return self

    def _plot_lines(self, ax, curve_label, curve_color, crossoffset_label, crossover_color):
        """Plot offset-traveltime curve and a vertical line for each crossover offset."""
        ax.plot(self.piecewise_offsets, self.piecewise_times, '-', color=curve_color, label=curve_label)
        if self.n_refractors > 1:
            crossoffset_label += 's'
        for i in range(1, self.n_refractors):
            label = crossoffset_label if i == 1 else None
            ax.axvline(self.piecewise_offsets[i], ls='--', color=crossover_color, label=label)
