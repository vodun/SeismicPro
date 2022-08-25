"""Implements RefractorVelocity class for estimating the velocity model of an upper part of the section."""

import re
from functools import partial

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import SGDRegressor

from .utils import get_param_names, postprocess_params
from ..muter import Muter
from ..decorators import batch_method, plotter
from ..utils import set_ticks, set_text_formatting
from ..utils.interpolation import interp1d


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
        self.max_offset = max_offset
        self.coords = coords

        # Fit-related attributes, set only when from_first_breaks is called
        self.is_fit = False
        self.fit_result = None
        self.init = None
        self.bounds = None
        self.offsets = None
        self.times = None

    @classmethod
    def from_first_breaks(cls, offsets, times, init=None, bounds=None, n_refractors=None, max_offset=None, loss="L1",
                          huber_coef=20, tol=1e-5, min_velocity_step=0, min_crossover_step=0, coords=None, **kwargs):
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

        # Sort offset-time pairs to be monotonically non-decreasing in offset
        ind = np.argsort(offsets, kind="mergesort")
        offsets = offsets[ind]
        times = times[ind]

        if all(param is None for param in (init, bounds, n_refractors)):
            raise ValueError("At least one of `init`, `bounds` or `n_refractors` must be defined")
        init = {} if init is None else init
        bounds = {} if bounds is None else bounds

        # Merge initial values of parameters with those defined by bounds
        init_by_bounds = {key: (val1 + val2) / 2 for key, (val1, val2) in bounds.items()}
        init = {**init_by_bounds, **init}

        # Check whether init dict contains only valid names of parameters
        pattern = re.compile(r"(t0)|([xv][1-9]\d*)")  # t0 or x{i}/v{i} for i >= 1
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

        # Automatically estimate all params that were not passed in init or bounds by n_refractors
        if n_refractors is not None:
            init = cls.complete_init_by_refractors(init, n_refractors, offsets, times, max_offset,
                                                   min_velocity_step, min_crossover_step)

        # Validate initial values of model parameters and calculate the number of refractors
        cls._validate_params(init, max_offset, min_velocity_step, min_crossover_step)
        n_refractors = len(init) // 2
        param_names = get_param_names(n_refractors)
        min_velocity_step = np.broadcast_to(min_velocity_step, n_refractors-1)
        min_crossover_step = np.broadcast_to(min_crossover_step, n_refractors)

        # Estimate maximum possible velocity: it should not be highly accurate, but should cover all initial velocities
        # and their bounds. Used only to early-stop a diverging optimization on poor data when optimal velocity
        # approaches infinity.
        last_refractor_velocity = init[f"v{n_refractors}"]
        velocity_bounds = [bounds.get(f"v{i}", [0, 0]) for i in range(n_refractors)]
        max_velocity_bounds_range = max(right - left for left, right in velocity_bounds)
        max_velocity = last_refractor_velocity + max(max_velocity_bounds_range, last_refractor_velocity)

        # Set default bounds for parameters that don't have them specified, validate the result for correctness
        default_t0_bounds = [[0, max(init["t0"], times.max())]]
        default_crossover_bounds = [[min_crossover_step[0], max_offset - min_crossover_step[-1]]
                                    for _ in range(n_refractors - 1)]
        default_velocity_bounds = [[0, max_velocity] for _ in range(n_refractors)]
        default_params_bounds = default_t0_bounds + default_crossover_bounds + default_velocity_bounds
        bounds = {**dict(zip(param_names, default_params_bounds)), **bounds}
        cls._validate_params_bounds(init, bounds)

        # Store init and bounds in the order defined by param_names
        init = {name: init[name] for name in param_names}
        bounds = {name: bounds[name] for name in param_names}

        # Calculate arrays of initial params and their bounds to be passed to minimize
        init_array = cls._scale_params(np.array(list(init.values()), dtype=np.float32))
        bounds_array = cls._scale_params(np.array(list(bounds.values()), dtype=np.float32))

        # Define model constraints
        constraints = []
        if n_refractors > 1:
            velocities_ascend = {
                "type": "ineq",
                "fun": lambda x: np.diff(cls._unscale_params(x)[n_refractors:]) - min_velocity_step
            }
            constraints.append(velocities_ascend)
        if n_refractors > 2:
            crossover_offsets_ascend = {
                "type": "ineq",
                "fun": lambda x: np.diff(cls._unscale_params(x)[1:n_refractors]) - min_crossover_step[1:-1]
            }
            constraints.append(crossover_offsets_ascend)

        # Fit a piecewise-linear velocity model
        loss_fn = partial(cls.calculate_loss, offsets=offsets, times=times, max_offset=max_offset,
                          loss=loss, huber_coef=huber_coef)
        fit_result = minimize(loss_fn, x0=init_array, bounds=bounds_array, constraints=constraints,
                              method="SLSQP", tol=tol, options=kwargs)
        param_values = postprocess_params(cls._unscale_params(fit_result.x))
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

    def __repr__(self):
        params_str = ", ".join([f"{param}={val:.0f}" for param, val in self.params.items()])
        max_offset_str = None if self.max_offset is None else f"{self.max_offset:.0f}"
        return f"RefractorVelocity({params_str}, max_offset={max_offset_str}, coords={repr(self.coords)})"

    def __getattr__(self, key):
        """Get requested parameter of the velocity model."""
        return self.params[key]

    def __call__(self, offsets):
        """Return the expected times of first breaks for the given offsets."""
        return self.interpolator(offsets)

    # Methods to validate model parameters and their bounds for correctness

    @staticmethod
    def _validate_params_names(params):
        err_msg = ("The model is underdetermined. Pass t0 and v1 to define a one-layer model. "
                   "Pass t0, x1, ..., x{N-1}, v1, ..., v{N} to define an N-layer model for N >= 2.")
        n_refractors = len(params) // 2
        if n_refractors < 1:
            raise ValueError(err_msg)
        wrong_keys = set(get_param_names(n_refractors)) ^ params.keys()
        if wrong_keys:
            raise ValueError(err_msg)

    @classmethod
    def _validate_params(cls, params, max_offset=None, min_velocity_step=0, min_crossover_step=0):
        cls._validate_params_names(params)
        n_refractors = len(params) // 2
        param_values = np.array([params[name] for name in get_param_names(n_refractors)])
        if max_offset is None:
            max_offset = np.inf

        negative_params = {key: val for key, val in params.items() if val < 0}
        if negative_params:
            raise ValueError(f"The following parameters contain negative values: {negative_params}")

        if np.any(np.diff(param_values[1:n_refractors], prepend=0, append=max_offset) < min_crossover_step):
            raise ValueError("Distance between two adjacent crossover offsets "
                             f"must be no less than {min_crossover_step}")

        if np.any(np.diff(param_values[n_refractors:]) < min_velocity_step):
            raise ValueError(f"Refractor velocities must increase by no less than {min_velocity_step}")

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

        out_of_bounds = {key for key, [left, right] in bounds.items() if params[key] < left or params[key] > right}
        if out_of_bounds:
            raise ValueError(f"Values of the following parameters are out of their bounds: {out_of_bounds}")

    # Methods to roughly estimate refractor velocities

    @staticmethod
    def estimate_refractor_velocity(offsets, times, refractor_bounds):
        """Perform rough estimation of a refractor velocity and intercept time by fitting a linear regression to an
        offset-time point cloud within given offsets bounds."""
        # Avoid fitting a regression if an empty refractor is processed
        refractor_mask = (offsets > refractor_bounds[0]) & (offsets <= refractor_bounds[1])
        n_refractor_points = refractor_mask.sum()
        if n_refractor_points == 0:
            return np.nan, np.nan, n_refractor_points

        # Avoid fitting a regression if all points have constant offsets or times of first breaks
        refractor_offsets = offsets[refractor_mask]
        refractor_times = times[refractor_mask]
        mean_offset, std_offset = np.mean(refractor_offsets), np.std(refractor_offsets)
        mean_time, std_time = np.mean(refractor_times), np.std(refractor_times)
        if np.isclose([std_offset, std_time], 0).any():
            return np.nan, np.nan, n_refractor_points

        # Fit the model to obtain velocity in km/s and intercept time in ms
        scaled_offsets = (refractor_offsets - mean_offset) / std_offset
        scaled_times = (refractor_times - mean_time) / std_time
        reg = SGDRegressor(loss="huber", epsilon=0.1, penalty=None, learning_rate="optimal", alpha=0.01,
                           max_iter=1000, tol=1e-5, shuffle=True)
        reg.fit(scaled_offsets.reshape(-1, 1), scaled_times, coef_init=1, intercept_init=0)
        slope = reg.coef_[0] * std_time / std_offset
        t0 = mean_time + reg.intercept_[0] * std_time - slope * mean_offset

        velocity = 1000 / max(1/5, slope)  # Convert slope to velocity in m/s, clip it to be in a [0, 5000] interval
        t0 = min(max(0, t0), times.max())  # Clip intercept time to lie within a [0, times.max()] interval
        return velocity, t0, n_refractor_points

    @staticmethod
    def enforce_step_constraints(values, fixed_indices, min_step=0):
        fixed_indices = np.sort(np.atleast_1d(fixed_indices))
        min_step = np.broadcast_to(min_step, len(values) - 1)

        # Refine values between each two adjacent fixed values
        for start, stop in zip(fixed_indices[:-1], fixed_indices[1:]):
            for pos in range(start + 1, stop):
                values[pos] = np.nanmax([values[pos], values[pos - 1] + min_step[pos - 1]])
            for pos in range(stop - 1, start, -1):
                values[pos] = np.nanmin([values[pos], values[pos + 1] - min_step[pos]])

        # Refine values with indices outside the fixed_indices range
        for pos in range(fixed_indices[-1] + 1, len(values)):
            values[pos] = np.nanmax([values[pos], values[pos - 1] + min_step[pos - 1]])
        for pos in range(fixed_indices[0] - 1, -1, -1):
            values[pos] = np.nanmin([values[pos], values[pos + 1] - min_step[pos]])

        return values

    @classmethod
    def complete_init_by_refractors(cls, init, n_refractors, offsets, times, max_offset,
                                    min_velocity_step, min_crossover_step):
        param_names = get_param_names(n_refractors)
        if init.keys() - set(param_names):
            raise ValueError("Parameters defined by init and bounds describe more refractors "
                             "than defined by n_refractors")

        # Linearly interpolate unknown crossover offsets but enforce min_crossover_step constraint
        cross_offsets = np.array([0] + [init.get(f"x{i}", np.nan) for i in range(1, n_refractors)] + [max_offset])
        defined_indices = np.where(~np.isnan(cross_offsets))[0]
        cross_indices = np.arange(n_refractors + 1)
        cross_offsets = np.interp(cross_indices, cross_indices[defined_indices], cross_offsets[defined_indices])
        cross_offsets = cls.enforce_step_constraints(cross_offsets, defined_indices, min_crossover_step)

        # Fit linear regressions to estimate unknown refractor velocities
        velocities = np.array([init.get(f"v{i}", np.nan) for i in range(1, n_refractors + 1)])
        undefined_mask = np.isnan(velocities)
        estimates = [cls.estimate_refractor_velocity(offsets, times, cross_offsets[i:i+2])
                     for i in np.where(undefined_mask)[0]]
        velocities[undefined_mask] = [vel for (vel, _, _) in estimates]

        min_velocity_step = np.broadcast_to(min_velocity_step, n_refractors-1)
        if np.isnan(velocities).all():
            # Use a dummy velocity range as an initial guess if no velocities were passed in init/bounds dicts and
            # non of them were successfully fit using estimate_refractor_velocity
            velocities = np.cumsum(np.r_[1600, min_velocity_step])
        else:
            fixed_indices = np.where(~undefined_mask)[0]
            if undefined_mask.all():
                # If no velocities were passed in init, start the refinement from the refractor with maximum number of
                # points among those with properly estimated velocity
                fixed_index = max(enumerate(estimates), key=lambda x: x[1][-1])[0]
                velocities[fixed_index] = max(velocities[fixed_index], min_velocity_step[:fixed_index].sum())
                fixed_indices = [fixed_index]
            velocities = cls.enforce_step_constraints(velocities, fixed_indices, min_velocity_step)

        # Estimate t0 if not given in init
        t0 = init.get("t0")
        if t0 is None:
            if undefined_mask[0]:  # regression is already fit
                t0 = estimates[0][1]
            else:
                _, t0 = cls.estimate_refractor_velocity(offsets, times, cross_offsets[:2])
            t0 = np.nan_to_num(t0)  # can be nan if the regression hasn't fit successfully

        return dict(zip(param_names, [t0, *cross_offsets[1:-1], *velocities]))

    # Methods to fit a piecewise-linear regression

    @staticmethod
    def _scale_params(unscaled_params):
        scaled = np.empty_like(unscaled_params)
        scaled[0] = unscaled_params[0] / 100
        scaled[1:] = unscaled_params[1:] / 1000
        return scaled

    @staticmethod
    def _unscale_params(scaled_params):
        unscaled_params = np.empty_like(scaled_params)
        unscaled_params[0] = scaled_params[0] * 100
        unscaled_params[1:] = scaled_params[1:] * 1000
        return unscaled_params

    @staticmethod
    def _calc_knots_by_params(unscaled_params, max_offset=None):
        """Calculate the coordinates of the knots of a piecewise linear function based on the given `params` and
        `max_offset`."""
        n_refractors = len(unscaled_params) // 2
        params_max_offset = unscaled_params[n_refractors - 1] if n_refractors > 1 else 0
        if max_offset is None or max_offset < params_max_offset:
            # Artificially set max_offset in order to properly define an interpolator
            max_offset = params_max_offset + 1000

        piecewise_offsets = np.concatenate([[0], unscaled_params[1:n_refractors], [max_offset]])
        piecewise_times = np.empty(n_refractors + 1)
        piecewise_times[0] = unscaled_params[0]
        params_zip = zip(piecewise_offsets[1:], piecewise_offsets[:-1], unscaled_params[n_refractors:])
        for i, (cross, prev_cross, vel) in enumerate(params_zip):
            piecewise_times[i + 1] = piecewise_times[i] + 1000 * (cross - prev_cross) / max(0.01, vel)  # m/s to km/s
        return piecewise_offsets, piecewise_times

    @classmethod
    def calculate_loss(cls, scaled_params, offsets, times, max_offset, loss='L1', huber_coef=20):
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
        piecewise_offsets, piecewise_times = cls._calc_knots_by_params(cls._unscale_params(scaled_params), max_offset)
        abs_diff = np.abs(np.interp(offsets, piecewise_offsets, piecewise_times) - times)

        if loss == 'MSE':
            loss_val = abs_diff ** 2
        elif loss == 'huber':
            loss_val = np.empty_like(abs_diff)
            mask = abs_diff <= huber_coef
            loss_val[mask] = 0.5 * (abs_diff[mask] ** 2)
            loss_val[~mask] = huber_coef * abs_diff[~mask] - 0.5 * (huber_coef ** 2)
        elif loss == 'L1':
            loss_val = abs_diff
        elif loss == 'soft_L1':
            loss_val = 2 * ((1 + abs_diff) ** 0.5 - 1)
        elif loss == 'cauchy':
            loss_val = np.log(abs_diff + 1).mean()
        else:
            raise ValueError("Unknown loss function")
        return loss_val.mean()

    # General processing methods

    @batch_method(target="for", copy_src=False)
    def create_muter(self, delay=0, velocity_reduction=0):
        return Muter.from_refractor_velocity(self, delay=delay, velocity_reduction=velocity_reduction)

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
