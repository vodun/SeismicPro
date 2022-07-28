"""Implements RefractorVelocity class for estimating the velocity model of an upper part of the section."""

from functools import partial

import numpy as np
from sklearn.linear_model import SGDRegressor
from scipy import optimize
# from utils.coordinates import Coordinates

from ..decorators import plotter
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
    def __init__(self):
        self.offsets = None
        self.fb_times = None
        self.max_offset = None
        self.coords = None
        self.init = None
        self.bounds = None
        self.n_refractors = None
        self.piecewise_offsets = None
        self.piecewise_times = None
        self.params = None
        self.interpolator = None

        self._valid_keys = None
        self._empty_layers = None
        self._model_params = None

    @classmethod
    def from_first_breaks(cls, offsets, fb_times, init=None, bounds=None, n_refractors=None, coords=None, **kwargs):
        """Create a `RefractorVelocity` instance from offsets and times of first breaks. At least one of `init`,
        `bounds` or `n_refractors` must be passed.

        Parameters
        ----------
        offsets : 1d ndarray
            Offsets of the traces. Measured in meters.
        fb_times : 1d ndarray
            First break times. Measured in milliseconds.
        init : dict, defaults to None
            Initial parameters of a velocity model.
        bounds : dict, defaults to None
            Lower and upper bounds of the velocity model parameters.
        n_refractors : int, defaults to None
            Number of layers of the velocity model.
        coords : Coordinates or None, optional
            Spatial coordinates of the created refractor velocity.
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
        self = cls()
        if all((param is None for param in (init, bounds, n_refractors))):
            raise ValueError("One of `init`, `bounds` or `n_refractors` should be defined.")
        init = {} if init is None else init
        bounds = {} if bounds is None else bounds
        self._validate_values(init, bounds)

        self.offsets = offsets
        self.fb_times = fb_times
        self.max_offset = offsets.max()
        self.coords = coords

        self.init = {**self._calc_init_by_layers(n_refractors), **self._calc_init_by_bounds(bounds), **init}
        self.bounds = {**self._calc_bounds_by_init(), **bounds}
        self._validate_keys(self.bounds)
        self.n_refractors = len(self.bounds) // 2
        self._valid_keys = self._get_valid_keys()

        # ordering `init` and `bounds` dicts to put all values in the required order.
        self.init = {key: self.init[key] for key in self._valid_keys}
        self.bounds = {key: self.bounds[key] for key in self._valid_keys}

        # piecewise func parameters
        # move max_offset to right if init crossoffset out of the data
        if self.n_refractors > 1:
            self.max_offset = max(self.init[f'x{self.n_refractors - 1}'], self.max_offset)
        self.piecewise_offsets, self.piecewise_times = \
            self._create_piecewise_coords(self.n_refractors, self.max_offset)
        self.piecewise_offsets, self.piecewise_times = \
            self._update_piecewise_coords(self.piecewise_offsets, self.piecewise_times,
                                          self._ms_to_kms(self.init), self.n_refractors)

        self._empty_layers = np.histogram(self.offsets, self.piecewise_offsets)[0] ==  0
        constraints_list = self._get_constraints()

        # fitting piecewise linear regression
        partial_loss_func = partial(self.loss_piecewise_linear, loss=kwargs.pop('loss', 'L1'),
                                    huber_coef=kwargs.pop('huber_coef', 20))
        minimizer_kwargs = {'method': 'SLSQP', 'constraints': constraints_list, **kwargs}
        self._model_params = optimize.minimize(partial_loss_func, x0=self._ms_to_kms(self.init),
                                               bounds=self._ms_to_kms(self.bounds), **minimizer_kwargs)
        self.params = dict(zip(self._valid_keys, self._postprocess_params(self._model_params.x)))
        self.interpolator = interp1d(self.piecewise_offsets, self.piecewise_times)
        return self

    @classmethod
    def from_params(cls, params, coords=None):
        """Create a `RefractorVelocity` instance from its parameters without model fitting.

        Parameters
        ----------
        params : dict,
            Parameters of the velocity model.
        coords : Coordinates or None, optional
            Spatial coordinates of the created refractor velocity.

        Returns
        -------
        RefractorVelocity
            RefractorVelocity instance based on passed `params`.

        Raises
        ------
        ValueError
            If passed `params` keys are insufficient or excessive.
        """
        self = cls()

        self._validate_keys(params)
        self.n_refractors = len(params) // 2
        self._valid_keys = self._get_valid_keys(self.n_refractors)
        self.params = {key: params[key] for key in self._valid_keys}
        self.coords = coords

        self.piecewise_offsets, self.piecewise_times = \
            self._create_piecewise_coords(self.n_refractors, self.params.get(f'x{self.n_refractors - 1}', 0) + 1000)
        self.piecewise_offsets, self.piecewise_times = \
            self._update_piecewise_coords(self.piecewise_offsets, self.piecewise_times, self._ms_to_kms(self.params),
                                          self.n_refractors)
        self.interpolator = interp1d(self.piecewise_offsets, self.piecewise_times)
        return self

    @classmethod
    def from_constant_velocity(cls, velocity, coords=None):
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
        self = cls()

        if velocity < 0:
            raise ValueError("Velocity should not be negative.")
        return self.from_params({"t0": 0, "v1": velocity}, coords=coords)

    @classmethod
    def from_file(cls, path):
        """Create a `RefractorVelocity` instance from file.

        Parameters
        ----------
        path : str
            Path to the file with a velocity model parameters.

        Returns
        -------
        RefractorVelocity
            RefractorVelocity instance based on parameters.
        """
        self = cls()
        return self.load(path)

    def __call__(self, offsets):
        """Return the expected times of first breaks for the given offsets."""
        return self.interpolator(offsets)

    def __getattr__(self, key):
        """Get requested parameter of the velocity model."""
        return self.params[key]

    def has_coords(self):
        """bool: Whether refractor velocity coordinates are not-None."""
        return self.coords is not None

    def _create_piecewise_coords(self, n_refractors, max_offset=np.nan):
        """Create two array corresponding to the piecewise linear function coords."""
        piecewise_offsets = np.zeros(n_refractors + 1)
        piecewise_times = np.zeros(n_refractors + 1)
        piecewise_offsets[-1] = max_offset
        return piecewise_offsets, piecewise_times

    def _update_piecewise_coords(self, piecewise_offsets, piecewise_times, params, n_refractors):
        """Update the given `offsets` and `times` arrays based on the `params` and `n_refractors`."""
        piecewise_times[0] = params[0]
        piecewise_offsets[1:n_refractors] = params[1:n_refractors]
        for i in range(n_refractors):
            piecewise_times[i + 1] = ((piecewise_offsets[i + 1] - piecewise_offsets[i]) / params[n_refractors + i]) + \
                                     piecewise_times[i]
        return piecewise_offsets, piecewise_times

    def loss_piecewise_linear(self, args, loss='L1', huber_coef=20):
        """Update the piecewise linear attributes and returns the loss function result.

        Method calls `_update_piecewise_coords` to update piecewise linear attributes of a RefractorVelocity instance.
        After that, the method calculates the loss function between the true first breaks times stored in the
        `self.fb_times` and predicted piecewise linear function. The loss function is calculated at the offsets points.

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
            The loss function type. Should be one of "MSE", "L1", "huber", "soft_L1", or "cauchy".
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
        self.piecewise_offsets, self.piecewise_times = \
            self._update_piecewise_coords(self.piecewise_offsets, self.piecewise_times, args, self.n_refractors)
        diff_abs = np.abs(np.interp(self.offsets, self.piecewise_offsets, self.piecewise_times) - self.fb_times)
        if loss == 'MSE':
            return (diff_abs ** 2).mean()
        if loss == 'L1':
            return diff_abs.mean()
        if loss == 'huber':
            loss = np.empty_like(diff_abs)
            mask = diff_abs <= huber_coef
            loss[mask] = .5 * (diff_abs[mask] ** 2)
            loss[~mask] = huber_coef * diff_abs[~mask] - .5 * (huber_coef ** 2)
            return loss.mean()
        if loss == 'soft_L1':
            return 2 * ((1 + diff_abs) ** .5 - 1).mean()
        if loss == 'cauchy':
            return np.log(diff_abs + 1).mean()
        raise ValueError('Unknown loss type for `loss_piecewise_linear`.')

    def _get_valid_keys(self, n_refractors=None):
        """Returns a list with the valid keys based on `n_refractors` or `self.n_refractors`."""
        n_refractors = self.n_refractors if n_refractors is None else n_refractors
        return ['t0'] + [f'x{i}' for i in range(1, n_refractors)] + [f'v{i + 1}' for i in range(n_refractors)]

    def _get_constraints(self):
        """Define the constraints and return a list them."""
        constraint_offset = {  # cross offsets ascend.
            "type": "ineq",
            "fun": lambda x: np.diff(np.concatenate((x[1:self.n_refractors], [self.max_offset])))}
        constraint_velocity = {  # velocities ascend.
            "type": "ineq",
            "fun": lambda x: np.diff(x[self.n_refractors:])}
        constraint_freeze_velocity = {  # freeze the velocity if no data for layer is found.
            "type": "eq",
            "fun": lambda x: self._ms_to_kms(self.init)[self.n_refractors:][self._empty_layers]
                             - x[self.n_refractors:][self._empty_layers]}
        constraint_freeze_t0 = {  # freeze the intercept time if no data for layer is found.
            "type": "eq",
            "fun": lambda x: x[:1][self._empty_layers[:1]] - np.array([self.init['t0']])[self._empty_layers[:1]]}
        return [constraint_offset, constraint_velocity, constraint_freeze_velocity, constraint_freeze_t0]

    def _fit_regressor(self, x, y, start_slope, start_time):
        """Method fits the linear regression by given data and initial values.

        Parameters
        ----------
        x : 1d ndarray of shape (n_samples, 1)
            Training data.
        y : 1d ndarray of shape (n_samples,)
            Target values.
        start_slope : float
            Starting coefficient to fit a linear regression.
        start_time : float
            Starting intercept to fit a linear regression.

        Returns
        -------
        params : tuple
            Linear regression `coef` and `intercept`.
        """
        lin_reg = SGDRegressor(loss='huber', penalty=None, shuffle=True, epsilon=.1, eta0=0.1, alpha=0.01, tol=1e-6,
                               max_iter=1000, learning_rate='optimal')
        lin_reg.fit(x, y, coef_init=start_slope, intercept_init=start_time)
        return lin_reg.coef_[0], lin_reg.intercept_

    def _standart_scaler(self, data):
        """Standart scaler to zero mean."""
        cur_mean, cur_std = data.mean(), data.std()
        data_scaled = (data - cur_mean) / cur_std
        return data_scaled, cur_mean, cur_std

    def _calc_init_by_layers(self, n_refractors):
        """Calculates `init` dict by a given an estimated quantity of layers.

        Method splits the first breaks times into `n_refractors` equal part by cross offsets and fits a separate linear
        regression on each part. These linear functions are compiled together as a piecewise linear function.
        Parameters of piecewise function are calculated to the velocity model parameters and returned as `init` dict.

        Parameters
        ----------
        n_refractors : int
            Number of layers.

        Returns
        -------
        init : dict
            Estimated initial to fit the piecewise linear function.
        """
        if n_refractors is None or n_refractors < 1:
            return {}

        cross_offsets = np.linspace(0, self.max_offset, num=n_refractors + 1)
        current_slope = np.empty(n_refractors)
        current_time = np.empty(n_refractors)

        for i in range(n_refractors):
            mask = (self.offsets > cross_offsets[i]) & (self.offsets <= cross_offsets[i + 1])
            if mask.sum() > 1:  # at least two point to fit
                # data normalization occurs independently for each layer
                scaled_offsets, mean_offset, std_offset = self._standart_scaler(self.offsets[mask])
                scaled_times, mean_time, std_time = self._standart_scaler(self.fb_times[mask])
                fitted_slope, fitted_time = self._fit_regressor(scaled_offsets.reshape(-1, 1), scaled_times, 1, 0)
                current_slope[i] = fitted_slope * std_time / std_offset
                current_time[i] = mean_time + fitted_time * std_time - current_slope[i] * mean_offset
            else:
                # raise base velocity for the next layer (v = 1 / slope)
                current_slope[i] = current_slope[i] * (n_refractors / (n_refractors + 1)) if i else 4 / 5
                current_time[i] = 0  # used for first layer only (`t0`)
            # move maximal velocity to 6 km/s
            current_slope[i] = max(.167, current_slope[i])
            current_time[i] =  max(0, current_time[i])
        velocities = 1 / current_slope
        init = [current_time[0], *cross_offsets[1:-1], *(velocities * 1000)]
        init = dict(zip(self._get_valid_keys(n_refractors), init))
        return init

    def _calc_init_by_bounds(self, bounds):
        """Return dict with a calculated init from a bounds dict."""
        return {key: val1 + (val2 - val1) / 3 for key, (val1, val2) in bounds.items()}

    def _calc_bounds_by_init(self):
        """Return dict with calculated bounds from a init dict."""
        bounds = {key: [val / 2, val * 2] for key, val in self.init.items()}
        if 't0' in self.init:
            bounds['t0'] = [min(0, bounds['t0'][0]), max(200, bounds['t0'][1])]
        return bounds

    def _validate_values(self, init, bounds):
        """Check the values of an `init` and `bounds` dicts."""
        negative_init = {key: val for key, val in init.items() if val < 0}
        if negative_init:
            raise ValueError(f"Init parameters contain negative values {negative_init}.")
        negative_bounds = {key: val for key, val in bounds.items() if min(val) < 0}
        if negative_bounds:
            raise ValueError(f"Bounds parameters contain negative values {negative_bounds}.")
        reversed_bounds = {key: [left, right] for key, [left, right] in bounds.items() if left > right}
        if reversed_bounds:
            raise ValueError(f"Left bound is greater than right bound for {reversed_bounds}.")
        both_keys = {*init.keys()} & {*bounds.keys()}
        outbounds_keys = {key for key in both_keys if init[key] < bounds[key][0] or init[key] > bounds[key][1]}
        if outbounds_keys:
            raise ValueError(f"Init parameters are out of the bounds for {outbounds_keys} key(s).")

    def _validate_keys(self, checked_dict):
        """Check the keys of given dict for a minimum quantity, an excessive, and an insufficient."""
        expected_layers = len(checked_dict) // 2
        if expected_layers < 1:
            raise ValueError("Insufficient parameters to fit a velocity model.")
        missing_keys = set(self._get_valid_keys(expected_layers)) - set(checked_dict.keys())
        if missing_keys:
            raise ValueError("Insufficient parameters to fit a velocity model. ",
                            f"Check {missing_keys} key(s) or define `n_refractors`")
        excessive_keys = set(checked_dict.keys()) - set(self._get_valid_keys(expected_layers))
        if excessive_keys:
            raise ValueError(f"Excessive parameters to fit a velocity model. Remove {excessive_keys}.")

    def _postprocess_params(self, params):
        """Fix parameters if constraints were violated due to `scipy` inaccuracies."""
        params[self.n_refractors:] *= 1000
        # `self.piecewise_offsets` have the same offset values as params but also have the zero and max_offset
        for i in range(1, self.n_refractors):
            if self.piecewise_offsets[i + 1] < params[i]:
                params[i] = self.piecewise_offsets[i + 1]
        for i in range(self.n_refractors, self.n_refractors - 1):
            if params[i + 1] < params[i]:
                params[i + 1] = params[i]
        return params

    def _ms_to_kms(self, params, as_array=True):
        """Convert the velocity in the given dict with parameters from m/s to km/s."""
        values = np.array(list(params.values()), dtype=float)
        values[self.n_refractors:] = values[self.n_refractors:] / 1000
        if as_array:
            return values
        return dict(zip(self._valid_keys, values))

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

        ax.scatter(self.offsets, self.fb_times, s=1, color='black', label='first breaks')
        self._plot_lines(ax, curve_label='offset-traveltime curve', curve_color='red',
                         crossoffset_label='crossover point', crossover_color='blue')

        if show_params:
            params = [self.params[key] for key in self._valid_keys]
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
