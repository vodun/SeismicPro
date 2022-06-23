"""Implements RefractorVelocity class for estimating the velocity model of an upper part of the section."""

from functools import partial

import numpy as np
from sklearn.linear_model import SGDRegressor
from scipy import optimize

from ..decorators import plotter
from ..utils import set_ticks, set_text_formatting
from ..utils.interpolation import interp1d


# pylint: disable=too-many-instance-attributes, protected-access
class RefractorVelocity:
    """The class stores and fits parameters of a velocity model of an upper part of the section.

    The class could be created from `from_first_breaks`, `from_params`, and `from_constant_velocity` classmethods.
        * `from_first_breaks` - creates a RefractorVelocity instance, fits the parameters of the velocity model by
                                the offsets, first break times, estimate parameters of the velocity model and stores
                                the fitted parameters.
        * `from_params` - creates a RefractorVelocity instance and stores the given parameters of the velocity model.
        * `from_constant` - creates a RefractorVelocity instance based on the 1-layer velocity model with zero
                            intercept time and given velocity. Stored parameters of the velocity model in `params`.

    Some of class attributes and classmethod's arguments have a valid key notation.
        `t0`: a intercept time of the first subweathering layer. Same as two-way travel time to the first
              refractor. Measured in milliseconds.
        `x{i}`: cross offset. The offset where refracted wave from the i-th layer comes at the same time with
                a refracted wave from the next underlaying layer. Measured in meters.
        `v{i}`: velocity of the i-th layer. Measured in meters per seconds.

    The RefractorVelocity created with `from_first_breaks` classmethod calculates the missing parameters from passed
    parameters. The passed `init`, `bounds` and `n_layers` parameters combine with the calculated parameters and stored
    in class atributes with corresponding names. All passed parameters have a greater priority.

    Examples
    --------
    Create the RefractorVelocity instance with `from_params` classmethod (avoid the fitting procedure):
    >>> refractor_velocity = RefractorVelocity.from_params(params={'t0': 100, 'x1': 1500, 'v1': 2, 'v2': 3})

    Create the RefractorVelocity instance with `from_first_breaks` classmethod :
    >>> survey = Survey(survey_path, header_index="FieldRecord", header_cols=["offset", "TraceNumber"])
    >>> survey = survey.load_first_breaks(first_breaks_path)
    Load the offsets and first breaks times from a randomly selected common source gather.
    >>> gather = survey.sample_gather()
    >>> offsets = gahter.offsets
    >>> fb_times = gather['FirstBreak'].ravel()

    Two layers refractor model by `n_layers`:
    >>> refractor_velocity = RefractorVelocity.from_first_breaks(offsets, fb_times, n_layers=2)

    The same can be done by calling the `calculate_refractor_velocity` method of the gather:
    >>> refractor_velocity = gather.calculate_refractor_velocity(n_layers=2)

    Two layers refractor model by `init`:
    >>> initial_params = {'t0': 100, 'x1': 1500, 'v1': 2, 'v2': 3}
    >>> refractor_velocity = RefractorVelocity.from_first_breaks(offsets, fb_times, init=initial_params)

    One layers refractor model by `bounds`:
    >>> refractor_velocity = RefractorVelocity.from_first_breaks(offsets, fb_times,
                                                                 bounds={'t0': [0, 200], 'v1': [1, 3]})

    Three layers refractor model by mixing parameters:
    >>> refractor_velocity = RefractorVelocity.from_first_breaks(offsets, fb_times, init={'x1': 200, 'v1': 1},
                                                                                    bounds={'t0': [0, 50]},
                                                                                    n_layers=3)

    Attributes
    ----------
    offsets : 1d ndarray
        Offsets of traces. Measured in meters.
    fb_times : 1d ndarray
        First breaks times of traces. Measured in milliseconds.
    max_offsets : float
        Maximum offset value.
    coords : Coordinates
        Spatial coordinates at which refractor velocity is estimated.
    init : dict
        The inital values used to fit the parameters of the velocity model. Includes the calculated non-passed
        keys and values. Have the valid key notation.
    bounds : dict
        The left and right bounds used to fit the parameters of the velocity model. Includes the calculated
        non-passed keys and values. Have the valid key notation.
    n_layers : int
        Number of the layers used to fit the parameters of the velocity model.
    params : dict
        The parameters of a velocity model. Have the valid key notation.
    interpolator : callable
        An interpolator returning expected first breaking by given offset.
    """
    def __init__(self):
        self.offsets = None
        self.fb_times = None
        self.max_offset = None
        self.coords = None
        self.init = None
        self.bounds = None
        self.n_layers = None
        self.params = None
        self.interpolator = None

        self._valid_keys = None
        self._empty_layers = None
        self._piecewise_offsets = None
        self._piecewise_times = None
        self._model_params = None

    @classmethod
    def from_first_breaks(cls, offsets, fb_times, init=None, bounds=None, n_layers=None, coords=None, **kwargs):
        """Create RefractorVelocity instanse from the offsets and first breaks times and fits the velocity model
        parameters.

        Parameters
        ----------
        offsets : 1d ndarray
            Offsets of the traces. Measured in meters.
        fb_times : 1d ndarray
            First break times. Measured in milliseconds.
        init : dict, defaults to None
            Initial parameters of a velocity model. Should have the valid key notation.
        bounds : dict, defaults to None
            Lower and upper bounds of the velocity model parameters. Should have the valid key notation.
        n_layers : int, defaults to None
            Number of layers of a velocity model.
        kwargs : dict, optional
            Additional keyword arguments to `scipy.optimize.minimize`.

        Raises
        ------
        ValueError
            If all `init`, `bounds`, and `n_layers` are None.
            If any `init` values are negative.
            If any `bounds` values are negative.
            If left bound greater than right bound.
            If init value is out of the bound interval.
            If passed `init` and/or `bounds` keys are insufficient or excessive.
            If an union of `init` and `bounds` keys less than 2 or `n_layers` less than 1.
        """
        self = cls()
        if all((param is None for param in (init, bounds, n_layers))):
            raise ValueError("One of `init`, `bounds` or `n_layers` should be defined.")
        init = {} if init is None else init
        bounds = {} if bounds is None else bounds
        self._validate_values(init, bounds)

        self.offsets = offsets
        self.fb_times = fb_times
        self.max_offset = offsets.max()
        self.coords = coords

        self.init = {**self._calc_init_by_layers(n_layers), **self._calc_init_by_bounds(bounds), **init}
        self.bounds = {**self._calc_bounds_by_init(), **bounds}
        self._validate_keys(self.bounds)
        self.n_layers = len(self.bounds) // 2
        self._valid_keys = self._get_valid_keys()

        # ordering `init` and `bounds` dicts to put all values in the required order.
        self.init = {key: self.init[key] for key in self._valid_keys}
        self.bounds = {key: self.bounds[key] for key in self._valid_keys}

        # piecewise func parameters
        self._piecewise_offsets, self._piecewise_times = self._create_piecewise_coords(self.n_layers, offsets.max())
        self._piecewise_offsets, self._piecewise_times = \
            self._update_piecewise_coords(self._piecewise_offsets, self._piecewise_times,
                                          self._ms_to_kms(self.init), self.n_layers)
        self._empty_layers = np.histogram(self.offsets, self._piecewise_offsets)[0] ==  0

        constraints_list = self._get_constraints()

        # fitting piecewise linear regression
        partial_loss_func = partial(self.loss_piecewise_linear, loss=kwargs.pop('loss', 'L1'),
                                    huber_coef=kwargs.pop('huber_coef', .1))
        minimizer_kwargs = {'method': 'SLSQP', 'constraints': constraints_list, **kwargs}
        self._model_params = optimize.minimize(partial_loss_func, x0=self._ms_to_kms(self.init),
                                               bounds=self._ms_to_kms(self.bounds), **minimizer_kwargs)
        self.params = dict(zip(self._valid_keys, self._params_postprocceissing(self._model_params.x)))
        self.interpolator = interp1d(self._piecewise_offsets, self._piecewise_times)
        return self

    @classmethod
    def from_params(cls, params, coords=None):
        """Create RefractorVelocity instanse from parameters.

        Parameters
        ----------
        params : dict,
            Parameters of the velocity model. Should have the valid key notation.

        Returns
        -------
        RefractorVelocity
            RefractorVelocity instance based on passed params.

        Raises
        ------
        ValueError
            If passed `params` keys are insufficient or excessive.
        """
        self = cls()

        self._validate_keys(params)
        self.n_layers = len(params) // 2
        self._valid_keys = self._get_valid_keys(self.n_layers)
        self.params = {key: params[key] for key in self._valid_keys}
        self.coords = coords

        self._piecewise_offsets, self._piecewise_times = \
            self._create_piecewise_coords(self.n_layers, self.params.get(f'x{self.n_layers - 1}', 0) + 1000)
        self._piecewise_offsets, self._piecewise_times = \
            self._update_piecewise_coords(self._piecewise_offsets, self._piecewise_times, self._ms_to_kms(self.params),
                                          self.n_layers)
        self.interpolator = interp1d(self._piecewise_offsets, self._piecewise_times)
        return self

    @classmethod
    def from_constant_velocity(cls, velocity, coords=None):
        """Create RefractorVelocity instanse with constant velocity and intercept time is 0.

        Parameters
        ----------
        velocity : float,
            Constant velocity for the 1-layer velocity model.

        Returns
        -------
        RefractorVelocity
            RefractorVelocity instance based on velocity.

        Raises
        ------
        ValueError
            If passed `velocity` is negative.
        """
        self = cls()

        if velocity < 0:
            raise ValueError("Velocity should not be negative.")
        return self.from_params({"t0": 0, "v1": velocity}, coords=coords)

    def __call__(self, offsets):
        """Return the expected first breaks times using the offsets."""
        return self.interpolator(offsets)

    def __getattr__(self, key):
        return self.params[key]

    def __getitem__(self, value):
        """Return ndarray with i-th refractor parameters (start offset, end offset, velocity). Starts from 0."""
        if value >= self.n_layers:
            raise IndexError(f"Index {value} is out of bounds.")
        velocity = self.params.get(f"v{value + 1}", None)
        return np.hstack([self._piecewise_offsets[value:value + 2], velocity])

    def has_coords(self):
        """bool: Whether RefractorVelocity coords are not None."""
        return self.coords is not None

    def get_coords(self, *args, **kwargs):
        """Get spatial coordinates of the RefractorVelocity.

        Ignores all passed arguments but accept them to preserve general `get_coords` interface.

        Returns
        -------
        coords : Coordinates
            RefractorVelocity spatial coordinates.
        """
        _ = args, kwargs
        return self.coords

    def _create_piecewise_coords(self, n_layers, max_offset=np.nan):
        """Create two array corresponding to the piecewise linear function coords."""
        piecewise_offsets = np.zeros(n_layers + 1)
        piecewise_times = np.zeros(n_layers + 1)
        piecewise_offsets[-1] = max_offset
        return piecewise_offsets, piecewise_times

    def _update_piecewise_coords(self, piecewise_offsets, piecewise_times, params, n_layers):
        """Update the given `offsets` and `times` arrays based on the `params` and `n_layers`."""
        piecewise_times[0] = params[0]
        piecewise_offsets[1:n_layers] = params[1:n_layers]
        for i in range(n_layers):
            piecewise_times[i + 1] = ((piecewise_offsets[i + 1] - piecewise_offsets[i]) / params[n_layers + i]) + \
                                     piecewise_times[i]
        return piecewise_offsets, piecewise_times

    def loss_piecewise_linear(self, args, loss='L1', huber_coef=.1):
        """Update the piecewise linear attributes and returns the loss function result.

        Method calls `_update_piecewise_coords` to update piecewise linear attributes of a RefractorVelocity instance.
        After that, the method calculates the loss function between the true first breaks times stored in the
        `self.fb_times` and predicted piecewise linear function. The loss function is calculated at the offsets points.

        Piecewise linear function is defined by the given `args`. `args` should be list-like and have the following
        structure:
            args[0] : intercept time in milliseconds.
            args[1:n_layers] : cross offsets points in meters.
            args[n_layers:] : velocities of each layer in kilometers/seconds.
            Total lenght of args should be n_layers * 2.

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
        huber_coef : float, default to 0.1
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
        self._piecewise_offsets, self._piecewise_times = \
            self._update_piecewise_coords(self._piecewise_offsets, self._piecewise_times, args, self.n_layers)
        diff_abs = np.abs(np.interp(self.offsets, self._piecewise_offsets, self._piecewise_times) - self.fb_times)
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

    def _get_valid_keys(self, n_layers=None):
        """Returns a list with the valid keys based on `n_layers` or `self.n_layers`."""
        n_layers = self.n_layers if n_layers is None else n_layers
        return ['t0'] + [f'x{i + 1}' for i in range(n_layers - 1)] + [f'v{i + 1}' for i in range(n_layers)]

    def _get_constraints(self):
        """Define the constraints and return a list them."""
        constraint_offset = {  # cross offsets ascend.
            "type": "ineq",
            "fun": lambda x: np.diff(np.concatenate((x[1:self.n_layers], [self.max_offset])))}
        constraint_velocity = {  # velocities ascend.
            "type": "ineq",
            "fun": lambda x: np.diff(x[self.n_layers:])}
        constraint_freeze_velocity = {  # freeze the velocity if no data for layer is found.
            "type": "eq",
            "fun": lambda x: self._ms_to_kms(self.init)[self.n_layers:][self._empty_layers]
                             - x[self.n_layers:][self._empty_layers]}
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
        lin_reg = SGDRegressor(loss='huber', early_stopping=True, penalty=None, shuffle=True, epsilon=0.1, eta0=.1,
                               alpha=0., tol=1e-6)
        lin_reg.fit(x, y, coef_init=start_slope, intercept_init=start_time)
        return lin_reg.coef_[0], lin_reg.intercept_

    def _calc_init_by_layers(self, n_layers):
        """Calculates `init` dict by a given an estimated quantity of layers.

        Method splits the first breaks times into `n_layers` equal part by cross offsets and fits a separate linear
        regression on each part. These linear functions are compiled together as a piecewise linear function.
        Parameters of piecewise function are calculated to the velocity model parameters and returned as `init` dict.

        Parameters
        ----------
        n_layers : int
            Number of layers.

        Returns
        -------
        init : dict
            Estimated initial to fit the piecewise linear function.
        """
        if n_layers is None or n_layers < 1:
            return {}

        max_fb_times = self.fb_times.max()  # times normalization parameter.
        initial_slope = 1  # base slope corresponding velocity is 1 km/s (v = 1 / slope)
        initial_slope = initial_slope * self.max_offset / max_fb_times  # add normalization
        initial_time = 0
        normalized_offsets = self.offsets / self.max_offset
        normalized_times = self.fb_times / max_fb_times

        cross_offsets = np.linspace(0, 1, num=n_layers+1)  # split cross offsets on an equal intervals
        current_slope = np.empty(n_layers)
        current_time = np.empty(n_layers)

        for i in range(n_layers):  # inside the loop work with normalized data only
            mask = (normalized_offsets > cross_offsets[i]) & (normalized_offsets <= cross_offsets[i + 1])
            if mask.sum() > 1:  # at least two point to fit
                current_slope[i], current_time[i] = \
                    self._fit_regressor(normalized_offsets[mask].reshape(-1, 1), normalized_times[mask],
                                        initial_slope, initial_time)
            else:
                current_slope[i] = initial_slope
                current_time[i] = initial_time
            # move maximal velocity to 6 km/s
            current_slope[i] = max(.167 * self.max_offset / max_fb_times, current_slope[i])
            current_time[i] =  max(0, current_time[i])
            # raise base velocity for the next layer (v = 1 / slope)
            initial_slope = current_slope[i] * (n_layers / (n_layers + 1))
            initial_time = current_time[i] + (current_slope[i] - initial_slope) * cross_offsets[i + 1]
        velocities = 1 / (current_slope * max_fb_times / self.max_offset)
        init = [current_time[0] * max_fb_times, *cross_offsets[1:-1] * self.max_offset, *(velocities * 1000)]
        init = dict(zip(self._get_valid_keys(n_layers), init))
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
                            f"Check {missing_keys} key(s) or define `n_layers`")
        excessive_keys = set(checked_dict.keys()) - set(self._get_valid_keys(expected_layers))
        if excessive_keys:
            raise ValueError(f"Excessive parameters to fit a velocity model. Remove {excessive_keys}.")

    def _params_postprocceissing(self, params):
        """Fix parameters if constraints are not respected due to `scipy` breadth calculation."""
        params[self.n_layers:] *= 1000
        # `self._piecewise_offsets` have the same offset values as params but also have the zero and max_offset
        for i in range(1, self.n_layers):
            if self._piecewise_offsets[i + 1] < params[i]:
                params[i] = self._piecewise_offsets[i + 1]
        for i in range(self.n_layers, self.n_layers - 1):
            if params[i + 1] < params[i]:
                params[i + 1] = params[i]
        return params

    def _ms_to_kms(self, params, as_array=True):
        """Convert the velocity in the given valid dict of parameters from m/s to km/s."""
        values = np.array(list(params.values()), dtype=float)
        values[self.n_layers:] = values[self.n_layers:] / 1000
        if as_array:
            return values
        return dict(zip(self._valid_keys, values))

    @plotter(figsize=(10, 5))
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
            Shows the velocity model parameters on a plot.
        threshold_times : float or None, optional. Defaults to None
            Neighborhood margins of the fitted curve to fill in the area inside. If None the area don't show.
        compare_to : RefractorVelocity or None, optional, defaults to None
            RefractorVelocity instance. Used to plot an additional RefractorVelocity on the same axis.
        text_kwargs : dict, optional
            Additional arguments to the `matplotlib.pyplot.text` function. This function plot velocity model parameters
            on the plot.
        kwargs : dict, optional
            Additional keyword arguments to the plotter.

        Returns
        -------
        self : RefractorVelocity
            RefractorVelocity without changes.
        """
        (title, x_ticker, y_ticker, text_kwargs), kwargs = set_text_formatting(title, x_ticker, y_ticker, text_kwargs,
                                                                              **kwargs)
        set_ticks(ax, "x", tick_labels=None, label="offset, m", **x_ticker)
        set_ticks(ax, "y", tick_labels=None, label="time, ms", **y_ticker)

        crossoffset_label = kwargs.get('crossoffset_label', 'crossover point')
        if 'compared' not in crossoffset_label:
            ax.scatter(self.offsets, self.fb_times, s=1, color='black', label='first breaks')
        ax.plot(self._piecewise_offsets, self._piecewise_times, '-', color=kwargs.get('curve_color', 'red'),
                label=kwargs.get('curve_label', 'offset-traveltime curve'))
        for i in range(self.n_layers - 1):
            ax.axvline(self._piecewise_offsets[i+1], 0, ls='--', color=kwargs.get('crossover_color', 'blue'),
                       label=crossoffset_label if self.n_layers <= 2 else crossoffset_label + 's' if i == 0 else None)
        if show_params:
            params = [self.params[key] for key in self._valid_keys]
            text_info = f"t0 : {params[0]:.2f} ms"
            if self.n_layers > 1:
                text_info += '\ncrossover offsets : ' + ', '.join(f"{round(x)}" for x in params[1:self.n_layers]) \
                              + ' m'
            text_info += '\nvelocities : ' + ', '.join(f"{v:.0f}" for v in params[self.n_layers:]) + ' m/s'
            text_kwargs = {'fontsize': 12, 'va': 'top', **text_kwargs}
            text_ident = text_kwargs.pop('x', .03), text_kwargs.pop('y', .94)
            ax.text(*text_ident, text_info, transform=ax.transAxes, **text_kwargs)

        if threshold_times is not None:
            ax.fill_between(self._piecewise_offsets, self._piecewise_times - threshold_times,
                            self._piecewise_times + threshold_times, color='red',
                            label=f'+/- {threshold_times}ms threshold area', alpha=.2)
        if compare_to is not None:
            compare_to.plot(ax=ax, show_params=False, curve_color='#ff7900', crossover_color='green',
                             curve_label='compared offset-traveltime curve',
                             crossoffset_label='compared crossover point')
        ax.set_xlim(0)
        ax.set_ylim(0)
        ax.legend(loc='lower right')
        ax.set_title(**{"label": None, **title})
        return self
