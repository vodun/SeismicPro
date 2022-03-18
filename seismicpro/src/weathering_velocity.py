"""Implements WeatheringVelocity class to fit piecewise function and store parameters of a fitted function."""

from functools import partial

import numpy as np
from sklearn.linear_model import SGDRegressor
from scipy import optimize

from .decorators import plotter
from .utils import set_ticks, set_text_formatting

# pylint: disable=too-many-instance-attributes
class WeatheringVelocity:
    """The class fits and stores parameters of a weathering model based on gather's offsets and first break picking
    times.

    The weathering model is a velocity model of the first few subsurface layers. The class uses the intercept time,
    cross offsets and velocities parameters of the weathering model. The weathering model could be present as
    a piecewise linear function the picking time from the offsets.
    Since the class uses the first break picking time to fit the detected velocities, any underlaying layers should
    have a higher velocity than any of the overlaying.

    The class could be initialized with data and estimate parameters of weathering model.
        Data : first breaking points times and the corresponding offsets.
        Parameters : `init`, `bounds`, `n_layers`.
            `init` : dict with prior weathering model parameters. Read the keys notation below.
            `bounds` : dict with left and right bound for each weathering model parameter.
                       Read the keys notation below.
            `n_layers` : the quantity of the weathering model layers.

    The WeatheringVelocity could calculate the missing parameters from the passed parameters to simplify class
    initialization.
    Missing parameters calculate rules:
        `init` : calculated from `bounds`. lower bound + (upper bounds - lower bounds) / 3
                 calculated from `n_layers`. Read `_calc_init_by_layers` docs to get more info.
        `bounds` : calculated from `init`. The lower bound is init / 2, the upper bound is init * 2.
                   calculated from `n_layers`. `init` calculates first and uses the above expression.
        `n_layers`: calculated from `bounds`. Half of the `bounds` parameters.
                    calculated from `init`. Calculates `bounds` from `init` first and uses the above rule.

    Keys notation.
        `init`, `bounds` and `params` is dict with the common key notation.
            `t0`: a intercept time of the first subweathering layer. a double wave travel time to the first refractor.
                Measured in milliseconds.
            `x{i}`: offset where refracted wave from i-th layer comes at the same time with a refracted wave from
                    the underlaying layer. Measured in meters.
            `v{i}`: velocity of an i-th layer. Measured in km/s.
        Note: in some case a direct wave could be detected on close offsets. In this point `v1` is mean velocity
        of the direct wave layer, and `x1` is cross offsets where direct wave comes at the same time with a refracted
        wave from the first subweathering layer.

    In case you have partial information about `init` and `bounds` you could pass part of keys and values in an `init`
    dict and a remaining part of keys in a `bounds` dict. Be sure that you pass all the needed keys.
    Note: All passed parameters have a greater priority than any calculated parameters.

    Examples
    --------
    A Weathering Velocity object with starting initial parameters for two layers weathering model:
    >>> weathering_velocity = gather.calculate_weathering_velocity(init={'t0': 100, 'x1': 1500, 'v1': 2, 'v2': 3})

    A Weathering Velocity object with bounds for final parameters of a piecewise function for 1-layer weathering model:
    >>> weathering_velocity = gather.calculate_weathering_velocity(init={'t0': [0, 200], 'v1': [1, 3]})

    A Weathering Velocity object for three layers weathering model:
    >>> weathering_velocity = gather.calculate_weathering_velocity(n_layers=3)

    Also mixing parameters are possible (two layers weathering model):
    >>> weathering_velocity = gather.calculate_weathering_velocity(init={'x1': 200, 'v1': 2, 'v2': 3},
                                                                   bounds={'t0': [0, 50]},
                                                                   n_layers=2)

    Parameters
    ----------
    offsets : 1d ndarray
        Offsets of the traces in meters.
    picking_times : 1d ndarray
        First break picking times in milliseconds.
    init : dict, defaults to None
        Initial parameters of a weathering model.
    bounds : dict, defaults to None
        Left and right bounds of the weathering model parameters.
    n_layers : int, defaults to None
        Number of layers of a weathering model.
    ascending_velocity : bool, defaults to True
        Keeps the ascend of the fitted velocities from layer to layer.
    kwargs : dict, optional
        Additional keyword arguments to `scipy.optimize.minimize`.

    Attributes
    ----------
    offsets : 1d ndarray
        Offsets of traces in meters.
    picking_times : 1d ndarray
        Picking times of traces in milliseconds.
    init : dict
        The inital values used to fit the parameters of the weathering model. Includes the calculated non-passed keys
        and values. Have the common keys notation.
    bounds : dict
        The left and right bounds used to fit the parameters of the weathering model. Includes the calculated
        non-passed keys and values. Have the common keys notation.
    n_layers : int
        Number of the weathering model layers used to fit the parameters of the weathering model.
    params : dict
        The fitted parameters of a weathering model. Have the common keys notation.

    Raises
    ------
    ValueError
        if any `init` values are negative.
        if any `bounds` values are negative.
        if left bound greater than right bound.
        if init value is out of the bound interval.
        if passed `init` and/or `bounds` keys are insufficient or excessive.
        if an union of `init` and `bounds` keys less than 2 or `n_layers` less than 1.
    """

    def __init__(self, offsets, picking_times, n_layers=None, init=None, bounds=None, ascending_velocity=True,
                 **kwargs):
        init = {} if init is None else init
        bounds = {} if bounds is None else bounds
        self._check_values(init, bounds)

        self.offsets = offsets
        self.picking_times = picking_times

        self.init = {**self._calc_init_by_layers(n_layers), **self._calc_init_by_bounds(bounds), **init}
        self.bounds = {**self._calc_bounds_by_init(), **bounds}
        self._check_keys()
        self.n_layers = len(self.bounds) // 2
        self._valid_keys = self._get_valid_keys()

        # ordering `init` and `bounds` dicts to put all values in the required order.
        self.init = {key: self.init[key] for key in self._valid_keys}
        self.bounds = {key: self.bounds[key] for key in self._valid_keys}

        # piecewise func parameters
        self._piecewise_times = np.empty(self.n_layers + 1)
        self._piecewise_offsets = np.zeros(self.n_layers + 1)
        self._piecewise_offsets[-1] = offsets.max()
        self._current_args = np.array(list(self.init.values()))

        # Constraints define
        constraint_offset = {"type": "ineq", "fun": lambda x: np.diff(x[1:self.n_layers])}  # crossoffsets ascend.
        constraint_velocity = {"type": "ineq", "fun": lambda x: np.diff(x[self.n_layers:])}  # velocities ascend.
        constraint_freeze_velocity = {  # freeze the velocity fitting if no data for layer is found.
            "type": "eq",
            "fun": lambda x: self._current_args[self.n_layers:][self._mask_empty_layers(self.init)] \
                             - x[self.n_layers:][self._mask_empty_layers(self.init)]}
        constraint_freeze_time = {  # freeze the intercept time fitting if no data for layer is found.
            "type": "eq",
            "fun": lambda x: self._current_args[0][self._mask_empty_layers(self.init)[0]] \
                             - x[0][self._mask_empty_layers(self.init)[0]]}
        constraints = [constraint_offset, constraint_freeze_velocity, constraint_freeze_time]
        if ascending_velocity:
            constraints.append(constraint_velocity)

        # Fitting piecewise linear regression
        partial_loss_func = partial(self.loss_piecewise_linear, loss=kwargs.pop('loss', 'L1'),
                                    huber_coef=kwargs.pop('huber_coef', .1))
        minimizer_kwargs = {'method': 'SLSQP', 'constraints': constraints, **kwargs}
        self._model_params = optimize.minimize(partial_loss_func, x0=list(self.init.values()),
                                               bounds=list(self.bounds.values()), **minimizer_kwargs)
        self.params = dict(zip(self._valid_keys, self._model_params.x))
        self.empty_layers = self._mask_empty_layers(self.params)

    def __call__(self, offsets):
        """Return predicted picking times using offsets and the fitted parameters of the weathering model."""
        return np.interp(offsets, self._piecewise_offsets, self._piecewise_times)

    def __getattr__(self, key):
        return self.params[key]

    def _update_piecewise_params(self, args):
        """Update the parameters of piecewise linear function stored in class attributes."""
        self._current_args = args
        self._piecewise_times[0] = args[0]
        self._piecewise_offsets[1:self.n_layers] = args[1:self.n_layers]

        for i in range(self.n_layers):
            self._piecewise_times[i + 1] = ((self._piecewise_offsets[i + 1] - self._piecewise_offsets[i]) /
                                             args[self.n_layers + i]) + self._piecewise_times[i]

    def loss_piecewise_linear(self, args, loss='L1', huber_coef=.1):
        """Update the piecewise linear attributes and returns the loss function result.

        Method calls `_update_piecewise_params` to update piecewise linear attributes of a WeatheringVelocity instance.
        After that, the method calculates the loss function between the true picking times stored in
        the `self.picking_times` and predicted piecewise linear function. The points at which the loss function
        is calculated correspond to the offset.

        Piecewise linear function is defined by the given `args`. `args` should be list-like and have the following
        structure:
            args[0] : t0
            args[1:n_layers] : cross offsets points in meters.
            args[n_layers:] : velocities of each weathering model layer in km/s.
            Total lenght of args should be n_layers * 2.
        The list-like initial is due to the `scipy.optimize.minimize`.

        Parameters
        ----------
        args : tuple, list, or 1d ndarray
            Parameters of the piecewise linear function.
        loss : str, optional, defaults to 'L1'.
            The loss function type. Should be one of 'L1', 'huber', 'soft_L1', or 'cauchy'.
            All implemented loss functions have a mean reduction.
        huber_coef : float, default to 0.1
            Delta coefficient for Huber loss.

        Returns
        -------
        loss : float
            Loss function result between true picking times and a predicted piecewise linear function.

        Raises
        ------
        ValueError
            If given `loss` does not exist.

        """
        self._update_piecewise_params(args)
        diff_abs = np.abs(np.interp(self.offsets, self._piecewise_offsets, self._piecewise_times) - self.picking_times)
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
        """Returns a valid list with keys based on `n_layers` or `self.n_layers`."""
        n_layers = self.n_layers if n_layers is None else n_layers
        return ['t0'] + [f'x{i+1}' for i in range(n_layers - 1)] + [f'v{i+1}' for i in range(n_layers)]

    def _fit_regressor(self, x, y, start_slope, start_time, fit_intercept):
        """Method fits the linear regression by given data and initinal values.

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
        fit_intercept : bool
            Fit the intercept with `True` or hold it with `False`.

        Returns
        -------
        params : tuple
            Linear regression `coef` and `intercept`.
        """
        lin_reg = SGDRegressor(loss='huber', early_stopping=True, penalty=None, shuffle=True, epsilon=0.1,
                               eta0=.05, alpha=0, tol=1e-4, fit_intercept=fit_intercept)
        lin_reg.fit(x, y, coef_init=start_slope, intercept_init=start_time)
        return lin_reg.coef_[0], lin_reg.intercept_

    def _calc_init_by_layers(self, n_layers):
        """Calculates `init` dict by a given an estimated quantity of layers.

        Method splits the picking times into `n_layers` equal part by cross offsets and fits a separate linear
        regression on each part. Fitting the coefficient and intercept for the first part and fitting the coefficient
        only for any next part. These linear functions are compiled together as a piecewise linear function.
        Parameters of this piecewise function are recalculated to the weathering model parameters and returned as
        `init` dict.

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

        min_picking_times = self.picking_times.min()  # normalization parameter.
        start_slope = 2/3  # base slope corresponding velocity is 1,5 km/s (v = 1 / slope)
        start_time = 1  # base time, equal to minimum picking times with the `min_picking` normalization.
        normalized_offsets = self.offsets / min_picking_times
        normalized_times = self.picking_times / min_picking_times

        # split cross offsets on an equal intervals
        cross_offsets = np.linspace(0, normalized_offsets.max(), num=n_layers+1)
        slopes = np.empty(n_layers)
        times = np.empty(n_layers)

        for i in range(n_layers):  # inside the loop work only with normalized data
            mask = (normalized_offsets > cross_offsets[i]) & (normalized_offsets <= cross_offsets[i + 1])
            if mask.sum() > 1:  # at least two point to fit
                slopes[i], times[i] = self._fit_regressor(normalized_offsets[mask].reshape(-1, 1),
                                                          normalized_times[mask], start_slope, start_time,
                                                          fit_intercept=(i==0))
            else:
                slopes[i] = start_slope
                times[i] = start_time - start_slope * normalized_offsets.min() * (i==0)  # add first layer correction
            slopes[i] = max(.167, slopes[i])  # move maximal velocity to 6 km/s
            times[i] = max([0, times[i]])  # move minimal time to zero
            start_slope = slopes[i] * (n_layers / (n_layers + 1)) # raise base velocity for next layers (v = 1 / slope)
            start_time = times[i] + (slopes[i] - start_slope) * cross_offsets[i + 1]
        velocities = 1 / slopes
        init = np.hstack((times[0] * min_picking_times, cross_offsets[1:-1] * min_picking_times, velocities))
        init = dict(zip(self._get_valid_keys(n_layers), init))
        return init

    def _calc_init_by_bounds(self, bounds):
        """Method returns dict with a calculated init from a bounds dict."""
        return {key: val1 + (val2 - val1) / 3 for key, (val1, val2) in bounds.items()}

    def _calc_bounds_by_init(self):
        """Method returns dict with calculated bounds from a init dict."""
        bounds = {key: [val / 2, val * 2] for key, val in self.init.items()}
        bounds['t0'] = [min(0, bounds['t0'][0]), max(200, bounds['t0'][1])]
        return bounds

    def _check_values(self, init, bounds):
        """Check the values of an `init` and `bounds` dicts."""
        negative_init = {key: val for key, val in init.items() if val < 0}
        if negative_init:
            raise ValueError(f"Init parameters contain negative values {str(negative_init)[1:-1]}")
        negative_bounds = {key: val for key, val in bounds.items() if min(val) < 0}
        if negative_bounds:
            raise ValueError(f"Bounds parameters contain negative values {str(negative_bounds)[1:-1]}")
        reversed_bounds = {key: [left, right] for key, [left, right] in bounds.items() if left > right}
        if reversed_bounds:
            raise ValueError(f"Left bound is greater than right bound for {list(reversed_bounds.keys())} key(s).")
        both_keys = {*init.keys()} & {*bounds.keys()}
        outbounds_keys = {key for key in both_keys if init[key] < bounds[key][0] or init[key] > bounds[key][1]}
        if outbounds_keys:
            raise ValueError(f"Init parameters are out of the bounds for {outbounds_keys} key(s).")

    def _check_keys(self):
        """Check the `self.bounds` keys for a minimum quantity, an excessive, and an insufficient."""
        expected_layers = len(self.bounds) // 2
        if expected_layers < 1:
            raise ValueError("Insufficient parameters to fit a weathering velocity curve.")
        missing_keys = set(self._get_valid_keys(expected_layers)) - set(self.bounds.keys())
        if missing_keys:
            raise ValueError("Insufficient parameters to fit a weathering velocity curve. ",
                            f"Check {missing_keys} key(s) or define `n_layers`")
        excessive_keys = set(self.bounds.keys()) - set(self._get_valid_keys(expected_layers))
        if excessive_keys:
            raise ValueError(f"Excessive parameters to fit a weathering velocity curve. Remove {excessive_keys}.")

    def _mask_empty_layers(self, params):
        """Method checks the layers for data and returns boolean mask. The mask value is 1 if no data is found."""
        cross_offsets = [0, *list(params.values())[1:self.n_layers], self.offsets.max()]
        mask = [self.offsets[(self.offsets > cross_offsets[i]) & (self.offsets <= cross_offsets[i+1])].shape[0] < 1
                for i in range(self.n_layers)]
        return mask

    def _calc_piecewise_coords_from_params(self, params):
        """Method calculate coords for the piecewise linear curve from params dict.

        Parameters
        ----------
        params : dict,
            Dict with parameters of weathering model. Keys have the common notation.

        Returns
        -------
        offsets : 1d npdarray
            coords of the points on the x axis
        times : 1d ndarray
            coords of the points on the y axis
        """
        comparing_layers = len(params) // 2
        keys = self._get_valid_keys(n_layers=comparing_layers)
        params = {key: params[key] for key in keys}
        params_values = list(params.values())

        offsets = np.empty(comparing_layers + 1)
        offsets[1:comparing_layers] = params_values[1:comparing_layers]
        offsets[-1] = self.offsets.max()

        times = np.zeros(comparing_layers + 1)
        times[0] = params_values[0]
        for i in range(comparing_layers):
            times[i + 1] = ((offsets[i + 1] - offsets[i]) / params_values[comparing_layers + i]) + times[i]
        return offsets, times

    @plotter(figsize=(10, 5))
    def plot(self, *, ax=None, title=None, x_ticker=None, y_ticker=None, show_params=True, threshold_time=None,
            compared_params=None, **kwargs):
        """Plot the WeatheringVelocity data, fitted curve, cross offsets, and additional information.

        Parameters
        ----------
        title : str, optional, defaults to None
            Plot title.
        x_ticker : dict, optional, defaults to None
            Parameters for ticks and ticklabels formatting for the x-axis; see `.utils.set_ticks` for more details.
        y_ticker : dict, optional, defaults to None
            Parameters for ticks and ticklabels formatting for the y-axis; see `.utils.set_ticks` for more details.
        show_params : bool, optional, defaults to True
            Shows the weathering model parameters on a plot.
        threshold_time : int or float, optional. Defaults to None
            Gap for plotting two outlines. If None additional outlines don't show.
        compared_params : dict, optional, defaults to None
            Dict with another weathering velocity params. Should have common keys notation.

        Returns
        -------
        self : WeatheringVelocity
            WeatheringVelocity without changes.
        """
        txt_kwargs = {**{'fontsize': 15, 'va': 'top'}, **kwargs.pop('txt_kwargs', {})}
        txt_ident = txt_kwargs.pop('ident', (.03, .94))

        (title, x_ticker, y_ticker), kwargs = set_text_formatting(title, x_ticker, y_ticker, **kwargs)
        set_ticks(ax, "x", tick_labels=None, label="offset, m", **x_ticker)
        set_ticks(ax, "y", tick_labels=None, label="time, ms", **y_ticker)

        ax.scatter(self.offsets, self.picking_times, s=1, color='black', label='fbp points')
        for i in range(self.n_layers):
            if self.params[f'v{i+1}'] is not np.nan:
                ax.plot(self._piecewise_offsets[i:i+2], self._piecewise_times[i:i+2], '-', color='red',
                        label='fitted piecewise function' if i == 0 else None)
            if i != self.n_layers - 1:
                ax.axvline(self._piecewise_offsets[i+1], 0, self.picking_times.max(), ls='--', color='blue',
                        label='crossover point(s)' if i == 0 else None)
        if show_params:
            params = [self.params[key] for key in self._valid_keys]
            txt_info = f"t0 : {params[0]:.2f} ms"
            if self.n_layers > 1:
                txt_info += '\ncrossover offsets : ' + ', '.join(f"{round(x)}" for x in params[1:self.n_layers]) + ' m'
            txt_info += '\nvelocities : ' + ', '.join(f"{v:.2f}" for v in params[self.n_layers:]) + ' km/s'
            ax.text(*txt_ident, txt_info, transform=ax.transAxes, **txt_kwargs)

        if threshold_time is not None:
            ax.plot(self._piecewise_offsets, self._piecewise_times + threshold_time, '--', color='red',
                    label=f'+/- {threshold_time}ms window')
            ax.plot(self._piecewise_offsets, self._piecewise_times - threshold_time, '--', color='red')
        if compared_params is not None:
            compared_offsets, compared_times = self._calc_piecewise_coords_from_params(compared_params)
            ax.plot(compared_offsets, compared_times, '-', color='#ff7900', label='compared piecewise function')
        ax.set_xlim(0)
        ax.set_ylim(0)
        ax.legend(loc='lower right')
        return self
