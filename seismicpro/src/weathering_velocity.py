"""Implements WeatheringVelocity class to fit piecewise function and store parameters of a fitted function."""

import warnings

import numpy as np
from sklearn.linear_model import SGDRegressor
from scipy import optimize

from .decorators import plotter

# pylint: disable=too-many-instance-attributes
class WeatheringVelocity:
    """The class fits and stores parameters of a weathering and some subweathering layers based on gather's offsets
    and first break picking times.

    The class could be initialized with data and estimate parameters of weathering model. Data is a first breaking
    points times and corresponding offsets. Model parameters could be passed by `init`, `bounds`, `n_layers`, or a mix.

    `init` should be dict with the explained below keys and estimate values of model parameters. When bounds are not
    defined (or anyone of bounds keys) init value is used to calculate bounds. The lower bound is init / 2 and
    the upper bound is init * 2.
    `bounds` should be dict with the same keys and lists with lower and upper bounds. Fitted values could not be out of
    bounds. When init are not defined (or anyone of init keys) missing `init` key's values are calculated from bounds as
    lower bound + (upper bounds - lower bounds) / 3 to be consistent with `init` to `bounds` processing.

    WeatheringVelocity uses the following key notation for `init` and `bounds`:
        `t0`: a double wave travel time to the weathering layer's base.
        `x1`: offsets where refracted wave from first subweathering layer comes at the same time with a reflected wave.
        `x{i}`: offset where refracted wave from i-th subweathering layer comes at the same time with a refracted wave
                from the previous layer.
        `v1`: velocity of a weathering layer.
        `v{i}`: velocity of an i-th layer. Indexing of subweathering layers starts with the second number.
    Fitted parameters are stored in the `params` attribute as a dict with a stated above keys.

    `n_layers` is the estimated number of total weathering and subwethering layers and could be useful if you haven't
    information about `init` or `bounds`. Used for a calculation reasonable estimation of an `init` and `bounds`.

    In case you have partial information about `init` and `bounds` you could pass part of keys and values in an `init`
    dict and a remaining part of keys in a `bounds` dict. Be sure that you pass all the needed keys.

    All passed parameters have a greater priority than any calculated parameters.

    Examples
    --------
    A Weathering Velocity object with starting initial parameters for two layers weathering model:
    >>> weathering_velocity = gather.calculate_weathering_velocity(init={'t0': 100, 'x1': 1500, 'v1': 2, 'v2': 3})

    A Weathering Velocity object with bounds for final parameters of a piecewise function for 1-layer weathering model:
    >>> weathering_velocity = gather.calculate_weathering_velocity(init={'t0': [0, 200], 'v1': [1, 3]})

    A Weathering Velocity object for three layers weathering model:
    >>> weathering_velocity = gather.calculate_weathering_velocity(n_layers=3)

    Also mixing parameters are possible (two layers weathering model):
    >>> weathering_velocity = gather.calculate_weathering_velocity(init={'t0': 100, 'x1': 1500},
                                                                   bounds={'v1': [1, 3], 'v2': [1, 5]})
    Note: follow closely for keys unions fullness of a passed dicts.

    Parameters
    ----------
    offsets : 1d ndarray
        Offsets of a traces.
    picking_times : 1d ndarray
        Picking times of a traces.
    init : dict, defaults to None
        Initial  values for fitting a piecewise function. Used to calculate `bounds` if these params not passed.
    bounds : Dict[List], defaults to None
        Left and right bounds for any parameter of a weathering model params. Used to calculate `init` and `n_layers`
        if these params are not passed.
    n_layers : int, defaults to None
        Quantity of layers of a weathering model. Used to calculate `init` and `bounds` if these params not enough.
    kwargs : dict, optional
        Additional keyword arguments to `scipy.optimize.minimize`.

    Attributes
    ----------
    offsets : 1d ndarray
        Offsets of traces.
    max_offset : int
        Maximum offsets value.
    picking_times : 1d ndarray
        Picking times of traces.
    init : dict
        Inital values are used to fit the weathering model params. Include the calculated unpassed keys and values.
    bounds : Dict[List]
        Left and right bounds are used to fit the weathering model params. Include the calculated unpassed keys and
        values.
    n_layers : int
        Quantity of weathering model layers. Calculated from `init`, `bounds`, and `n_layers` parameters.
    params : dict
        Contains fitted parameters of a weathering model.

    Raises
    ------
    ValueError
        if any `init` values are negative.
        if any `bounds` values are negative.
        if left bound greater than right bound.
        if passed `init` and/or `bounds` keys are insufficient or excessive.
        if an union of `init` and `bounds` keys less than 2 or `n_layers` less than 1.
    """

    def __init__(self, offsets, picking_times, n_layers=None, init=None, bounds=None, **kwargs):
        init = {} if init is None else init
        bounds = {} if bounds is None else bounds

        self.offsets = offsets
        self.max_offset = offsets.max()
        self.picking_times = picking_times

        self._check_values(init, bounds)

        self.init = {**self._calc_init_by_layers(n_layers), **self._calc_init_by_bounds(bounds), **init}
        self.bounds = {**self._calc_bounds_by_init(self.init), **bounds}
        self._check_keys()
        self.n_layers = len(self.bounds) // 2
        self._valid_keys = self._get_valid_keys()

        # piecewise func variables
        self._piecewise_times = np.empty(self.n_layers + 1)
        self._piecewise_offsets = np.zeros(self.n_layers + 1)
        self._piecewise_offsets[-1] = self.max_offset

        # Fitting piecewise linear regression
        constraints = {"type": "ineq", "fun": lambda x: (np.diff(x[1:self.n_layers]) >= 0).all(out=np.array(0))}
        minimizer_kwargs = {'method': 'SLSQP', 'constraints': constraints, **kwargs}
        self._model_params = optimize.minimize(self.loss_piecewise_linear, x0=self._stack_values(self.init),
                                               bounds=self._stack_values(self.bounds), **minimizer_kwargs)
        self.params = dict(zip(self._valid_keys, self._model_params.x))

    def __call__(self, offsets):
        '''Returns predicted first breaking times using the fitted parameters of the weathering model.'''
        return np.interp(offsets, self._piecewise_offsets, self._piecewise_times)

    def __getattr__(self, key):
        return self.params[key]

    def loss_piecewise_linear(self, *args):
        '''Returns the L1 loss between true picking times and a predicted piecewise linear function.

        Method update piecewise linear attributes of a WeatheringVelocity instance and calculate L1 loss between
        true picking times stored in the `self.picking_times` and a predicted piecewise linear function. Points for
        the L1 loss calculated for the offsets corresponding with true picking times.
        predicted_times = piecewise_linear(self.offsets).

        Piecewise linear function defined by the given `args` should be list-like and have the following structure:
            args[0] : t0
            args[1:n_layers] : cross offsets points in meters.
            args[n_layers:] : velocities of each weathering model layer in km/s.
            Total lenght of args should be n_layers * 2.
        The list-like initial is due to the `scipy.optimize.minimize`.

        Parameters
        ----------
        args: tuple, list, or 1d ndarray
            Parameters for a piecewise linear function.

        Returns
        -------
        loss : float
            L1 loss between true picking times and a predicted piecewise linear function.
        '''
        args = args[0]
        self._piecewise_times[0] = args[0]
        self._piecewise_offsets[1:self.n_layers] = args[1:self.n_layers]
        for i in range(self.n_layers):
            self._piecewise_times[i + 1] = ((self._piecewise_offsets[i + 1] - self._piecewise_offsets[i]) /
                                             args[self.n_layers + i]) + self._piecewise_times[i]
        # TODO: add different loss function
        return np.abs(np.interp(self.offsets, self._piecewise_offsets, self._piecewise_times) -
                     self.picking_times).mean()

    def _get_valid_keys(self, n_layers=None):
        '''Returns a valid list with keys based on `n_layers` or `self.n_layers`.'''
        n_layers = self.n_layers if n_layers is None else n_layers
        return ['t0'] + [f'x{i+1}' for i in range(n_layers - 1)] + [f'v{i+1}' for i in range(n_layers)]

    def _stack_values(self, params_dict): # reshuffle params
        '''Returns values of the sorted input dict. Dict sorted by order stored in `_valid_keys` attribute.'''
        return np.stack([params_dict[key] for key in self._valid_keys], axis=0)

    def _fit_regressor(self, x, y, start_slope, start_time, fit_intercept):
        '''Returns parameters of a fitted linear regression.

        Parameters
        ----------
        x : 1d ndarray of shape (n_samples,)
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
            Linear regression `coef` and `intercept`
        '''
        lin_reg = SGDRegressor(loss='huber', early_stopping=True, penalty=None, shuffle=True, epsilon=0.1,
                               eta0=.05, alpha=0, tol=1e-4, fit_intercept=fit_intercept)
        lin_reg.fit(x, y, coef_init=start_slope, intercept_init=start_time)
        return lin_reg.coef_[0], lin_reg.intercept_

    def _calc_init_by_layers(self, n_layers):
        '''Returns `init` dict by a given an estimated quantity of layers.

        Method split picking times on a `n_layers` equal part by cross offsets and fit separate linear regression
        on each part. Fit a coefficient and intercept for the first part and fit a coefficient only for any next part.
        These linear functions are compiled together as an estimated piecewise linear function. Parameters of
        estimated piecewise function return as `init` dict.

        Parameters
        ----------
        n_layers : int
            Number of layers.

        Returns
        -------
        init : dict
            Estimated initial to fit the piecewise linear function.
        '''
        if n_layers is None:
            return {}

        # split cross offsets on an equal intervals
        cross_offsets = np.linspace(0, self.max_offset, num=n_layers+1)
        times = np.empty(n_layers)
        slopes = np.empty(n_layers)

        min_picking_times = self.picking_times.min()  # normalization parameter.
        start_time = 1  # base time. equal to minimum picking times with the `min_picking` normalization.
        start_slope = 2/3  # base slope corresponding velocity is 1,5 km/s (v = 1 / slope)
        for i in range(n_layers):
            mask = (self.offsets > cross_offsets[i]) & (self.offsets <= cross_offsets[i + 1])
            if mask.sum() > 1:  # at least two point to fit
                slopes[i], times[i] = self._fit_regressor(self.offsets[mask].reshape(-1, 1) / min_picking_times,
                                                          self.picking_times[mask] / min_picking_times,
                                                          start_slope, start_time, fit_intercept=(i==0))
            else:
                slopes[i], times[i] = start_slope, start_time
                warnings.warn("Not enough first break points to fit an init params. Using a base estimation.")
            start_slope = slopes[i] * (n_layers / (n_layers + 1)) # raise base velocity for next layers (v = 1 / slope)
            start_time = times[i] + (slopes[i] - start_slope) * (cross_offsets[i + 1] / min_picking_times)
        velocities = 1 / slopes
        init = np.hstack((times[0] * min_picking_times, cross_offsets[1:-1], velocities))
        init = dict(zip(self._get_valid_keys(n_layers), init))
        return init

    def _calc_init_by_bounds(self, bounds):
        '''Returns dict with a calculated init from a bounds dict.'''
        return {key: val1 + (val2 - val1) / 3 for key, (val1, val2) in bounds.items()}

    def _calc_bounds_by_init(self, init):
        '''Returns dict with calculated bounds from a init dict.'''
        return {key: [val / 2, val * 2] for key, val in init.items()}

    def _check_values(self, init, bounds):
        '''Check the values of an `init` and `bounds` dicts.'''
        negative_init = {key: val for key, val in init.items() if val < 0}
        if negative_init:
            raise ValueError(f"Init parameters contain negative values {str(negative_init)[1:-1]}")
        negative_bounds = {key: val for key, val in bounds.items() if min(val) < 0}
        if negative_bounds:
            raise ValueError(f"Bounds parameters contain negative values {str(negative_bounds)[1:-1]}")
        reversed_bounds = {key: [left, right] for key, [left, right] in bounds.items() if left > right}
        if reversed_bounds:
            raise ValueError(f"Left bound is greater than right bound for {list(reversed_bounds.keys())} key(s).")

    def _check_keys(self):
        '''Check the `self.bounds` keys for a minimum quantity, an excessive, and an insufficient.'''
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

    @plotter(figsize=(10, 5))
    def plot(self, ax, title=None, show_params=True, threshold_time=None, x_label="offset, m", y_label="time, ms"):
        '''Plot the WeatheringVelocity data, fitted curve, cross offsets, and additional information.

        Parameters
        ----------
        show_params : bool, optional, defaults to True
            Shows a t0, cross offsets, and velocities on a plot.
        threshold_time : int or float, optional. Defaults to None.
            Gap for plotting two outlines. If None additional outlines don't show.
        x_label : str, optional. Defaults to "offset, m".
            Label for the x-axis of a plot.
        y_label : str, optional. Defaults to "time, ms".
            Label for the y-axis of a plot.

        Returns
        -------
        self : WeatheringVelocity
            WeatheringVelocity without changes.
        '''
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        ax.scatter(self.offsets, self.picking_times, s=1, color='black', label='fbp points')
        ax.plot(self._piecewise_offsets, self._piecewise_times, '-', color='red', label='fitted linear function')
        for i in range(self.n_layers-1):
            ax.axvline(self._piecewise_offsets[i+1], 0, self.picking_times.max(), ls='--', c='blue',
                       label='crossover point')
        if show_params:
            params = [self.params[key] for key in self._valid_keys]
            title = f"t0 : {params[0]:.2f} ms"
            if self.n_layers > 1:
                title += '\ncrossover offsets : ' + ', '.join(f"{round(x)}" for x in params[1:self.n_layers]) + ' m'
            title += '\nvelocities : ' + ', '.join(f"{v:.2f}" for v in params[self.n_layers:]) + ' km/s'

            ax.text(0.03, .94, title, fontsize=15, va='top', transform=ax.transAxes)

        if threshold_time is not None:
            ax.plot(self._piecewise_offsets, self._piecewise_times + threshold_time, '--', color='red',
                    label=f'+/- {threshold_time}ms window')
            ax.plot(self._piecewise_offsets, self._piecewise_times - threshold_time, '--', color='red')
        ax.legend(loc='lower right')
        return self
