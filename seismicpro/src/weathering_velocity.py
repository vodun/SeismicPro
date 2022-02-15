"""Implements WeatheringVelocity class to fit piecewise function and store parameters of a fitted function."""

import warnings

import numpy as np
from sklearn.linear_model import SGDRegressor
from scipy import optimize

from .decorators import plotter
from .utils import set_ticks


class WeatheringVelocity:
    """ A class calculate and store parameters of a weathering and some subweathering layers based on gather's offsets
    and times of a first break picking.

    `WeatheringVelocity` object use next parameters:
        `t0`: double travel time to the weathering layer's base
        `x1`: offsets where refracted wave from first subweathering layer comes at same time with reflected wave.
        `x{i}`: offset where refracted wave from i-th subweathering layer comes at same time with refracted wave from
                previous layer.
        `v1`: velocity of a weathering layer
        `v{i}`: velocity of a i-th layer. Subweathering layers start with second number.
    All parameters stores in `params` attribute as dict with a stated above keys.

    Class could be initialize with an `init`, `bounds`, `n_layers` or by mix of it.
    `init` should be dict with an before discussed keys and estimate values and took as based params. Used for 
    calculate bounds. Lower bounds is init / 2 and upper bounds is init * 2.
    `bounds` should be dict with a same keys and lists with lower and upper bounds. Resulting parameters could not be
    out of a bounds. If a some part of `init` is not defined a `bounds` used for calculate missing `init` values.
    `n_layers` is number of total weathering and subwethering layers and could be usefull if you haven't information
    about `init` or `bounds`. Used for calculate `init` and `bounds`.

    In case when you have partial information about `init` and `bounds` you could pass part of params in an `init` dict
    and a remaininig part in a `bounds`. Be sure that you pass all needed keys.

    All passed parameters have the greater priority than calculated parameters.

    Examples
    --------
    A Weathering Velocity object with starting initial parameters for two layers model
    >>> weathering_velocity = gather.calculate_weathering_velocity(init={'t0': 100, 'x1': 1500, 'v1': 2, 'v2': 3})

    A Weathering Velocity object with bounds for final parameters of a piecewise function for one layer model
    >>> weathering_velocity = gather.calculate_weathering_velocity(init={'t0': [0, 200], 'v1': [1, 3]})

    A Weathering Velocity object for three layers model.
    >>> weathering_velocity = gather.calculate_weathering_velocity(n_layers=3)

    Also mixing parameters possible
    >>> weathering_velocity = gather.calculate_weathering_velocity(init={'t0': 100, 'x1': 1500},
                                                                   bounds={'v1': [1, 3], 'v2': [1, 5]})
    Note: follow closely for keys fullness of dicts unions.

    Parameters
    ----------
    offsets : 1d ndarray
        offsets of a traces
    picking_times : 1d ndarray
        picking times of a traces
    init : dict, defaults to None
        inital values for fitting a piecewise function. Used to calculate `bounds` and `n_layers` if these params
        not enough.
    bounds : Dict[List], defaults to None
        left and right bounds for any parameter of a piecewise function. Used to calculate `init` and `n_layers`
        if these params not enough.
    n_layers : int, defualts to None
        prior quantity of a layers of a weathering model. Interpreting like a quantity piece of a piecewise funtion.
        Used for calculate `init` and `bounds` if these params not enough.

    Attributes
    ----------
    offsets : ndarray
        contains offsets of a traces.
    max_offset : int
        maximum offsets value.
    picking_times : ndarray
        picking times of a traces.
    init : dict
        inital values used for fitting a piecewise function.
    bounds : Dict[List]
        left and right bounds used for fitting a piecewise function.
    n_layers : int
        quantity piece of a fitted piecewise function.
    params : dict
        contains fitted values of a piecewise function.
    _n_iters : int
        quantity of iteration for fitting a piecewise function.

    Raises
    ------
    ValueError
        if any `init` values is negative
        if any `bounds` values is negative
        if left bound greater than right bound
        if passed `init` and/or `bounds` keys are insufficient or excess
        if n

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

        # piecewise func variables
        self._piecewise_times = np.empty(self.n_layers + 1)
        self._piecewise_offsets = np.zeros(self.n_layers + 1)
        self._piecewise_offsets[-1] = self.max_offset

        # Piecewise linear regression minimization
        constraints = {"type": "ineq", "fun": lambda x: (np.diff(x[1:self.n_layers]) >= 0).all(out=np.array(0))}
        minimizer_kwargs = {'method': 'SLSQP', 'constraints': constraints, **kwargs}
        model_params = optimize.minimize(self.loss_piecewise_linear, x0=self._stack_values(self.init),
                                         bounds=self._stack_values(self.bounds), **minimizer_kwargs)
        self._model_params = model_params
        self.params = dict(zip(self._get_valid_keys(), model_params.x))

    def __call__(self, offsets):
        ''' Return a predicted times using the fitted crossovers and velocities. '''
        return np.interp(offsets, self._piecewise_offsets, self._piecewise_times)

    def __getattr__(self, key):
        return self.params[key]

    def loss_piecewise_linear(self, *args):
        '''
        args = [t0, *crossovers, *velocities]
        '''
        args = args[0]
        self._piecewise_times[0] = args[0]
        self._piecewise_offsets[1:self.n_layers] = args[1:self.n_layers]
        for i in range(self.n_layers):
            self._piecewise_times[i+1] = ((self._piecewise_offsets[i + 1] - self._piecewise_offsets[i]) /
                                           args[self.n_layers + i]) + self._piecewise_times[i]
        # TODO: add different loss function
        return np.abs(np.interp(self.offsets, self._piecewise_offsets, self._piecewise_times) -
                     self.picking_times).mean()

    def _get_valid_keys(self, n_layers=None):
        '''Return valid list with keys based on `n_layers` or `self.n_layers`.'''
        n_layers = self.n_layers if n_layers is None else n_layers
        return ['t0'] + [f'x{i+1}' for i in range(n_layers - 1)] + [f'v{i+1}' for i in range(n_layers)]

    def _stack_values(self, params_dict): # reshuffle params
        ''' docstring '''
        return np.stack([params_dict[key] for key in self._get_valid_keys()], axis=0)

    def _fit_regressor(self, x, y, start_slope, start_time, fit_intercept):
        ''' docstring '''
        lin_reg = SGDRegressor(loss='huber', early_stopping=True, penalty=None, shuffle=True, epsilon=0.01,
                               eta0=.03, alpha=0, tol=1e-5, fit_intercept=fit_intercept)
        lin_reg.fit(x, y, coef_init=start_slope, intercept_init=start_time)
        return lin_reg.coef_[0], lin_reg.intercept_

    def _calc_init_by_layers(self, n_layers):
        ''' n regressions '''
        if n_layers is None:
            return {}

        # default cross offsets makes by an equal intervals
        cross_offsets = np.linspace(0, self.max_offset, num=n_layers+1)
        times = np.empty(n_layers)
        slopes = np.empty(n_layers)

        min_picking = self.picking_times.min()  # normalization parameter
        start_time = 1  # base time, same as minimum picking times with `min_picking` normalization
        start_slope = 2/3  # base slope, corresponding velocity is 1,5 km/s ( v = 1 / slope )
        for i in range(n_layers):
            mask = (self.offsets > cross_offsets[i]) & (self.offsets <= cross_offsets[i + 1])
            if mask.sum() > 1:  # at least 2 point to fit
                slopes[i], times[i] = self._fit_regressor(self.offsets[mask].reshape(-1, 1) / min_picking,
                                                        self.picking_times[mask] / min_picking,
                                                        start_slope, start_time, fit_intercept=(i==0))
            else:
                slopes[i], times[i] = start_slope, start_time
                warnings.warn("Not enough first break points to fit a init params. Using a base estimation.")
            start_slope = slopes[i] * (n_layers / (n_layers + 1))
            start_time = times[i] + (slopes[i] - start_slope) * (cross_offsets[i + 1] / min_picking)
        velocities = 1 / slopes

        init = np.hstack((times[0] * min_picking, cross_offsets[1:-1], velocities))
        init = dict(zip(self._get_valid_keys(n_layers), init))
        return init

    def _calc_init_by_bounds(self, bounds):
        '''Return dict with a calculated init from a bounds dict.'''
        return {key: val1 + (val2 - val1) / 3 for key, (val1, val2) in bounds.items()}

    def _calc_bounds_by_init(self, init):
        '''Return dict with a calcualated bounds from a init dict.'''
        # t0 bounds could be too narrow
        return {key: [val / 2, val * 2] for key, val in init.items()}

    def _check_values(self, init, bounds):
        '''Checking values of an input dicts'''
        negative_init = {key: val for key, val in init.items() if val < 0}
        if negative_init:
            raise ValueError(f"Init parameters {list(negative_init.keys())} contain ",
                             f"negative values {list(negative_init.values())}")
        negative_bounds = {key: val for key, val in bounds.items() if min(val) < 0}
        if negative_bounds:
            raise ValueError(f"Bounds parameters {list(negative_bounds.keys())} contain ",
                             f"negative values {list(negative_bounds.values())}")
        reversed_bounds = {key: [left, right] for key, [left, right] in bounds.items() if left > right}
        if reversed_bounds:
            raise ValueError(f"Left bound is greater than right bound for {list(reversed_bounds.keys())} key(s).")

    def _check_keys(self):
        '''Checking keys of `self.bounds` for excessive and insufficient and `n_layers` for possitive.'''
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
    def plot(self, ax, title=None, show_params=True, threshold_time=None):
        ''' Plot input data, fitted curve, crossoffset and some additional information.

        Parameters
        ----------
        show_params : bool, optional, defaults to True
            shows a t0, cross offsets and velocities on a plot.
        threshold_times : int or float, optional. Defaults to None.
            gap for plotting two outlines. If None additional outlines doesn't show.

        Returns
        -------
        self : WeatheringVelocity
            WeatheringVelocity without changes.

        '''
        # TODO: add ticks and ticklabels and labels
        set_ticks(ax, 'x', axis_label='x', tick_labels=['offset, m'])
        set_ticks(ax, 'y', axis_label='y', tick_labels=['time, ms'])

        ax.scatter(self.offsets, self.picking_times, s=1, color='black', label='fbp points')
        ax.plot(self._piecewise_offsets, self._piecewise_times, '-', color='red', label='fitted linear function')
        for i in range(self.n_layers-1):
            ax.axvline(self._piecewise_offsets[i+1], 0, self.picking_times.max(), ls='--', c='blue',
                       label='crossover point')
        if show_params:
            params = [self.params[key] for key in self._get_valid_keys()]
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
