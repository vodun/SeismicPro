"""Implements WeatheringVelocity class to calc piecewise function and store parameters of this function."""

import numpy as np
from sklearn.linear_model import SGDRegressor
from scipy import optimize

from .decorators import plotter


class WeatheringVelocity:
    """ A class calculate and store parameters of a weathering and some subweathering layers based on gather's offsets
    and times of a first break picking.
    
    `WeatheringVelocity` object use next parameters:
        `t0`: double travel time to weathering layer's base
        `x1`: offsets where refracted wave from first subweathering layer comes at same time with reflected wave.
        `x{i}`: offset where refracted wave from i-th subweathering layer comes at same time with refracted wave from
                previous layer.
        `v1`: velocity of a weathering layer
        `v{i}`: velocity of a i-th layer. Subweathering layers start with 2 number.
    All parameters stores in `params` attribute as dict with a stated above keys.

    Class could be initialize with a `init`, `bounds`, `n_layers` or by mix of it.
    `init` should be dict with an early discussed keys and estimate values and took as based params to futher calculation.
    `bounds` should be dict with a same keys and lists with lower and upper bounds. Resulting parameters could not be out
    of a bounds.
    `n_layers` is proir information about total weathering and subwethering layers and could be usefull if you haven't
    information about `init` or `bounds`.
    
    In case when you have partial information about `init` and `bounds` you could pass part of params in an `init` dict 
    and a remaininig part in a `bounds`. Be sure that you pass all needed keys.


    Examples
    --------
    A Weathering Velocity object with starting initial parameters for two layers model
    >>> weathering_velocity = gather.calculate_weathering_velocity(init={'t0': 100, 'x1': 1500, 'v1': 2, 'v2': 3})

    A Weathering Velocity object with bounds for final parameters of a piecewise function for one layer model
    >>> weathering_velocity = gather.calculate_weathering_velocity(init={'t0': [0, 200], 'x1': [1, 3]})

    A Weathering Velocity object for three layers model.
    >>> weathering_velocity = gather.calculate_weathering_velocity(n_layers=3)

    Also mixing parameters possible
    >>> weathering_velocity = gather.calculate_weathering_velocity(init={'t0': 100, 'x1': 1500},
                                                                   bounds={'v1': [1, 3], 'v2': [1, 5]})
    Note: follow closely for keys fullness of dicts unions.

    Parameters
    ----------
    offsets : ndarray
        offsets of a traces
    max_offset : int
        maximum offsets value
    picking_times : ndarray
        picking times of a traces
    init : dict
        inital values for fitting a piecewise function. Used to calculate `bounds` and `n_layers` if these params not enough.
    bounds : Dict[List]
        left and right bounds for any parameter of a piecewise function. Used to calculate `init` and `n_layers` if these params not enough.
    n_layers : int
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
        quantity piece of a fitted piecewise funtion.
        # check naming with Dan
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

    """

    def __init__(self, offsets, picking_times, n_layers=None, init=None, bounds=None, **kwargs):

        init = {} if init is None else init
        bounds = {} if bounds is None else bounds

        self.offsets = offsets
        self.max_offset = offsets.max()
        self.picking_times = picking_times

        if n_layers is None and init is None and bounds is None:
            raise ValueError('At least one of the `n_layers`, `init` or `bounds` should be passed')

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

        self.init = {**self._calc_init_by_layers(n_layers), **self._calc_init_by_bounds(bounds), **init}
        self.bounds = {**self._calc_bounds_by_init(self.init), **bounds}
        self.n_layers = len(self.bounds) // 2

        # change naming and error message
        missing_keys = set(self._get_valid_keys()) ^ set(self.bounds.keys())
        if missing_keys:
            raise ValueError("Inconsistent parameters to fit a weathering velocity curve. ",
                            f"Check {missing_keys} key(s) or define `n_layers`")

        # piecewise func variables
        self._n_iters = 0
        self._piecewise_times = np.empty(self.n_layers + 1)
        self._piecewise_offsets = np.zeros(self.n_layers + 1)
        self._piecewise_offsets[-1] = self.max_offset

        # using `optimize.curve_fit` faster at 25% than `optimize.minimize` but could be affected to swap 'x1' and `x2`
        # and turn to wrong velocity values with a small chance. Testing on noisly linear data shows 0.1% to it 
        # on a `n_layers` > 2
        
        # Piecewise linear regression fitting
        # optimizer_kwargs  = {'method': 'trf', 'loss': 'soft_l1', **kwargs}
        # model_params, _ = optimize.curve_fit(self.piecewise_linear, offsets, picking_times,
        #                                      p0=self._stack_values(self.init), bounds=self._stack_values(self.bounds),
        #                                      **optimizer_kwargs)
        # self.params = dict(zip(self._get_valid_keys(), model_params))

        # Piecewise linear regression minimization
        constraints = {"type": "ineq", "fun": lambda x: (np.diff(x[1:self.n_layers]) >= 0).all(out=np.array(0))}
        minimizer_kwargs = {'method': 'SLSQP', 'constraints': constraints, **kwargs}
        model_params = optimize.minimize(self.piecewise_linear, x0=self._stack_values(self.init), 
                                         bounds=self._stack_values(self.bounds), **minimizer_kwargs)
        self.params = dict(zip(self._get_valid_keys(), model_params.x))

    def __call__(self, offsets):
        ''' return a predicted times using the fitted crossovers and velocities. '''
        return np.interp(offsets, self._piecewise_offsets, self._piecewise_times)

    def __getattr__(self, key):
        return self.params[key]

    def piecewise_linear(self, *args):
        '''
        args = [t0, *crossovers, *velocities]
        '''
        args = args[0]
        self._piecewise_times[0] = args[0]
        self._piecewise_offsets[1:self.n_layers] = args[1:self.n_layers]
        for i in range(self.n_layers):
            self._piecewise_times[i+1] = ((self._piecewise_offsets[i + 1] - self._piecewise_offsets[i]) /
                                           args[self.n_layers + i]) + self._piecewise_times[i]
        self._n_iters += 1
        # TODO: add different loss function
        return np.abs(np.interp(self.offsets, self._piecewise_offsets, self._piecewise_times) - 
                     self.picking_times).mean()

    def _get_valid_keys(self, n_layers=None):
        n_layers = self.n_layers if n_layers is None else n_layers
        return ['t0'] + [f'x{i+1}' for i in range(n_layers - 1)] + [f'v{i+1}' for i in range(n_layers)]

    def _stack_values(self, params_dict):
        ''' docstring '''
        return np.stack([params_dict[key] for key in self._get_valid_keys()], axis=0)

    def _fit_regressor(self, x, y, start_params, fit_intercept):
        ''' docstring '''
        lin_reg = SGDRegressor(loss='huber', early_stopping=True, penalty=None, shuffle=True, epsilon=0.01,
                               eta0=.003, alpha=0, fit_intercept=fit_intercept) 
        lin_reg.fit(x, y, coef_init=start_params[0], intercept_init=start_params[1])
        return lin_reg.coef_[0], lin_reg.intercept_

    def _calc_init_by_layers(self, n_layers):
        ''' n regressions '''
        if n_layers is None:
            return {}

        # cross offsets makes equal interval
        cross_offsets = np.linspace(0, self.max_offset, num=n_layers+1)
        times = np.empty(n_layers)
        slopes = np.empty(n_layers)
        # start_params contains slopes value and 't0'. base slope corresponds with velocity is 1.5 km/s
        start_params = [2/3, min(self.picking_times)]
        for i in range(n_layers):
            mask = (self.offsets > cross_offsets[i]) & (self.offsets <= cross_offsets[i + 1])
            slopes[i], times[i] = self._fit_regressor(self.offsets[mask].reshape(-1, 1), self.picking_times[mask],
                                                      start_params, fit_intercept=(i==0))
            start_params[0] = slopes[i] * (n_layers / (n_layers + 1))
            start_params[1] = times[i] + (slopes[i] - start_params[0]) * cross_offsets[i + 1]

        velocities = 1 / slopes

        init = np.empty(shape=2 * n_layers)
        init[0] = times[0]
        init[1:n_layers] = cross_offsets[1:-1]
        init[n_layers:] = velocities

        init = dict(zip(self._get_valid_keys(n_layers), init))
        return init

    def _calc_init_by_bounds(self, bounds):
        ''' docstring '''
        return {key: val1 + (val2 - val1) / 3 for key, (val1, val2) in bounds.items()}
    
    def _calc_bounds_by_init(self, init):
        ''' calc bounds based on init or calc init based on bounds '''
        # t0 bounds could be too narrow
        return {key: [val / 2, val * 2] for key, val in init.items()}

    @plotter(figsize=(10, 5))
    def plot(self, ax, title=None, show_params=False, threshold_times=None):
        ''' docstring '''
        # TODO: add ticks and ticklabels and labels
        ax.scatter(self.offsets, self.picking_times)
        ax.scatter(self.offsets, self(self.offsets), s=5)

        if show_params:
            crossover_title = 'crossovers offsets = '
            crossover_title += ', '.join(f"{self.params[f'x{i + 1}']:.2f}" for i in range(self.n_layers - 1))
            if self.n_layers == 1:
                crossover_title += 'None'
            velocity_title = 'velocities = '
            velocity_title += ', '.join(f"{self.params[f'v{i + 1}']:.2f}" for i in range(self.n_layers))

            ax.text(0.03, .94, f"t0={self.t0:.2f}\n{crossover_title}\n{velocity_title}", fontsize=15, va='top',
                    transform=ax.transAxes)

        if threshold_times is not None:
            ax.plot(self._piecewise_offsets, self._piecewise_times + threshold_times, '--', color='gray')
            ax.plot(self._piecewise_offsets, self._piecewise_times - threshold_times, '--', color='gray')

        return self
