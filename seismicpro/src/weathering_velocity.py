import numpy as np
from sklearn.linear_model import SGDRegressor
from scipy import optimize

from .decorators import plotter


class WeatheringVelocity:
    ''' TODO: docstring '''
    def __init__(self, offsets, picking_times, n_layers=None, init=None, bounds=None, **kwargs):
        '''
        bounds passed as dict with next structure:
        {'t0': [0, 1000],
         'c1': [1000, 2000],
         'c2': [1500, 2500],
         'v1': [1, 3],
         'v2': [1, 4],
         'v3': [2, 5]}
        init passed as dict with next structure:
        {'t0': 200,
         'c1': 1000,
         'c2': 2000,
         'v1': 1,
         'v2': 2,
         'v3': 3}
        '''

        if n_layers is None and init is None and bounds is None:
            raise ValueError('One of the `n_layers`, `init`, `bounds` should be passed')

        self.offsets = offsets
        self.max_offset = offsets.max()
        self.picking_times = picking_times

        init = {} if init is None else init
        bounds = {} if bounds is None else bounds

        negative_init = {key: val for key, val in init.items() if val < 0}
        if negative_init:
            raise ValueError(f"Init parameters {list(negative_init.keys())} contains ", 
                             f"non positive values {list(negative_init.values())}")
        negative_bounds = {key: [left, right] for key, (left, right) in bounds.items() if any((left < 0, right < 0))}
        if negative_bounds:
            raise ValueError(f"Bounds parameters {list(negative_bounds.keys())} contains ",
                             f"non positive values. Parameters {list(negative_bounds.values())}")
        revert_bounds = {key: [left, right] for key, [left, right] in bounds.items() if left > right}
        if revert_bounds:
            raise ValueError(f"Left bound is greater than right bound for {list(revert_bounds.keys())} key(s).")

        self.init = {**self._calc_init_by_layers(n_layers), **self._calc_init_by_bounds(bounds), **init}
        self.bounds = {**self._calc_bounds_by_init(self.init), **bounds}
        self.n_layers = len(self.bounds) // 2

        missing_keys = set(self._get_valid_keys()) ^ set(self.bounds.keys())
        if missing_keys:
            raise ValueError("Inconsistent parameters to fit a weathering velocity curve. ",
                            f"Check {missing_keys} key(s) or define `n_layers`")

        # piecewise func variables
        self._n_iters = 0
        self._piecewise_times = np.empty(self.n_layers + 1)
        self._piecewise_offsets = np.zeros(self.n_layers + 1)
        self._piecewise_offsets[-1] = self.max_offset
        # fitting
        kwargs = {'method': 'trf', 'loss': 'soft_l1', **kwargs}
        fitted, _ = optimize.curve_fit(self.piecewise_linear, offsets, picking_times, p0=self._stack_values(self.init),
                                       bounds=self._stack_values(self.bounds), **kwargs)
        self.params = dict(zip(self._get_valid_keys(), fitted))

    def __call__(self, offsets):
        ''' return a predicted times using the fitted crossovers and velocities. '''
        return np.interp(offsets, self._piecewise_offsets, self._piecewise_times)

    def __getattr__(self, key):
        return self.params[key]

    def _get_valid_keys(self, n_layers=None):
        n_layers = self.n_layers if n_layers is None else n_layers
        return ['t0'] + [f'c{i+1}' for i in range(n_layers - 1)] + [f'v{i+1}' for i in range(n_layers)]

    def _stack_values(self, params_dict):
        ''' docstring '''
        return np.stack([params_dict[key] for key in self._get_valid_keys()], axis=-1)


    def _calc_bounds_by_init(self, init):
        ''' calc bounds based on init or calc init based on bounds '''
        # t0 bounds could be too narrow
        return {key: [val / 2, val * 2] for key, val in init.items()}

    def _calc_init_by_bounds(self, bounds):
        ''' docstring '''
        return {key: val1 + (val2 - val1) / 3 for key, (val1, val2) in bounds.items()}

    def piecewise_linear(self, offsets, *args):
        '''
        args = [t0, *crossovers, *velocities]
        '''
        self._piecewise_times[0] = args[0]
        self._piecewise_offsets[1:self.n_layers] = args[1:self.n_layers]
        for i in range(self.n_layers):
            self._piecewise_times[i+1] = ((self._piecewise_offsets[i + 1] - self._piecewise_offsets[i]) /
                                           args[self.n_layers + i]) + self._piecewise_times[i]
        self._n_iters += 1
        return np.interp(offsets, self._piecewise_offsets, self._piecewise_times)

    def _calc_init_by_layers(self, n_layers):
        ''' n regressions '''
        if n_layers is None:
            return {}

        cross_offsets = np.linspace(0, self.max_offset, num=n_layers+1)
        times = np.empty(n_layers)
        slopes = np.empty(n_layers)
        start_params = [2/3, min(self.picking_times)]
        for i in range(n_layers):
            mask = (self.offsets >= cross_offsets[i]) & (self.offsets < cross_offsets[i + 1])
            slopes[i], times[i] = self._fit_regressor(self.offsets[mask].reshape(-1, 1), self.picking_times[mask], 
                                                      start_params, fit_intercept=(i==0))
            start_params[0] = slopes[i] * (n_layers / (n_layers + 1))
            start_params[1] = times[i] + (slopes[i] - start_params[0]) * self.max_offset * (i + 1) / n_layers

        velocities = 1 / slopes

        init = np.empty(shape=2 * n_layers)
        init[0] = times[0]
        init[1:n_layers] = cross_offsets[1:-1]
        init[n_layers:] = velocities

        init = dict(zip(self._get_valid_keys(n_layers), init))
        return init

    def _fit_regressor(self, x, y, start_params, fit_intercept):
        ''' docstring '''
        lin_reg = SGDRegressor(loss='huber', early_stopping=True, penalty=None, shuffle=True, epsilon=0.01,
                               eta0=.003, alpha=0, fit_intercept=fit_intercept) 
        lin_reg.fit(x, y, coef_init=start_params[0], intercept_init=start_params[1])
        return lin_reg.coef_[0], lin_reg.intercept_

    @plotter(figsize=(10, 5))
    def plot(self, ax, title=None, show_params=False, threshold_times=None, **kwargs):
        ''' docstring '''
        ax.scatter(self.offsets, self.picking_times)
        ax.scatter(self.offsets, self(self.offsets), s=5)

        if show_params:
            crossover_title = 'crossovers offsets = '
            crossover_title += ', '.join(f"{self.params[f'c{i + 1}']:.2f}" for i in range(self.n_layers - 1))
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
