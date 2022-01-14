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
        self.offsets_max = offsets.max()
        self.picking_times = picking_times

        init = {} if init is None else init
        bounds = {} if bounds is None else bounds

        self.init = {**self._calc_params_by_layers(n_layers), **self._calc_init(bounds), **init}
        self.bounds = {**self._calc_bounds(self.init), **bounds}
        self.n_layers = len(self.bounds) // 2

        if set(self._create_keys()) != set(self.bounds.keys()):
            raise ValueError(f"Insufficient parameters to fit a weathering velocity curve. ",
                             f"Add {set(self._create_keys()) - set(self.bounds.keys())} keys or define `n_layers`")

        # piecewise variables
        self._count = 0
        self._piecewise_times = np.empty(self.n_layers + 1)
        self._piecewise_cross_offsets = np.zeros(self.n_layers + 1)
        self._piecewise_cross_offsets[-1] = self.offsets_max
        # fitting
        fitted, _ = optimize.curve_fit(self.piecewise_linear, offsets, picking_times, p0=self._parse_params(self.init),
                                       bounds=self._parse_params(self.bounds), method='trf', loss='soft_l1', **kwargs)
        self._fitted_args = dict(zip(self._create_keys(), fitted))

    def __call__(self, offsets):
        ''' return a predicted times using the fitted crossovers and velocities. '''
        return np.interp(offsets, self._piecewise_cross_offsets, self._piecewise_times)

    def __getattr__(self, key):
        return self._fitted_args[key]

    def _create_keys(self, n_layers=None):
        n_layers = self.n_layers if n_layers is None else n_layers
        return ['t0'] + [f'c{i+1}' for i in range(n_layers - 1)] + [f'v{i+1}' for i in range(n_layers)]

    def _calc_bounds(self, init):
        ''' calc bounds based on init or calc init based on bounds '''
        # checking inital values
        for key, value in init.items():
            if value < 0:
                raise ValueError(f"Used parameters for a bounds calculation is non positive. " \
                                    f"Parameter {key} is {float(init[key]):.2f}")
        # t0 bounds could be too narrow
        return {key: [val / 2, val * 2] for key, val in init.items()}

    def _calc_init(self, bounds):
        ''' docstring '''
        return {key: val1 + (val2 - val1) / 3 for key, (val1, val2) in bounds.items()}

    def _parse_params(self, parsing_dict):
        ''' docstring '''
        return np.stack([parsing_dict[key] for key in self._create_keys()], axis=-1)

    def piecewise_linear(self, offsets, *args):
        '''
        args = [t0, *crossovers, *velocities]
        '''
        self._piecewise_times[0] = args[0]
        self._piecewise_cross_offsets[1:self.n_layers] = args[1:self.n_layers]
        for i in range(self.n_layers):
            self._piecewise_times[i+1] = ((self._piecewise_cross_offsets[i + 1] - self._piecewise_cross_offsets[i]) /
                                           args[self.n_layers + i]) + self._piecewise_times[i]
        self._count += 1
        return np.interp(offsets, self._piecewise_cross_offsets, self._piecewise_times)

    def _calc_params_by_layers(self, n_layers):
        ''' n regressions '''
        if n_layers is None:
            return {}

        cross_offsets = np.linspace(0, self.offsets_max, num=n_layers+1)
        times = np.empty(n_layers)
        slopes = np.empty(n_layers)
        start_params = [0.5, min(self.picking_times)]
        for i in range(n_layers):
            idx = np.argwhere((self.offsets >= cross_offsets[i]) & (self.offsets < cross_offsets[i +1 ]))[:, 0]
            slopes[i], times[i] = self._fit_regressor(np.take(self.offsets, idx).reshape(-1, 1),
                                                      np.take(self.picking_times, idx), start_params)
            start_params[0] = slopes[i] * (n_layers / (n_layers + 1))
            start_params[1] = times[i] + (slopes[i] - start_params[0]) * self.offsets_max * (i + 1) / n_layers

        velocities = 1 / slopes

        init = np.empty(shape=2 * n_layers)
        init[0] = times[0]
        init[1:n_layers] = cross_offsets[1:-1]
        init[n_layers:] = velocities

        init = dict(zip(self._create_keys(n_layers), init))
        return init

    def _fit_regressor(self, x, y, start_params):
        ''' docstring '''
        lin_reg = SGDRegressor(loss='huber', early_stopping=True, penalty=None, shuffle=True, epsilon=0.1,
                               eta0=.003, alpha=0)
        lin_reg.fit(x, y, coef_init=start_params[0], intercept_init=start_params[1])
        return lin_reg.coef_[0], lin_reg.intercept_

    @plotter(figsize=(10, 5))
    def plot(self, ax, title=None, show_params=False, threshold=None, **kwargs):
        ''' docstring '''
        ax.scatter(self.offsets, self.picking_times)
        ax.scatter(self.offsets, self(self.offsets), s=5)

        if show_params:
            crossover_title = 'crossovers offsets = '
            if self.n_layers > 1:
                crossovers = [f"{getattr(self, f'c{i + 1}'):.2f}" for i in range(self.n_layers - 1)]
                crossover_title += ', '.join(crossovers)
            else:
                crossover_title += 'None'
            velocity_title = 'velocities = '
            velocities = [f"{getattr(self, f'v{i + 1}'):.2f}" for i in range(self.n_layers)]
            velocity_title += ', '.join(velocities)

            ax.text(0.03, .94, f"t0={self.t0:.2f}\n{crossover_title}\n{velocity_title}", fontsize=15, va='top',
                    transform=ax.transAxes)

        if threshold is not None:
            ax.plot(self._piecewise_cross_offsets, self._piecewise_times + threshold, '--', color='gray')
            ax.plot(self._piecewise_cross_offsets, self._piecewise_times - threshold, '--', color='gray')


        return self
