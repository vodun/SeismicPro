from collections import OrderedDict

import matplotlib.transforms as mtransforms
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import optimize

from .decorators import plotter
from .utils import to_list
from .utils.interpolation import interp1d


class WeatheringVelocity:

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
        # print(f'n_layers={n_layers}, init={init}, bounds={bounds}')
        if n_layers is None and init is None and bounds is None:
            raise ValueError('One of the `n_layers`, `init`, `bounds` should be passed')

        self.offsets = offsets
        self.picking = picking_times

        self.n_layers = n_layers
        self.init = init
        self.bounds = bounds

        if init is not None:
            self.n_layers = len(init) // 2
            self._check_keys(init)
            self.bounds = self._calc_bounds(init) if bounds is None else bounds

        elif bounds is not None:
            self.n_layers = len(bounds) // 2
            self._check_keys(bounds)
            self.init = self._calc_init(bounds)

        else:
            self.init, self.bounds = self._calc_params_by_layers(n_layers)

        # fitting
        fitted, _ = optimize.curve_fit(self.piecewise_linear, offsets, picking_times, p0=self._parse_params(self.init),
                                       bounds=self._parse_params(self.bounds), method='trf', loss='soft_l1', **kwargs)
        self._fitted_args = dict(zip(self._create_keys(), fitted))

    def __call__(self, offsets):
        ''' return a predicted times using the fitted crossovers and velocities. '''
        return self.piecewise_linear(offsets, *self._parse_params(self._fitted_args))

    def __getattr__(self, key):
        return self._fitted_args[key]

    def _calc_bounds(self, init):
        ''' calc bounds based on init or calc init based on bounds '''
        result = {}
        for key, value in init.items():
            if key[0] == 't':
                result[key] = [0, value * 3]
            else:
                result[key] = [value / 2, value * 2]
        return result

    def _calc_init(self, bounds):
        result = {}
        for key, value in bounds.items():
            result[key] = value[0] + (value[1] - value[0]) / 3
        return result

    def _parse_params(self, work_dict):
        # from dict to list or tuple of two list
        work_dict = work_dict.copy()
        t0 = work_dict.pop('t0')
        data = np.full((self.n_layers * 2, len(to_list(t0))), None)
        data[0] = t0

        work_dict = OrderedDict(sorted(work_dict.items()))
        for idx, value in enumerate(work_dict.values()):
            data[idx + 1] = value
        if len(to_list(t0)) == 1:
            return data.ravel()
        return (data[:, 0], data[:, 1])

    def piecewise_linear(self, offsets, *args):
        '''
        args = [t0, *crossovers, *velocities]
        '''
        t0 = args[0]
        cross_offsets = [0] + list(args[1:self.n_layers]) + [offsets.max()]
        velocites = args[self.n_layers:]
        times = [t0] + [0] * self.n_layers
        for i in range(1, self.n_layers + 1):
            times[i] = (cross_offsets[i] - cross_offsets[i-1]) / velocites[i-1] + times[i-1]
        return np.interp(offsets, cross_offsets, times)

    def _create_keys(self):
        return ['t0'] + [f'c{i+1}' for i in range(self.n_layers - 1)] + [f'v{i+1}' for i in range(self.n_layers)]

    def _check_keys(self, work_dict):
        expected_keys = set(self._create_keys())
        given_keys = set(work_dict.keys())
        if expected_keys != given_keys:
            raise KeyError('Given dict with parameters contains unexpected keys.')

    def _calc_params_by_layers(self, n_layers):
        ''' use _precal_params is 1.5 times slower than put init'''
        lin_reg = LinearRegression().fit(np.atleast_2d(self.offsets).T, self.picking)
        base_v = 1 / lin_reg.coef_

        init = np.empty(shape=2 * n_layers)
        init[0] = lin_reg.intercept_ / 2
        init[1:n_layers] = np.linspace(0, self.offsets.max(), num=n_layers+1)[1:-1]
        init[n_layers:] = np.linspace(base_v / 1.5, base_v * 1.33, num=n_layers).ravel()
        init = dict(zip(self._create_keys(), init))
        return init, self._calc_bounds(init)

    @plotter(figsize=(10, 5))
    def plot(self, ax, title=None, show_params=False, **kwargs):
        # TODO: add thresholds lines
        ax.scatter(self.offsets, self.picking)
        ax.scatter(self.offsets, self(self.offsets), s=5)

        if show_params:
            crossover_title = 'crossovers offsets = '
            if self.n_layers > 1:
                crossovers = ['{:.2f}'.format(getattr(self, f'c{i + 1}')) for i in range(self.n_layers - 1)]
                crossover_title += ', '.join(crossovers)
            else:
                crossover_title += 'None'
            velocity_title = 'velocities = '
            velocities = ['{:.2f}'.format(getattr(self, f'v{i + 1}')) for i in range(self.n_layers)]
            velocity_title += ', '.join(velocities)

            # transform need to move text from edges
            trans = mtransforms.ScaledTranslation(1 / 5, -1 / 5, scale_trans=mtransforms.Affine2D([[100, 0, 0],
                                                                                                   [0, 100, 0],
                                                                                                   [0, 0, 1]]))
            ax.text(0.0, 1.0, f"t0={self.t0:.2f}\n{crossover_title}\n{velocity_title}", fontsize=15, va='top',
                    transform=ax.transAxes + trans,
                    )
        return self
