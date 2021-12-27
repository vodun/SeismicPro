from collections import OrderedDict

import numpy as np
from scipy import optimize
import matplotlib.transforms as mtransforms

from .decorators import plotter
from .utils import to_list


class WeatheringVelocity:

    def __init__(self, offset, picking_times, n_layers=None, initial=None, bounds=None):
        #t0=200, crossovers=1500, velocities=[2, 3],
        '''
        bounds passed as dict with next structure:
        {'t0': [0, 1000],
         'c1': [1000, 2000],
         'c2': [1500, 2500],
         'v1': [1, 3],
         'v2': [1, 4],
         'v3': [2, 5]}       
        initial passed as dict with next structure:
        {'t0': 200,
         'c1': 1000,
         'c2': 2000,
         'v1': 1,
         'v2': 2,
         'v3': 3}  
        
        '''
        # keep a base parameters when class initializate and rewrite it after fit function is done -> denied
        print(f'n_layers={n_layers}, initial={initial}, bounds={bounds}')
        if n_layers is None and initial is None and bounds is None:
            raise ValueError('One of the `n_layers`, `initial`, `bounds` should be passed')

        self.offset = offset
        self.picking = picking_times

        self.n_layers = n_layers
        self.initial = initial
        self.bounds = bounds

        if initial is not None:
            self.n_layers = len(initial) // 2
            self.bounds = self._calc_params(initial) if bounds is None else bounds

        elif bounds is not None:
            self.n_layers = len(bounds) // 2
            self.initial = self._calc_params(bounds)

        else:
            try:
                self.initial, self.bounds = self._precalc_by_layers(n_layers)
            except:
                raise NotImplementedError('Use initial or bounds')

        # fitting
        _args, _ = optimize.curve_fit(self.piecewise_linear, offset, picking_times, p0=self._parse_params(self.initial),
                                      bounds=self._parse_params(self.bounds), method='trf', loss='soft_l1')
        self._fitted_args = dict(zip(self._create_keys(), _args))

    def __call__(self, offset):
        ''' return a predicted times using the fitted crossovers and velocities. '''
        return self.piecewise_linear(offset, *self._fitted_args.values())

    def __getattr__(self, key):
        # print('__getattr__ keys: ', key)
        # base_dict = self._fitted_args if key[0] in 'tcv' and len(key) <= 3 else self.__dict__
        # try:
        #     return base_dict[key]
        # except:
        #     raise AttributeError
        return self._fitted_args[key]

    def _calc_params(self, params: dict):
        ''' calc bounds based on initial or calc initial based on bounds '''
        result = {}
        for key, value in params.items():
            if len(to_list(value)) == 1:
                if key[0] == 't':
                    result[key] = [0, value * 3]
                else:
                    result[key] = [value / 2, value * 2]
            else:
                result[key] = value[0] + (value[1] - value[0]) / 3
                # another variant
                # result[key] = (value[0] + value[1]) / 2
        return result

    def _parse_params(self, initial=None, bounds=None):
        # from dict to list ot tuple of two list
        work_dict = initial if bounds is None else bounds
        work_dict.copy()
        t0 = work_dict.pop('t0')
        data = np.full((self.n_layers * 2, len(to_list(t0))), None)
        data[0] = t0
        
        work_dict = OrderedDict(sorted(work_dict.items()))
        for idx, value in enumerate(work_dict.values()):
            data[idx + 1] = value
        if len(to_list(t0)) == 1:
            return data.ravel()
        return (data[:, 0], data[:, 1])

    def piecewise_linear(self, offset, *args):
        '''
        args = [t0, *crossovers, *velocities]
        '''
        t0 = args[0]
        cross_offset = [0] + list(args[1:self.n_layers]) + [offset.max()]
        velocites = args[self.n_layers:]
        times = [t0] + [0] * self.n_layers
        for i in range(1, self.n_layers + 1):
            times[i] = (cross_offset[i] - cross_offset[i-1]) / velocites[i-1] + times[i-1]
        return np.interp(offset, cross_offset, times)

    def _create_keys(self):
        return ['t0'] + [f'c{i+1}' for i in range(self.n_layers - 1)][:self.n_layers - 1] + \
               [f'v{i+1}' for i in range(self.n_layers)]

    @plotter(figsize=(10, 5))
    def plot(self, ax, title=None, show_params=False, **kwargs):
        # TODO: add thresholds lines
        ax.scatter(self.offset, self.picking)
        ax.scatter(self.offset, self(self.offset), s=5)

        if show_params:
            crossover_title = 'crossovers offset = '
            if self.n_layers > 1:
                crossovers = ['{:.2f}'.format(getattr(self, f'c{i + 1}', -1)) for i in range(self.n_layers - 1)]
                crossover_title += ', '.join(crossovers)
            else:
                crossover_title += 'None'
            velocity_title = 'velocities = '
            velocities = ['{:.2f}'.format(getattr(self, f'v{i + 1}', -1)) for i in range(self.n_layers)]
            velocity_title += ', '.join(velocities)

            # transform need to move text from edge
            trans = mtransforms.ScaledTranslation(1 / 5, -1 / 5, scale_trans=mtransforms.Affine2D([[100, 0, 0],
                                                                                                   [0, 100, 0],
                                                                                                   [0, 0, 1]]))
            ax.text(0.0, 1.0, f"t0={self.t0:.2f}\n{crossover_title}\n{velocity_title}", fontsize=15, va='top',
                    transform=ax.transAxes + trans,
                    )
