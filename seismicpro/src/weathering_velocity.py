from collections import OrderedDict

import numpy as np
from scipy import optimize
import matplotlib.transforms as mtransforms

from .decorators import plotter
from .utils import to_list


class WeatheringVelocity:

    def __init__(self, offset, picking_times, t0=200, crossovers=1500, velocities=[2, 3], bounds=None):
        '''
        bounds passed as dict with next structure:
        {'t0': [0, 1000],
         'c1': [1000, 2000],
         'c2': [1500, 2500],
         'v1': [1, 3],
         'v2': [1, 4],
         'v3': [2, 5]}       
        
        '''
        # keep a base parameters when class initializate and rewrite it after fit function is done -> denied
        self.offset = offset
        self.picking = picking_times
        self.n_layers = len(velocities)

        self.bounds = self._calc_bounds(t0, crossovers, velocities) if bounds is None else self._parse_dict(bounds)

        self._fitted_args = None
        self._fit(self.offset, self.picking, start_params=self._params_to_args(t0, crossovers, velocities))

    def __call__(self, offset):
        ''' return a predicted times using the fitted crossovers and velocities. '''
        return self.piecewise_linear(offset, *self._fitted_args.values())

    def __getattr__(self, key):
        base_dict = self._fitted_args if key[0] in 'tcv' and len(key) <= 3 else self.__dict__
        try: 
            return base_dict[key]
        except:
            raise AttributeError

    def _calc_bounds(self, t0, crossovers, velocities):
        lower_bound = [0]
        upper_bound = [0 + t0 * 3]
        for item in to_list(crossovers)[:self.n_layers - 1] + to_list(velocities):
            lower_bound += [item / 2]
            upper_bound += [item * 2]
        return (lower_bound, upper_bound)

    def _parse_dict(self, bounds):
        bounds = OrderedDict(sorted(bounds.items()))
        lower_bounds = [None] * (self.n_layers * 2)
        upper_bounds = [None] * (self.n_layers * 2)
        for i, key in enumerate(bounds.keys()):
            idx = i + 1 if key[0] == 'c' else i if key[0] == 'v' else 0
            lower_bounds[idx] = bounds[key][0]
            upper_bounds[idx] = bounds[key][1]
        return (lower_bounds, upper_bounds)

    def _params_to_args(self, t0, crossovers, velocities):
        return to_list(t0) + to_list(crossovers)[:self.n_layers - 1] + to_list(velocities)

    # @staticmethod
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
    
    # another piecewise func
    # two times slower and needs args [*times, *crossovers, *velocities], times = [t0, t1, ..., tn]
    # ----------------------------
    # @staticmethod
    # def _lambda_factory(t, v):
    #     return lambda x:  x / v + t

    # @staticmethod
    # def _calc_times(t, c, v):
    #     for i in range(1, len(t)):
    #         t[i] = (v[i] - v[i-1])/(v[i] * v[i-1]) * c[i] + t[i-1]
    #     return t
    
    # def piecewise_piecewise(self, offset, *args):
    #     '''
    #     args = [*times, *crossovers, *velocities]
    #     '''
    #     times = list(args[:self.n_layers])
    #     cross_offset = [0] + list(args[self.n_layers:(2*self.n_layers-1)]) + [offset.max()]
    #     velocites = args[-self.n_layers:]
    #     times = _calc_times(times, cross_offset, velocites)

    #     condition_list = [None] * self.n_layers
    #     func_list = [None] * self.n_layers
    #     for i in range(n_layers):
    #         condition_list[i] = (cross_offset[i] < offset) & (offset <= cross_offset[i+1])
    #         func_list[i] = _lambda_factory(times[i], velocites[i])
    #     return np.piecewise(offset, condition_list, func_list)
    # ----------------------------

    def _create_keys(self):
        return ['t0'] + [f'c{i+1}' for i in range(self.n_layers - 1)][:self.n_layers - 1] + \
               [f'v{i+1}' for i in range(self.n_layers)]

    def _fit(self, offset, picking_times, start_params):
        _args, _ = optimize.curve_fit(self.piecewise_linear, offset, picking_times, p0=start_params,
                                      bounds=self.bounds, method='trf', loss='soft_l1')
        self._fitted_args = dict(zip(self._create_keys(), _args))

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
