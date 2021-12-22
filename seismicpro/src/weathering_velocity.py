from collections import OrderedDict

import numpy as np
from scipy import optimize
import matplotlib.transforms as mtransforms

from .decorators import plotter
from .utils import to_list


class WeatheringVelocity:

    def __init__(self, offset, picking_times, t0=200, crossovers=1500, velocities=[2, 3], bounds=None):
        # keep a base parameters when class initializate and rewrite it after fit function is done -> denied

        # L namedexpression used in pipeline call returned list.
        self.offset = offset[0].ravel() if isinstance(offset, list) else offset.ravel()
        self.picking = picking_times[0].ravel() if isinstance(picking_times, list) else picking_times.ravel()

        # params after fitting only
        # self.t0, self.cs1, self.v1, self.v2
        # self.t0 = None
        # self.crossovers = None
        # self.velocities = None

        self.n_layers = len(velocities)
        self._base_args = self._params_to_args(t0, crossovers, velocities)

        # bounds passed as dict with next structure:
        # {'t0': [0, 1000],
        #  'c1': [1000, 2000],
        #  'c2': [1500, 2500],
        #  'v1': [1, 3],
        #  'v2': [1, 4],
        #  'v3': [2, 5]}
        self.bounds = self._calc_bounds(t0, crossovers, velocities) if bounds is None else self._parse_dict(bounds)

        self._fitted_args = None
        self._fit(self.offset, self.picking)

    # @property
    # def n_layers(self):
    #     return len(self.velocities)

    def _calc_bounds(self, t0, crossovers, velocities):
        # is we need to change multipliers?
        # t0 block
        lower_bound = [0]
        upper_bound = [0 + t0 * 3]
        # crossovers block
        for offset in to_list(crossovers):
            lower_bound += [offset / 2]
            upper_bound += [offset * 2]
        # velocities block
        for v_i in to_list(velocities):
            lower_bound += [v_i / 2]
            upper_bound += [v_i * 2]
        return (lower_bound, upper_bound)

    def _parse_dict(self, bounds):
        bounds = OrderedDict(sorted(bounds.items()))
        lower_bounds = [None] * (self.n_layers * 2)
        upper_bounds = [None] * (self.n_layers * 2)
        for i, key in enumerate(bounds.keys()):
            if key[0] == 'c':
                lower_bounds[i + 1] = bounds[key][0]
                upper_bounds[i + 1] = bounds[key][1]
            if key[0] == 't':
                lower_bounds[0] = bounds[key][0]
                upper_bounds[0] = bounds[key][1]
            if key[0] == 'v':
                lower_bounds[i] = bounds[key][0]
                upper_bounds[i] = bounds[key][1]
        return (lower_bounds, upper_bounds)

    def _params_to_args(self, t0, crossovers, velocities):
        result_args = to_list(t0)
        if self.n_layers > 1:
            result_args += to_list(crossovers)
        result_args += to_list(velocities)
        return result_args

    def _args_to_params(self, *args):
        self.t0 = args[0]
        if self.n_layers > 1:
            for i, offset in enumerate(args[1:self.n_layers]):
                setattr(self, f'c{i + 1}', offset)
        else:
            setattr(self, f'c1', None)
        for i, velocity in enumerate(args[self.n_layers:]):
            setattr(self, f'v{i + 1}', velocity)

    @staticmethod
    def piecewise_linear(offset, *args):
        '''
        args = [t0, *crossovers, *velocities]
        '''
        t0 = args[0]
        crunch = list(args[1:len(args) // 2]) + [offset.max()]
        velocity = args[len(args) // 2:]
        offset_coords = [0]
        times_coords = [t0]
        for i, (v_i, offset_i) in enumerate(zip(velocity, crunch)):
            times_coords.append((offset_i - offset_coords[-1]) / v_i + times_coords[-1])
            offset_coords.append(offset_i)
        return np.interp(offset, offset_coords, times_coords)

    def _fit(self, offset, picking_times):
        _args, _ = optimize.curve_fit(self.piecewise_linear, offset, picking_times, p0=self._base_args,
                                      bounds=self.bounds, method='trf', loss='soft_l1')
        self._args_to_params(*_args)
        self._fitted_args = _args

    def __call__(self, offset):
        ''' return a predicted times using the fitted crossovers and velocities. '''
        return self.piecewise_linear(offset, *self._fitted_args)

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

            trans = mtransforms.ScaledTranslation(1 / 5, -1 / 5, scale_trans=mtransforms.Affine2D([[100, 0, 0],
                                                                                                   [0, 100, 0],
                                                                                                   [0, 0, 1]]))  # fig.dpi_scale_trans
            ax.text(0.0, 1.0, f"t0={self.t0:.2f}\n{crossover_title}\n{velocity_title}", fontsize=15, va='top',
                    transform=ax.transAxes + trans,
                    )
