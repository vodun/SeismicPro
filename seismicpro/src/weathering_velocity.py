import numpy as np
from scipy import optimize
import matplotlib.transforms as mtransforms

from .decorators import plotter
from .utils import to_list

class WeatheringVelocity:
    
    def __init__(self, gather, offset=None, picking_times_col='PredictedBreak', t0=200, offset_breakpoints=2000, velocity_layers=[2, 3], bounds=None):
        # keep a base parameters when class initializate and rewrite it after fit function is done

        # offset_breakpoints -> crossover_offsets

        # offset and picking times not from gather
        self.gather = gather
        self.offset = offset or gather['offset'].ravel()
        self.picking = gather[picking_times_col].ravel()

        self.t0 = None
        self.offset_breakpoints = None
        self.velocity_layers = None
        self._args = self._params_to_args(t0, offset_breakpoints, velocity_layers)
        # self._args = [t0] + to_list(offset_breakpoints) + to_list(velocity_layers)
        self.n_layers = len(velocity_layers)

        self.bounds = bounds or ([0, 1000, 1, 1], [1000, 3000, 3, 5])
        # self.bounds = bound or {'t0': [0, 1000], 
        #                         'offset_breakpoints': [0, 3000],
        #                         'Vp1': [1, 3],
        #                         'Vp2': [1, 5]}
        self.predict = None
        self._fit_predict(self.offset, self.picking)
    
    # @property
    # def n_layers(self):
        # return len(velocity_layers)


    def _params_to_args(self, t0, crossovers, velocities):
    #     # (t0, offsetoffset_breakpoints_crunch, velocity_layers)
        result_args = to_list(t0)
        if crossovers is not None:
            result_args += to_list(crossovers)
        result_args += to_list(velocities)
        return result_args
        
    def _args_to_params(self, *args):
        self.t0 = args[0]
        self.offset_breakpoints = args[1:self.n_layers]
        self.velocity_layers = args[self.n_layers:]

    @staticmethod
    def piecewise_linear(offset, *args):
        '''
        args = [t0, *offset_breakpoints, *velocity_layers]
        '''
        t0 = args[0]
        crunch = list(args[1:len(args)//2]) + [offset.max()]
        velocity = args[len(args)//2:]
        offset_coords = [0]
        times_coords = [t0]
        for i, (v_i, offset_i) in enumerate(zip(velocity, crunch)):
            times_coords.append((offset_i - offset_coords[-1]) / v_i + times_coords[-1])
            offset_coords.append(offset_i)
        return np.interp(offset, offset_coords, times_coords)

    def _fit_predict(self, offset, picking_times):
        # offset = offset.ravel()
        # picking_times = picking_times.ravel()
        print(offset.shape, picking_times.shape)
        print(self._args, self.bounds)
        _args, _ = optimize.curve_fit(self.piecewise_linear, offset, picking_times,
                            p0 = self._args,
                            bounds=self.bounds, 
                            method='trf', 
                            loss='soft_l1'
                            )
        self.predict = self.piecewise_linear(offset, *_args)  # from calc_metrics
        self._args_to_params(*_args)

    @plotter(figsize=(10, 5))
    def plot(self, ax, title=None, show_params=False, **kwargs):
        # show_params=True don't work yet
        ax.scatter(self.offset, self.picking)
        ax.scatter(self.offset, self.predict, s=5)
        # print(self.t0, self.offset_breakpoints, self.velocity_layers)
        if show_params:
            velocity_title = 'velocities = '
            for i in range(len(self.velocity_layers)):
                if i > 0:
                    velocity_title += ', '
                velocity_title += f"{self.velocity_layers[i]:.2f}"
            breakpoint_title = 'breakpoints = '
            for i in range(len(self.offset_breakpoints)):
                if i > 0:
                    breakpoint_title += ', '
                breakpoint_title += f"{self.offset_breakpoints[i]:.2f}"
            if len(self.offset_breakpoints) == 0:
                breakpoint_title += 'None'
                
            trans = mtransforms.ScaledTranslation(1/5, -1/5, scale_trans=mtransforms.Affine2D([[100, 0, 0],
                                                                                               [0, 100, 0],
                                                                                               [0, 0, 1]]))  # fig.dpi_scale_trans
            ax.text(0.0, 1.0, f"t0={self.t0:.2f}\n{breakpoint_title}\n{velocity_title}", fontsize=15, va='top',
                    transform=ax.transAxes + trans,
                    )

