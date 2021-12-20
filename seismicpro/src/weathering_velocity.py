import numpy as np
from scipy import optimize

class WeatheringVelocity:
    
    def __init__(self, offset, picking_times, t0=200, offset_breakpoints=2000, velocity_layers=[2, 3], bounds=None):
        # keep a base parameters when class initializate and rewrite it after fit function is done
        self.offset = offset
        self.picking = np.array(picking_times).ravel() if isinstance(picking_times, list) else picking_times
        self.t0 = None
        self.offset_breakpoints = None
        self.velocity_layers = None
        self._args = self._params_to_args(t0, offset_breakpoints, velocity_layers)
        self.n_layers = 2 if self.velocity_layers is None else len(self.velocity_layers)

        self.bounds = bounds or ([0, 1000, 1, 1], [1000, 3000, 3, 5])
        # self.bounds = bound or {'t0': [0, 1000], 
        #                         'offset_breakpoints': [0, 3000],
        #                         'Vp1': [1, 3],
        #                         'Vp2': [1, 5]}
        self.predict = None
        self._fit_predict(offset[0], picking_times[0])
    
    def _params_to_args(self, t0, offset_breakpoints, velocity_layers):
        # (t0, offsetoffset_breakpoints_crunch, velocity_layers)
        result_args = []
        result_args.append(t0)
        if offset_breakpoints is not None:
            if isinstance(offset_breakpoints, list):
                result_args.extend(offset_breakpoints)
            else:
                result_args.append(offset_breakpoints)
    
        if isinstance(velocity_layers, list):
            result_args.extend(velocity_layers)
        else:
            result_args.append(velocity_layers)
        return result_args
        
    def _agrs_to_params(self, *args):
        self.t0 = args[0]
        self.offset_breakpoints = args[1:self.n_layers + 1]
        self.velocity_layers = args[self.n_layers + 1:]

    @staticmethod
    def piecewise_linear(offset, *args):
        '''
        args = [t0, offset_breakpoints, velocity_layers]
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
        offset = offset.ravel()
        picking_times = picking_times.ravel()
        _args, _ = optimize.curve_fit(self.piecewise_linear, offset, picking_times,
                            p0 = self._args,
                            bounds=self.bounds, 
                            method='trf', 
                            loss='soft_l1'
                            )
        self.predict = self.piecewise_linear(offset, *_args)
        self._agrs_to_params(_args)

    def plot(self, ax):
        ax.scatter(self.offset, self.picking)
        ax.scatter(self.offset, self.predict, s=5)
        velocity_title = 'velocities = '
        for i in range(len(self.velocity_layers)):
            velocity_title += f"{self.velocity_layers[i]}:.2f"
        crunch_title = 'breakpoints = '
        for i in range(len(self.velocity_layers)):
            crunch_title += f"{self.offset_breakpoints[i]}:.2f"
        ax.set_title(f"t0={self.t0}:.2f, {crunch_title}, {velocity_title}")
