import numpy as np
from scipy import optimize

class WeatheringVelocity:
    
    def __init__(self, offset, picking_times, t0=200, offset_crunch=2000, velocity_layers=[2, 3], bounds=None):
        # keep a base parameters when class initializate and rewrite it after fit function is done
        self.offset = offset
        self.picking = np.array(picking_times).ravel() if isinstance(picking_times, list) else picking_times
        self.t0 = t0
        self.offset_crunch = offset_crunch
        self.velocity_layers = velocity_layers
        self._args = self._params_to_args(t0, offset_crunch, velocity_layers)

        self.bounds = bounds or ([0, 1000, 1, 1], [1000, 3000, 3, 5])
        # self.bounds = bound or {'t0': [0, 3000], 
        #                         'offset_crunch': [0, 5000],
        #                         'Vp1': [1, 4],
        #                         'Vp2': [1, 6]}
        self.predict = None
        self._fit_predict(offset[0], picking_times[0])
    
    @property
    def n_layers(self):
        return len(self.velocity_layers)
    
    def _params_to_args(self, t0, offset_crunch, velocity_layers):
        # (t0, offset_crunch, velocity_layers)
        result_args = []
        result_args.append(t0)
        if offset_crunch is not None:
            if isinstance(offset_crunch, list):
                result_args.extend(offset_crunch)
            else:
                result_args.append(offset_crunch)
    
        if isinstance(velocity_layers, list):
            result_args.extend(velocity_layers)
        else:
            result_args.append(velocity_layers)
        return result_args
        
    def _agrs_to_params(self, *args):
        self.t0 = args[0]
        self.offset_crunch = args[1:self.n_layers + 1]
        self.velocity_layers = args[self.n_layers + 1:]

    @staticmethod
    def piecewise_linear(offset, *args):
        '''
        args = [t0, offset_crunch, velocity_layers]
        '''
        # print('args_in', args)
        t0 = args[0]
        crunch = list(args[1:len(args)//2]) + [offset.max()]
        # print('debug', args[1:len(args)//2])
        velocity = args[len(args)//2:]
        offset_coords = [0]
        times_coords = [t0]
        # print('velocity and crunch', velocity, crunch)
        for i, (v_i, offset_i) in enumerate(zip(velocity, crunch)):
            times_coords.append((offset_i - offset_coords[-1]) / v_i + times_coords[-1])
            offset_coords.append(offset_i)
        # print('offset_coords', offset_coords)
        # print('times_coords', times_coords)
        return np.interp(offset, offset_coords, times_coords)

    def _fit_predict(self, offset, picking_times, **kwargs):
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
