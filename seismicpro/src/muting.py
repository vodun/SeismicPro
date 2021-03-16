import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression


class BaseMuting:
    def create_mask(self, trace_len, offsets, sample_rate):
        time_ixs = (self.get_time(offsets) / sample_rate).astype(np.int32) # m/samples
        return (np.arange(trace_len).reshape(1, -1) - time_ixs.reshape(-1, 1)) > 0

    def get_time(self, offsets):
        raise NotImplementedError


class PickingMuting(BaseMuting):
    def __init__(self, times, offsets, add_velocity=0):
        times = np.array(times, dtype=np.int32)
        add_velocity = add_velocity / 1000 # from m/s to m/ms
        lin_reg = LinearRegression(fit_intercept=True)
        lin_reg.fit(times.reshape(-1, 1), offsets)
        # If one wants to mute below given points, the found velocity reduces by given indent.
        self.velocity = lin_reg.coef_ - add_velocity

    def get_time(self, offsets):
        return offsets / self.velocity # m/ms


class Muting(BaseMuting):
    def __init__(self, points=None, path=None, bias=None, fill_value='extrapolate'):
        if points is None:
            if path is None:
                raise ValueError('`Points` or `path` must be given.')
            points = self.load_muting(path)
        self.times, self.offsets = points[:, 0], points[:, 1]
        self.interp_func = interp1d(self.offsets, self.times, fill_value=fill_value)

    def get_time(self, offsets):
        return self.interp_func(offsets) # m/ms

    def load_muting(self, path):
        muting = path
        # some loading process
        return muting
