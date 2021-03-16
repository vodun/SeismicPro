import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression


class BaseMuting:
    def create_mask(self, trace_len, offsets, sample_rate):
        time_ixs = (self.get_time(offsets) / sample_rate).astype(np.int32) # from ms to samples
        time_ixs = np.clip(time_ixs, 0, trace_len)
        return (np.arange(trace_len).reshape(1, -1) - time_ixs.reshape(-1, 1)) > 0

    def get_time(self, offsets, trace_len):
        raise NotImplementedError


class PickingMuting(BaseMuting):
    def __init__(self, times, offsets, add_velocity=0):
        times = np.array(times, dtype=np.int32)
        add_velocity = add_velocity / 1000 # from m/s to m/ms
        lin_reg = LinearRegression(fit_intercept=True)
        lin_reg.fit(times.reshape(-1, 1), offsets)
        # If one wants to mute below given points, the found velocity reduces by given indent.
        self.intercept = lin_reg.intercept_
        self.velocity = lin_reg.coef_ - add_velocity

    def get_time(self, offsets):
        return (offsets - self.intercept) / self.velocity # ms


class Muting(BaseMuting):
    def __init__(self, path=None, points=None, bias=None, fill_value='extrapolate'):
        if path is not None:
            points = self.load_muting(path)
        if points is None:
            raise ValueError('`Points` or `path` must be given.')
        self.times, self.offsets = points[:, 0], points[:, 1]
        self.interp_func = interp1d(self.offsets, self.times, fill_value=fill_value)

    def get_time(self, offsets):
        return self.interp_func(offsets) # ms

    def load_muting(self, path):
        muting = path
        # some loading process
        return muting
