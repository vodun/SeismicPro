import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression


class BaseMuting:
    def create_mask(self, gather_shape):
        return (np.arange(gather_shape[1]).reshape(1, -1) - self.mute_ixs.reshape(-1, 1)) > 0

    def interpolate(self, x, y, offsets, fill_value=None):
        interp_func = interp1d(x, y, fill_value=fill_value)
        mute_samples = interp_func(offsets) / self.sample_rate
        return mute_samples


class PickingMuting(BaseMuting):
    def __init__(self, times, offsets, sample_rate, add_velocity=0):
        self.sample_rate = sample_rate
        times = np.array(times, dtype=np.int32)
        lin_reg = LinearRegression(fit_intercept=True)
        lin_reg.fit(times.reshape(-1, 1), offsets)
        add_velocity /= 1000 # from m/s to m/ms
        # If one wants to mute below given points, the found velocity reduces by given indent.
        velocity = lin_reg.coef_ - add_velocity
        mute_ixs = offsets / (velocity * self.sample_rate) # m/samples
        self.mute_ixs = mute_ixs.astype(int)


class Muting(BaseMuting):
    def __init__(self, sample_rate, offsets, points=None, path=None, indent=None):
        self.sample_rate = sample_rate
        if points is None:
            if path is None:
                raise ValueError('`Points` or `path` must be given.')
            points = self.load_muting(path)
        point_times, point_offsets = points[:, 0], points[:, 1]
        # Pointwise interpolation for specified muting points.
        self.mute_ixs = self.interpolate(point_offsets, point_times, offsets, fill_value='extrapolate')

    def load_muting(self, path):
        muting = path
        # some loading process
        return muting
