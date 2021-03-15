import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression


class BaseMuting:
    def create_mask(self, gather_shape):
        return (np.arange(gather_shape[1]).reshape(1, -1) - self.mute_ixs.reshape(-1, 1)) > 0

    def interpolate(self, offsets, sample_rate, fill_value=None):
        interp_func = interp1d(self.x, self.y, fill_value=fill_value)
        self.mute_ixs = interp_func(offsets) / sample_rate


class PickingMuting(BaseMuting):
    def __init__(self, times, offsets, sample_rate, add_velocity=0):
        self.add_velocity = add_velocity / 1000 # from m/s to m/ms
        self.x = np.array(times, dtype=np.int32)
        self.y = offsets
        lin_reg = LinearRegression(fit_intercept=True)
        lin_reg.fit(self.x.reshape(-1, 1), self.y)
        # If one wants to mute below given points, the found velocity reduces by given indent.
        velocity = lin_reg.coef_ - self.add_velocity
        mute_ixs = self.y / (velocity * sample_rate) # m/samples
        self.mute_ixs = mute_ixs.astype(int)


class Muting(BaseMuting):
    def __init__(self, points=None, path=None, indent=None):
        if points is None:
            if path is None:
                raise ValueError('`Points` or `path` must be given.')
            points = self.load_muting(path)
        self.y, self.x = points[:, 0], points[:, 1]

    def load_muting(self, path):
        muting = path
        # some loading process
        return muting
