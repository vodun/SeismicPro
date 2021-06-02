import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

from .utils import read_single_vfunc


class Muter:
    def __init__(self):
        self.muter = None

    @classmethod
    def from_points(cls, offsets, times, fill_value="extrapolate"):
        self = cls()
        self.muter = interp1d(offsets, times, fill_value=fill_value)
        return self

    @classmethod
    def from_file(cls, path, **kwargs):
        _, _, offsets, times = read_single_vfunc(path)
        return cls.from_points(offsets, times, **kwargs)

    @classmethod
    def from_first_breaks(cls, offsets, times, velocity_reduction=0):
        velocity_reduction = velocity_reduction / 1000  # from m/s to m/ms
        lin_reg = LinearRegression(fit_intercept=True)
        lin_reg.fit(np.array(times).reshape(-1, 1), np.array(offsets))

        # The fitted velocity is reduced by velocity_reduction in order to mute amplitudes near first breaks
        intercept = lin_reg.intercept_
        velocity = lin_reg.coef_ - velocity_reduction

        self = cls()
        self.muter = lambda offsets: (offsets - intercept) / velocity
        return self

    def __call__(self, offsets):
        return self.muter(offsets)
