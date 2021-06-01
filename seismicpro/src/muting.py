import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

from .utils import read_single_vfunc


class Muter:
    def __init__(self, mode="file", **kwargs):
        if not hasattr(self, f"from_{mode}"):
            raise ValueError(f"Unknown mode {mode}")
        self.muter = getattr(self, f"from_{mode}")(**kwargs)

    def from_points(self, offsets, times, fill_value="extrapolate"):
        return interp1d(offsets, times, fill_value=fill_value)

    def from_file(self, path, **kwargs):
        _, _, offsets, times = read_single_vfunc(path)
        return self.from_points(offsets, times, **kwargs)

    def from_first_breaks(self, offsets, times, velocity_reduction=0):
        velocity_reduction = velocity_reduction / 1000  # from m/s to m/ms
        lin_reg = LinearRegression(fit_intercept=True)
        lin_reg.fit(np.array(times).reshape(-1, 1), np.array(offsets))

        # The fitted velocity is reduced by velocity_reduction in order to mute amplitudes near first breaks
        intercept = lin_reg.intercept_
        velocity = lin_reg.coef_ - velocity_reduction
        return lambda offsets: (offsets - intercept) / velocity

    def __call__(self, offsets):
        return self.muter(offsets)
