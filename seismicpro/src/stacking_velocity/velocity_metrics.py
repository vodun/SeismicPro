"""Implements metrics for quality control of stacking velocities.

Each of the defined functions:
* accepts a set of stacking velocities in a spatial window stacked into a 2d `np.ndarray` with shape
  `(n_velocities, n_times)`, where the central velocity of the window has index 0 along the first axis,
* returns a single value of a metric being plotted on the field map.

In order for a function to be available in the :func:`~velocity_cube.VelocityCube.qc` method, it should also be
appended to a `VELOCITY_QC_METRICS` list.
"""

import numpy as np
from numba import njit

from ..metrics import PlottableMetric
from ..utils import set_ticks


class StackingVelocityMetric(PlottableMetric):
    name = None
    is_window_metric = True

    def __init__(self, nearest_neighbors, times, velocities):
        self.nearest_neighbors = nearest_neighbors
        self.times = times
        self.velocities = velocities
        self.min_vel = self.velocities.min()
        self.max_vel = self.velocities.max()

    def coords_to_args(self, coords):
        window_indices = self.nearest_neighbors.radius_neighbors([coords], return_distance=False)[0]
        if not self.is_window_metric:
            window_indices = window_indices[0]
        return (self.velocities[window_indices],)

    def plot(self, window, ax, x_ticker, y_ticker):
        if not self.is_window_metric:
            window = window.reshape(1, -1)
        for vel in window:
            ax.plot(vel, self.times)
        ax.invert_yaxis()
        set_ticks(ax, "x", "Stacking velocity (m/s)", **x_ticker)
        set_ticks(ax, "y", "Time", self.times, **y_ticker)
        ax.set_xlim(self.min_vel, self.max_vel)


class IsDecreasing(StackingVelocityMetric):
    name = "is_decreasing"
    is_lower_better = True
    is_window_metric = False
    vmin = 0
    vmax = 1

    @staticmethod
    @njit(nogil=True)
    def calc(stacking_velocity):
        """Return whether the central stacking velocity of the window decreases at some time."""
        for cur_vel, next_vel in zip(stacking_velocity[:-1], stacking_velocity[1:]):
            if cur_vel > next_vel:
                return True
        return False


class MaxStandardDeviation(StackingVelocityMetric):
    name = "max_standard_deviation"
    is_lower_better = True
    is_window_metric = True
    vmin = 0
    vmax = None

    @staticmethod
    @njit(nogil=True)
    def calc(window):
        """Return the maximal spatial velocity standard deviation in a window over all the times."""
        max_std = 0
        for i in range(window.shape[1]):
            current_std = window[:, i].std()
            max_std = max(max_std, current_std)
        return max_std


class MaxRelativeVariation(StackingVelocityMetric):
    name = "max_relative_variation"
    is_lower_better = True
    is_window_metric = True
    vmin = 0
    vmax = None

    @staticmethod
    @njit(nogil=True)
    def calc(window):
        """Return the maximal absolute relative difference between central stacking velocity and the average of all
        remaining velocities in the window over all the times."""
        max_rel_var = 0
        for i in range(window.shape[1]):
            current_rel_var = abs(np.mean(window[1:, i]) - window[0, i]) / window[0, i]
            max_rel_var = max(max_rel_var, current_rel_var)
        return max_rel_var


VELOCITY_QC_METRICS = [IsDecreasing, MaxStandardDeviation, MaxRelativeVariation]
