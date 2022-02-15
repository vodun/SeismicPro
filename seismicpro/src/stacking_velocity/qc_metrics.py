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

from .qc_maps import StackingVelocityScatterMapPlot
from ..metrics import PlottableMetric
from ..utils import set_ticks


class StackingVelocityMetric(PlottableMetric):
    is_window_metric = True
    interactive_scatter_map_class = StackingVelocityScatterMapPlot

    def __init__(self, times, velocities, nearest_neighbors, velocity_cube):
        self.times = times
        self.velocities = velocities
        self.min_vel = self.velocities.min()
        self.max_vel = self.velocities.max()
        self.nearest_neighbors = nearest_neighbors
        self.velocity_cube = velocity_cube

    def coords_to_window(self, coords):
        _, window_indices = self.nearest_neighbors.radius_neighbors([coords], return_distance=True, sort_results=True)
        window_indices = window_indices[0]
        if not self.is_window_metric:
            window_indices = window_indices[:1]
        return self.velocities[window_indices]

    def plot(self, window, ax, x_ticker, y_ticker):
        for vel in window:
            ax.plot(vel, self.times)
        ax.invert_yaxis()
        set_ticks(ax, "x", "Stacking velocity (m/s)", **x_ticker)
        set_ticks(ax, "y", "Time", **y_ticker)
        ax.set_xlim(self.min_vel, self.max_vel)

    def plot_on_click(self, coords, ax, x_ticker, y_ticker):
        window = self.coords_to_window(coords)
        self.plot(window, ax=ax, x_ticker=x_ticker, y_ticker=y_ticker)


class IsDecreasing(StackingVelocityMetric):
    name = "is_decreasing"
    min_value = 0
    max_value = 1
    is_lower_better = True
    is_window_metric = False

    @staticmethod
    @njit(nogil=True)
    def calc(times, stacking_velocity):
        """Return whether the central stacking velocity of the window decreases at some time."""
        _ = times
        for cur_vel, next_vel in zip(stacking_velocity[:-1], stacking_velocity[1:]):
            if cur_vel > next_vel:
                return True
        return False

    def plot(self, window, ax, x_ticker, y_ticker):
        super().plot(window, ax, x_ticker, y_ticker)

        # Highlight decreasing sections
        stacking_velocity = window[0]
        decreasing_pos = np.where(np.diff(stacking_velocity) < 0)[0]
        if len(decreasing_pos):
            # Process each continuous decreasing section independently
            for section in np.split(decreasing_pos, np.where(np.diff(decreasing_pos) != 1)[0] + 1):
                section_slice = slice(section[0], section[-1] + 2)
                ax.plot(stacking_velocity[section_slice], self.times[section_slice], color="red")


class MaxAccelerationDeviation(StackingVelocityMetric):
    name = "max_acceleration_deviation"
    min_value = 0
    max_value = None
    is_lower_better = None
    is_window_metric = False

    @staticmethod
    @njit(nogil=True)
    def calc(times, stacking_velocity):
        mean_acc = (stacking_velocity[-1] - stacking_velocity[0]) / (times[-1] - times[0])
        max_deviation = 0
        for i in range(len(times) - 1):
            instant_acc = (stacking_velocity[i + 1] - stacking_velocity[i]) / (times[i + 1] - times[i])
            deviation = abs(instant_acc - mean_acc)
            if deviation > max_deviation:
                max_deviation = deviation
        return max_deviation

    def plot(self, window, ax, x_ticker, y_ticker):
        super().plot(window, ax, x_ticker, y_ticker)

        # Plot a mean-acceleration line
        stacking_velocity = window[0]
        ax.plot([stacking_velocity[0], stacking_velocity[-1]], [self.times[0], self.times[-1]], "--")


class MaxStandardDeviation(StackingVelocityMetric):
    name = "max_standard_deviation"
    min_value = 0
    max_value = None
    is_lower_better = True
    is_window_metric = True

    @staticmethod
    @njit(nogil=True)
    def calc(times, window):
        """Return the maximal spatial velocity standard deviation in a window over all the times."""
        _ = times
        max_std = 0
        for i in range(window.shape[1]):
            current_std = window[:, i].std()
            max_std = max(max_std, current_std)
        return max_std


class MaxRelativeVariation(StackingVelocityMetric):
    name = "max_relative_variation"
    min_value = 0
    max_value = None
    is_lower_better = True
    is_window_metric = True

    @staticmethod
    @njit(nogil=True)
    def calc(times, window):
        """Return the maximal absolute relative difference between central stacking velocity and the average of all
        remaining velocities in the window over all the times."""
        _ = times
        max_rel_var = 0
        for i in range(window.shape[1]):
            current_rel_var = abs(np.mean(window[1:, i]) - window[0, i]) / window[0, i]
            max_rel_var = max(max_rel_var, current_rel_var)
        return max_rel_var


VELOCITY_QC_METRICS = [IsDecreasing, MaxAccelerationDeviation, MaxStandardDeviation, MaxRelativeVariation]
