"""Implements metrics for quality control of stacking velocities.

In order to define your own metric you need to inherit a new class from `StackingVelocityMetric` and do the following:
* Set an `is_window_metric` class attribute to `True` or `False` depending on whether your metric needs all stacking
  velocities in a spatial window or only the central one in its `calc` method. In the first case, the central velocity
  will be the first one in the stacked 2d array of velocities.
* Optionally define all other class attributes of `Metric` for future convenience.
* Redefine `calc_metric` static method, which must accept two arguments: stacking velocities and times they are
  estimated for. If `is_window_metric` is `False`, stacking velocities will be a 1d array, otherwise it will be a 2d
  array with shape `(n_velocities, n_times)`. Times are always represented as a 1d array. `calc_metric` must return a
  single metric value.
* Optionally redefine `plot` method which will be used to plot stacking velocities on click on a metric map in
  interactive mode. By default it plots all stacking velocities used by `calc` during metric calculation. Note, that
  `plot` always accepts a 2d array of velocities as its first argument regardless of the `is_window_metric` value.

If you want the created metric to be calculated by :func:`~velocity_cube.VelocityCube.qc` method by default, it should
also be appended to a `VELOCITY_QC_METRICS` list.
"""

import numpy as np
from numba import njit
from matplotlib import patches

from ..metrics import Metric, ScatterMapPlot, MetricMap
from ..utils import calculate_axis_limits, set_ticks, set_text_formatting


class StackingVelocityScatterMapPlot(ScatterMapPlot):
    """Equivalent to `ScatterMapPlot` class except for `click` method, which also highlights a spatial window in which
    the metric was calculated."""
    def __init__(self, *args, plot_window=True, **kwargs):
        self.plot_window = plot_window
        self.window = None
        super().__init__(*args, **kwargs)

    def click(self, coords):
        """Process the click and highlight a spatial window in which the metric was calculated."""
        coords = super().click(coords)
        if self.window is not None:
            self.window.remove()
        if self.metric_map.is_window_metric and self.plot_window:
            self.window = patches.Circle(coords, self.metric_map.nearest_neighbors.radius, color="blue", alpha=0.3)
            self.main.ax.add_patch(self.window)
        return coords


class StackingVelocityMetricMap(MetricMap):
    """Equivalent to `MetricMap` class except for interactive scatter plot class, which highlights a spatial window in
    which the metric was calculated on click."""
    interactive_scatter_map_class = StackingVelocityScatterMapPlot


class StackingVelocityMetric(Metric):
    """Base metric class for quality control of stacking velocities."""
    is_window_metric = True
    map_class = StackingVelocityMetricMap
    views = "plot_on_click"

    def __init__(self, times, velocities, nearest_neighbors):
        super().__init__()
        self.times = times
        self.velocities = velocities
        self.velocity_limits = calculate_axis_limits(velocities)
        self.nearest_neighbors = nearest_neighbors

    @staticmethod
    def calc_metric(*args, **kwargs):
        """Calculate the metric. Must be overridden in child classes."""
        _ = args, kwargs
        raise NotImplementedError

    @classmethod
    def calc(cls, *args, **kwargs):
        """Redirect metric calculation to a static `calc_metric` method which may be njitted."""
        return cls.calc_metric(*args, **kwargs)

    def coords_to_window(self, coords):
        """Return all stacking velocities in a spatial window around given `coords`."""
        _, window_indices = self.nearest_neighbors.radius_neighbors([coords], return_distance=True, sort_results=True)
        window_indices = window_indices[0]
        if not self.is_window_metric:
            window_indices = window_indices[:1]
        return self.velocities[window_indices]

    def plot(self, window, ax, x_ticker=None, y_ticker=None, **kwargs):
        """Plot all stacking velocities in a spatial window."""
        (x_ticker, y_ticker), kwargs = set_text_formatting(x_ticker, y_ticker, **kwargs)
        for vel in window:
            ax.plot(vel, self.times, color="tab:blue")
        ax.invert_yaxis()
        set_ticks(ax, "x", "Stacking velocity (m/s)", **x_ticker)
        set_ticks(ax, "y", "Time", **y_ticker)
        ax.set_xlim(*self.velocity_limits)

    def plot_on_click(self, coords, ax, **kwargs):
        """Plot all stacking velocities used by `calc` during metric calculation."""
        window = self.coords_to_window(coords)
        self.plot(window, ax=ax, **kwargs)


class IsDecreasing(StackingVelocityMetric):
    """Check if a stacking velocity decreases at some time."""
    name = "is_decreasing"
    min_value = 0
    max_value = 1
    is_lower_better = True
    is_window_metric = False

    @staticmethod
    @njit(nogil=True)
    def calc_metric(stacking_velocity, times):
        """Return whether the stacking velocity decreases at some time."""
        _ = times
        for cur_vel, next_vel in zip(stacking_velocity[:-1], stacking_velocity[1:]):
            if cur_vel > next_vel:
                return True
        return False

    def plot(self, window, ax, **kwargs):
        """Plot the stacking velocity and highlight segments with decreasing velocity in red."""
        super().plot(window, ax, **kwargs)

        # Highlight decreasing sections
        stacking_velocity = window[0]
        decreasing_pos = np.where(np.diff(stacking_velocity) < 0)[0]
        if len(decreasing_pos):
            # Process each continuous decreasing section independently
            for section in np.split(decreasing_pos, np.where(np.diff(decreasing_pos) != 1)[0] + 1):
                section_slice = slice(section[0], section[-1] + 2)
                ax.plot(stacking_velocity[section_slice], self.times[section_slice], color="tab:red")


class MaxAccelerationDeviation(StackingVelocityMetric):
    """Calculate maximal deviation of instantaneous acceleration from the mean acceleration over all times."""
    name = "max_acceleration_deviation"
    min_value = 0
    max_value = None
    is_lower_better = None
    is_window_metric = False

    @staticmethod
    @njit(nogil=True)
    def calc_metric(stacking_velocity, times):
        """Return the maximal deviation of instantaneous acceleration from the mean acceleration over all times."""
        mean_acc = (stacking_velocity[-1] - stacking_velocity[0]) / (times[-1] - times[0])
        max_deviation = 0
        for i in range(len(times) - 1):
            instant_acc = (stacking_velocity[i + 1] - stacking_velocity[i]) / (times[i + 1] - times[i])
            deviation = abs(instant_acc - mean_acc)
            if deviation > max_deviation:
                max_deviation = deviation
        return max_deviation

    def plot(self, window, ax, **kwargs):
        """Plot the stacking velocity and a mean-acceleration line in dashed red."""
        super().plot(window, ax, **kwargs)

        # Plot a mean-acceleration line
        stacking_velocity = window[0]
        ax.plot([stacking_velocity[0], stacking_velocity[-1]], [self.times[0], self.times[-1]], "--", color="tab:red")


class MaxStandardDeviation(StackingVelocityMetric):
    """Calculate maximal spatial velocity standard deviation in a window over all times."""
    name = "max_standard_deviation"
    min_value = 0
    max_value = None
    is_lower_better = True
    is_window_metric = True

    @staticmethod
    @njit(nogil=True)
    def calc_metric(window, times):
        """Return the maximal spatial velocity standard deviation in a window over all times."""
        _ = times
        if window.shape[0] == 0:
            return 0

        max_std = 0
        for i in range(window.shape[1]):
            current_std = window[:, i].std()
            max_std = max(max_std, current_std)
        return max_std


class MaxRelativeVariation(StackingVelocityMetric):
    """Calculate maximal absolute relative difference between central stacking velocity and the average of all
    remaining velocities in the window over all times."""
    name = "max_relative_variation"
    min_value = 0
    max_value = None
    is_lower_better = True
    is_window_metric = True

    @staticmethod
    @njit(nogil=True)
    def calc_metric(window, times):
        """Return the maximal absolute relative difference between central stacking velocity and the average of all
        remaining velocities in the window over all times."""
        _ = times
        if window.shape[0] == 0:
            return 0

        max_rel_var = 0
        for i in range(window.shape[1]):
            current_rel_var = abs(np.mean(window[1:, i]) - window[0, i]) / window[0, i]
            max_rel_var = max(max_rel_var, current_rel_var)
        return max_rel_var

    def plot(self, window, ax, **kwargs):
        """Plot all stacking velocities in spatial window and highlight the central one in red."""
        super().plot(window[1:], ax, **kwargs)
        ax.plot(window[0], self.times, color="tab:red")


VELOCITY_QC_METRICS = [IsDecreasing, MaxAccelerationDeviation, MaxStandardDeviation, MaxRelativeVariation]
