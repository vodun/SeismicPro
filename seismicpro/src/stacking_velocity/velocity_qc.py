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


VELOCITY_QC_METRICS = ["is_decreasing", "max_standard_deviation", "max_relative_variation"]


@njit(nogil=True)
def is_decreasing(window):
    """Return whether the central stacking velocity of the window decreases at some time."""
    for i in range(window.shape[1] - 1):
        if window[0, i] > window[0, i + 1]:
            return True
    return False


@njit(nogil=True)
def max_standard_deviation(window):
    """Return the maximal spatial velocity standard deviation in a window over all the times."""
    max_std = 0
    for i in range(window.shape[1]):
        current_std = window[:, i].std()
        max_std = max(max_std, current_std)
    return max_std


@njit(nogil=True)
def max_relative_variation(window):
    """Return the maximal absolute relative difference between central stacking velocity and the average of all
    remaining velocities in the window over all the times."""
    max_rel_var = 0
    for i in range(window.shape[1]):
        current_rel_var = abs(np.mean(window[1:, i]) - window[0, i]) / window[0, i]
        max_rel_var = max(max_rel_var, current_rel_var)
    return max_rel_var
