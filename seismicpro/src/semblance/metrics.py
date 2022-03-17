"""Implements a metric estimating signal leakage"""

import numpy as np

from ..metrics import PipelineMetric, pass_calc_args


class SignalLeakage(PipelineMetric):
    """Calculate signal leakage during ground-roll attenuation.

    The metric is based on the assumption that a vertical velocity semblance calculated for the difference between
    processed and source gathers should not have pronounced energy maxima.
    """
    name = "signal_leakage"
    min_value = 0
    max_value = None
    is_lower_better = True
    views = ("plot_diff_gather", "plot_diff_semblance")
    args_to_unpack = ("gather_before", "gather_after")

    @staticmethod
    def get_diff_gather(gather_before, gather_after):
        """Construct a new gather whose amplitudes are elementwise differences of amplitudes from `gather_after` and
        `gather_before`."""
        gather_diff = gather_after.copy(ignore=["data", "headers", "samples"])
        gather_diff.data = gather_after.data - gather_before.data
        return gather_diff

    @classmethod
    def calc(cls, gather_before, gather_after, velocities):
        """Calculate signal leakage when moving from `gather_before` to `gather_after`."""
        gather_diff = cls.get_diff_gather(gather_before, gather_after)
        semblance_diff = gather_diff.calculate_semblance(velocities)
        semblance_before = gather_before.calculate_semblance(velocities)
        signal_leakage = semblance_diff.semblance.ptp(axis=1) / (1 + 1e-6 - semblance_before.semblance.ptp(axis=1))
        return max(0, np.max(signal_leakage))

    @pass_calc_args
    def plot_diff_gather(cls, gather_before, gather_after, velocities, ax, **kwargs):
        """Plot the difference between `gather_after` and `gather_before`."""
        _ = velocities
        gather_diff = cls.get_diff_gather(gather_before, gather_after)
        gather_diff.plot(ax=ax, **kwargs)

    @pass_calc_args
    def plot_diff_semblance(cls, gather_before, gather_after, velocities, ax, **kwargs):
        """Plot a semblance of the difference between `gather_after` and `gather_before`."""
        gather_diff = cls.get_diff_gather(gather_before, gather_after)
        semblance_diff = gather_diff.calculate_semblance(velocities)
        semblance_diff.plot(ax=ax, **{"title": None, **kwargs})
