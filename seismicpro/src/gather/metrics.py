"""Implements the gather quality control metrics."""

import numpy as np

from ..metrics import PipelineMetric, pass_calc_args
from ..const import HDR_FIRST_BREAK


class FirstBreaksOutliers(PipelineMetric):
    """Calculate the first break outliers metric.

    The outlier is considered to be the first break time that stands out from the expected arrival time
    (weathering velocity curve) for more than threshold.
    """
    name = "weathering_metrics"
    vmin = 0
    vmax = 0.05
    is_lower_better = True
    views = ("plot_gather", "plot_weathering_velocity")
    args_to_unpack = ("gather", "weathering_velocity")

    @staticmethod
    def calc(gather, weathering_velocity, first_breaks_col=HDR_FIRST_BREAK, threshold_times=50, **kwargs):
        """Calculates the first break outliers metric value.

        Weathering metric calculated as a fraction of traces in the gather whose first break times differ
        from estimated by weathering velocity model for more than `threshold_times`.

        Parameters
        ----------
        weathering_velocity : WeatheringVelocity
            Weathering velocity used to estimate expected first break times.
        first_breaks_col : str, defaults to HDR_FIRST_BREAK
            Column name  from `gather.headers` where first breaking times are stored.
        threshold_times: float, defaults to 50
            Threshold for the weathering metric calculation.

        Returns
        -------
        metric : float
            Fraction of traces in the gather whose first break times differ from estimated by weathering velocity model
            for more than threshold_times.
        """
        _ = kwargs
        metric = np.abs(weathering_velocity(gather.offsets) - gather[first_breaks_col].ravel()) > threshold_times
        return np.mean(metric)

    @pass_calc_args
    def plot_gather(cls, gather, weathering_velocity, first_breaks_col=HDR_FIRST_BREAK, threshold_times=50, **kwargs):
        """Plot the gather and the first break points."""
        _ = weathering_velocity, threshold_times
        event_headers = kwargs.pop('event_headers', {'headers': first_breaks_col})
        gather.plot(event_headers=event_headers, **kwargs)

    @pass_calc_args
    def plot_weathering_velocity(cls, gather, weathering_velocity, first_breaks_col=HDR_FIRST_BREAK,
                                 threshold_times=50, **kwargs):
        """Plot the weathering velocity curve and show the threshold area used for metric calculation."""
        _ = gather, first_breaks_col
        weathering_velocity.plot(threshold_times=threshold_times, **kwargs)


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
