"""Implements gather quality control metrics."""

from copy import copy

import numpy as np

from ..const import HDR_FIRST_BREAK
from ..metrics import PipelineMetric
from ..refractor_velocity import RefractorVelocity, RefractorVelocityField


class FirstBreaksOutliers(PipelineMetric):
    """Calculate the first break outliers metric.

    A first break time is considered to be an outlier if it differs from the expected arrival time defined by
    an offset-traveltime curve by more than a given threshold.
    """
    name = "first_breaks_outliers"
    is_lower_better = True
    min_value = 0
    vmin = 0
    vmax = 0.05
    views = ("plot_gather", "plot_refractor_velocity")
    args_to_unpack = ("gather", "refractor_velocity")

    def __call__(self, gather, refractor_velocity, first_breaks_col=HDR_FIRST_BREAK, threshold_times=50,
                 correct_uphole=None):
        """Calculate the first break outliers metric.

        Returns the fraction of traces in the gather whose times of first breaks differ from those estimated by a
        near-surface velocity model by more than `threshold_times`.

        Parameters
        ----------
        gather : Gather
            A seismic gather to get offsets and times of first breaks from.
        refractor_velocity : RefractorVelocity or RefractorVelocityField
            Near-surface velocity model to estimate the expected times of first breaks at `gather` offsets.
        first_breaks_col : str, optional, defaults to :const:`~const.HDR_FIRST_BREAK`
            Column name from `gather.headers` where times of first breaks are stored.
        threshold_times: float, optional, defaults to 50
            Threshold for the first breaks outliers metric calculation. Measured in milliseconds.
        correct_uphole : bool, optional
            Whether to perform uphole correction by adding values of "SourceUpholeTime" header to times of first breaks
            emulating the case when sources are located on the surface. If not given, correction is performed if
            "SourceUpholeTime" header is loaded and given `refractor_velocity` was also uphole corrected.

        Returns
        -------
        metric : float
            Fraction of traces in the gather whose times of first breaks differ from those estimated by the velocity
            model by more than `threshold_times`.
        """
        if isinstance(refractor_velocity, RefractorVelocityField):
            refractor_velocity = refractor_velocity(gather.coords)
        if not isinstance(refractor_velocity, RefractorVelocity):
            raise ValueError("refractor_velocity must be of RefractorVelocity or RefractorVelocityField type")
        expected_times = refractor_velocity(gather.offsets)
        fb_times = gather[first_breaks_col]
        if correct_uphole is None:
            correct_uphole = "SourceUpholeTime" in gather.available_headers and refractor_velocity.is_uphole_corrected
        if correct_uphole:
            fb_times = fb_times + gather["SourceUpholeTime"]
        metric = np.abs(expected_times - fb_times) > threshold_times
        return np.mean(metric)

    @staticmethod
    def plot_gather(gather, refractor_velocity, first_breaks_col=HDR_FIRST_BREAK, threshold_times=50,
                    correct_uphole=None, *, ax, **kwargs):
        """Plot the gather and its first breaks."""
        _ = refractor_velocity, threshold_times, correct_uphole
        event_headers = kwargs.pop('event_headers', {'headers': first_breaks_col})
        gather.plot(ax=ax, event_headers=event_headers, **kwargs)

    @staticmethod
    def plot_refractor_velocity(gather, refractor_velocity, first_breaks_col=HDR_FIRST_BREAK, threshold_times=50,
                                correct_uphole=None, *, ax, **kwargs):
        """Plot the refractor velocity curve and show the threshold area used for metric calculation."""
        fb_times = gather[first_breaks_col]
        if correct_uphole is None:
            correct_uphole = "SourceUpholeTime" in gather.available_headers and refractor_velocity.is_uphole_corrected
        if correct_uphole:
            fb_times = fb_times + gather["SourceUpholeTime"]

        if isinstance(refractor_velocity, RefractorVelocityField):
            refractor_velocity = refractor_velocity(gather.coords)
        elif isinstance(refractor_velocity, RefractorVelocity):
            refractor_velocity = copy(refractor_velocity)
        else:
            raise ValueError("refractor_velocity must be of RefractorVelocity or RefractorVelocityField type")
        refractor_velocity.offsets = gather["offset"]
        refractor_velocity.times = fb_times
        refractor_velocity.max_offset = max(refractor_velocity.max_offset, gather["offset"].max())
        refractor_velocity.plot(ax=ax, threshold_times=threshold_times, **kwargs)


class SignalLeakage(PipelineMetric):
    """Calculate signal leakage after ground-roll attenuation.

    The metric is based on the assumption that a vertical velocity spectrum calculated for the difference between
    processed and source gathers should not have pronounced energy maxima.
    """
    name = "signal_leakage"
    is_lower_better = True
    min_value = 0
    max_value = None
    views = ("plot_diff_gather", "plot_diff_velocity_spectrum")
    args_to_unpack = ("gather_before", "gather_after")

    @staticmethod
    def get_diff_gather(gather_before, gather_after):
        """Construct a new gather whose amplitudes are element-wise differences of amplitudes from `gather_after` and
        `gather_before`."""
        gather_diff = gather_after.copy(ignore=["data", "headers", "samples"])
        gather_diff.data = gather_after.data - gather_before.data
        return gather_diff

    def __call__(self, gather_before, gather_after, velocities=None):
        """Calculate signal leakage when moving from `gather_before` to `gather_after`."""
        gather_diff = self.get_diff_gather(gather_before, gather_after)
        spectrum_diff = gather_diff.calculate_vertical_velocity_spectrum(velocities)
        spectrum_before = gather_before.calculate_vertical_velocity_spectrum(velocities)
        # TODO: update calculation logic, probably sum of semblance values of gather diff along stacking velocity,
        # picked for gather_before / gather_after will perform way better
        signal_leakage = (spectrum_diff.velocity_spectrum.ptp(axis=1) /
                          (1 + 1e-6 - spectrum_before.velocity_spectrum.ptp(axis=1)))
        return max(0, np.max(signal_leakage))

    def plot_diff_gather(self, gather_before, gather_after, velocities=None, *, ax, **kwargs):
        """Plot the difference between `gather_after` and `gather_before`."""
        _ = velocities
        gather_diff = self.get_diff_gather(gather_before, gather_after)
        gather_diff.plot(ax=ax, **kwargs)

    def plot_diff_velocity_spectrum(self, gather_before, gather_after, velocities=None, *, ax, **kwargs):
        """Plot a velocity spectrum of the difference between `gather_after` and `gather_before`."""
        gather_diff = self.get_diff_gather(gather_before, gather_after)
        spectrum_diff = gather_diff.calculate_vertical_velocity_spectrum(velocities)
        spectrum_diff.plot(ax=ax, **{"title": "", **kwargs})
