"""Implements the weathering velocity metrics"""

from ..metrics import PipelineMetric, pass_calc_args
from ..const import HDR_FIRST_BREAK

# pylint: disable=unused-argument
class WeatheringVelocityMetric(PipelineMetric):
    """Calculate a weathering metric. Weathering metric is fraction of first breaking times that stands out from
    a weathering velocity curve (piecewise linear function) more that threshold"""
    name = "weathering metrics"
    vmin = 0
    vmax = 0.05
    is_lower_better = True
    views = ("plot", "plot_wv")

    @staticmethod
    def calc(gather, weathering_velocity, first_breaks_col=HDR_FIRST_BREAK, threshold_times=50):
        """Return calculated a fraction of first breaking times that stands out from a weathering velocity curve
        (piecewise linear function) more that `threshold_times`"""
        return gather.calculate_weathering_metrics(weathering_velocity, first_breaks_col=first_breaks_col,
                                                   threshold_time=threshold_times)

    @pass_calc_args
    def plot(cls, gather, weathering_velocity, ax, threshold_times=50, mode='seismogram',
             first_breaks_col=HDR_FIRST_BREAK, **kwargs):
        """Plot the gather and first break points over it."""
        gather.plot(ax=ax, event_headers={'headers': HDR_FIRST_BREAK}, mode=mode, **kwargs)

    @pass_calc_args
    def plot_wv(cls, gather, weathering_velocity, ax, first_breaks_col=HDR_FIRST_BREAK, threshold_times=50, **kwargs):
        """Plot the first break picking points, weathering velocity curve, thresholds and weathering model
        parameters."""
        weathering_velocity.plot(ax=ax, threshold_time=threshold_times, **kwargs)
