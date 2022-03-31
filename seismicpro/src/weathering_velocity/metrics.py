"""Implements the weathering velocity metrics"""

from ..metrics import PipelineMetric, pass_calc_args
from ..const import HDR_FIRST_BREAK

class WeatheringVelocityMetric(PipelineMetric):
    name = "weathering metrics"
    vmin = 0
    vmax = 0.05
    is_lower_better = True
    views = ("plot", "plot_wv")

    @staticmethod
    def calc(gather, weathering_velocity, first_breaks_col=HDR_FIRST_BREAK, threshold_times=50):
        return gather.calculate_weathering_metrics(weathering_velocity, first_breaks_col=first_breaks_col,
                                                   threshold_time=threshold_times)

    @pass_calc_args
    def plot(cls, gather, weathering_velocity, ax, threshold_times=50, mode='seismogram',
             first_breaks_col=HDR_FIRST_BREAK, **kwargs):
        gather.plot(ax=ax, event_headers={'headers': HDR_FIRST_BREAK}, mode=mode, **kwargs)

    @pass_calc_args
    def plot_wv(cls, gather, weathering_velocity, ax, first_breaks_col=HDR_FIRST_BREAK, threshold_times=50, **kwargs):
        weathering_velocity.plot(ax=ax, threshold_time=threshold_times, **kwargs)
