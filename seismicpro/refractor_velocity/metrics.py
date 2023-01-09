import numpy as np
from numba import njit
from matplotlib import patches
from scipy.signal import hilbert

from ..metrics import Metric, ScatterMapPlot, MetricMap
from ..const import HDR_FIRST_BREAK


class RefractorVelocityScatterMapPlot(ScatterMapPlot):
    def __init__(self, metric_map, plot_on_click, **kwargs):
        plot_on_click_kwargs = kwargs.get('plot_on_click_kwargs', [{}, {}])
        plot_on_click_kwargs = [plot_on_click_kwargs, plot_on_click_kwargs.copy()]\
                                if not isinstance(plot_on_click_kwargs, list) else plot_on_click_kwargs

        plot_on_click_kwargs[1]['map_data'] = metric_map.map_data
        kwargs['plot_on_click_kwargs'] = plot_on_click_kwargs
        super().__init__(metric_map, plot_on_click, **kwargs)


class RefractorVelocityMetricMap(MetricMap):
    interactive_scatter_map_class = RefractorVelocityScatterMapPlot


class RefractorVelocityMetric(Metric):
    map_class = RefractorVelocityMetricMap
    views = ("plot_gather", "plot_refractor_velocity")

    def __init__(self, survey=None, field=None,
                    first_breaks_col=HDR_FIRST_BREAK, threshold_times=50, **kwargs):
        super().__init__(**kwargs)
        self.survey = survey
        self.field = field
        self.first_breaks_col = first_breaks_col
        self.threshold_times = threshold_times

    def plot_gather(self, coords, ax, **kwargs):
        gather = self.survey.get_gather(coords)
        event_headers = kwargs.pop('event_headers', {'headers': self.first_breaks_col})
        gather.plot(event_headers=event_headers, ax=ax, **kwargs)
        
    def plot_refractor_velocity(self, coords, ax, **kwargs):
        refractor_velocity = self.field(coords)
        if not refractor_velocity.is_fit:
            gather_headers = self.survey.headers.loc[coords]
            refractor_velocity.times = gather_headers[self.first_breaks_col]
            refractor_velocity.offsets = gather_headers['offset']
        refractor_velocity.plot(threshold_times=self.threshold_times, ax=ax, **kwargs)

        
class FirstBreaksOutliers(RefractorVelocityMetric):
    name = "first_breaks_outliers"
    vmin = 0
    vmax = 0.05
    is_lower_better = True
    
    @staticmethod
    def _calc(times, offsets, rv, threshold_times):
        return np.mean(np.abs(rv(offsets) - times) > threshold_times)
    
    @classmethod
    def calc(cls, gathers, refractor_velocities, threshold_times=50, first_breaks_col=HDR_FIRST_BREAK):
        metric = []
        for gather, rv in zip(gathers, refractor_velocities):
            metric.append(cls._calc(gather[first_breaks_col], gather['offset'], rv, threshold_times))
        return metric 
    
    def plot_refractor_velocity(self, coords, ax, **kwargs):
        _ = kwargs.pop('map_data')
        super().plot_refractor_velocity(coords, ax, **kwargs)


class FirstBreaksAmplitudes(RefractorVelocityMetric):
    name = "firs_break_amplitudes"
    vmin = 0
    vmax = 3
    is_lower_better = True # ?
    
    @classmethod
    def calc(cls, gathers, refractor_velocities, first_breaks_col=HDR_FIRST_BREAK):
        _ = refractor_velocities
        metric = []
        for gather in gathers:
            ix = np.round(gather[first_breaks_col] / gather.sample_rate).astype(int)
            gather.headers['amps'] = gather.data[np.arange(gather.shape[0]), ix]
            metric.append(gather['amps'].mean())
        return metric
    
#     def plot_gather(self, coords, ax, **kwargs):
#         kwargs['top_header'] = 'amps'
#         super().plot_gather(coords, ax, **kwargs)
        

class DivergencePoint(RefractorVelocityMetric):
    """Find offset after that first breaks are most likely to significantly diverge from expected time.
    Such an offset is found as one with the maximum increase of outliers after it(actually, maximum increase of 
    increase of outliers, as it seems to be more robust).
    Divergence point is set to be the maximum offset when the overall fraction of outliers does not exceeds given tolerance.
    """
    name = "divergence_point"
    is_lower_better = False
    
    def __init__(self, survey=None, **kwargs):
        super().__init__(survey, **kwargs)
        self.vmax = survey['offset'].max() if survey is not None else None
        self.vmin = survey['offset'].min() if survey is not None else None

    @staticmethod
    def _calc(times, offsets, rv, threshold_times, offset_step):
        sorted_offsets_idx = np.argsort(offsets)
        offsets = offsets[sorted_offsets_idx]
        times = times[sorted_offsets_idx]
        
        check_offsets = list(range(offsets.min() + offset_step, max(offsets) - offset_step, offset_step))
        check_offsets += list(range(offsets.min() + int(offset_step * 0.25),
                                    offsets.max() - offset_step, offset_step))
        check_offsets += list(range(offsets.min() + int(offset_step * 0.5),
                                    offsets.max() - offset_step, offset_step))
        
        check_offsets.sort()
        check_indices = np.searchsorted(offsets, check_offsets)
        n = len(check_indices)
        
        time_windows = [times[:idx] for idx in check_indices] + [times[idx:] for idx in check_indices] 
        offset_windows = [offsets[:idx] for idx in check_indices] + [offsets[idx:] for idx in check_indices]
        outliers = [FirstBreaksOutliers._calc(time_window, offset_window, rv, threshold_times=threshold_times) 
                    for time_window, offset_window in zip(time_windows, offset_windows)]
        
        diffs = [outliers[i + n] - outliers[i] for i in range(n)]
        diffs_delta = [diffs[i + 1] - diffs[i] for i in range(n - 1)]
        return check_offsets[np.argmax(diffs_delta) + 1]
    
    @classmethod
    def calc(cls, gathers, refractor_velocities, first_breaks_col=HDR_FIRST_BREAK,
             threshold_times=50, max_offset=None, tol=0.03, offset_step=100):
        metric = []
        outliers = FirstBreaksOutliers.calc(gathers, refractor_velocities, first_breaks_col=first_breaks_col, 
                                            threshold_times=threshold_times)
        
        max_offset = np.max(np.concatenate([gather['offset'] for gather in gathers])) if max_offset is None else max_offset
        metric = np.where(np.array(outliers) <= tol, max_offset, np.nan)
        tol_idx = np.nonzero(np.array(outliers) > tol)[0]
        for idx in tol_idx:
            metric[idx] = cls._calc(times=gathers[idx][first_breaks_col], offsets=gathers[idx]['offset'],
                                    rv=refractor_velocities[idx],
                                    threshold_times=threshold_times, offset_step=offset_step)
        return metric
        
    def plot_refractor_velocity(self, coords, ax, **kwargs):
        x_coord = kwargs.pop('map_data')[coords]
        ax.axvline(x=x_coord, color='k', linestyle='--')
        super().plot_refractor_velocity(coords, ax, **kwargs)
        
REFRACTOR_VELOCITY_QC_METRICS = [FirstBreaksOutliers, DivergencePoint, FirstBreaksAmplitudes]
        