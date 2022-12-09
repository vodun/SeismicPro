import numpy as np
from numba import njit
from matplotlib import patches

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
    
    def __init__(self, survey=None, refractor_velocities=None,
                    first_breaks_col=HDR_FIRST_BREAK, threshold_times=50, **kwargs):
        super().__init__()
        self.survey = survey
        self.first_breaks_col = first_breaks_col
        self.threshold_times = threshold_times
        self.refractor_velocities = {rv.coords.coords: rv for rv in refractor_velocities}\
                                    if refractor_velocities is not None else None
    
    @staticmethod
    def calc_metric(*args, **kwargs):
        _ = args, kwargs
        raise NotImplementedError

    @classmethod
    def calc(cls, *args, **kwargs):
        return cls.calc_metric(*args, **kwargs)
    
    def plot_gather(self, coords, ax, **kwargs):
        gather = self.survey.get_gather(coords)
        event_headers = kwargs.pop('event_headers', {'headers': self.first_breaks_col})
        gather.plot(event_headers=event_headers, ax=ax, **kwargs)
        
    def plot_refractor_velocity(self, coords, ax, **kwargs):
        refractor_velocity = self.refractor_velocities[coords]
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
    def calc_metric(times, offsets, refractor_velocities, threshold_times=50, **kwargs):
        metric = []
        for gather_times, gather_offsets, gather_rv in zip(times, offsets, refractor_velocities):
            metric.append(np.mean(np.abs(gather_rv(gather_offsets) - gather_times) > threshold_times))
        return metric 
    
    def plot_refractor_velocity(self, coords, ax, **kwargs):
        _ = kwargs.pop('map_data')
        super().plot_refractor_velocity(coords, ax, **kwargs)
        

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
        self.vmax = survey.headers['offset'].max() if survey is not None else None
        self.vmin = survey.headers['offset'].min() if survey is not None else None
        
    @staticmethod
    def calc_(times, offsets, rv, fb_metric, threshold_times, offset_step):
        sorted_offsets_idx = np.argsort(offsets)
        offsets = offsets.iloc[sorted_offsets_idx]
        times = times.iloc[sorted_offsets_idx]
        
        check_offsets = list(range(offsets.min() + offset_step, max(offsets) - offset_step, offset_step))
        check_offsets += list(range(offsets.min() + int(offset_step * 0.25), offsets.max() - offset_step, offset_step))
        check_offsets += list(range(offsets.min() + int(offset_step * 0.5), offsets.max() - offset_step, offset_step))
        
        check_offsets.sort()
        check_indices = np.searchsorted(offsets, check_offsets)
        n = len(check_indices)
        
        time_windows = [times[:idx] for idx in check_indices] + [times[idx:] for idx in check_indices] 
        offset_windows = [offsets[:idx] for idx in check_indices] + [offsets[idx:] for idx in check_indices]
        outliers = fb_metric.calc_metric(time_windows, offset_windows, [rv] * n * 2, threshold_times=threshold_times)
        
        diffs = [outliers[i + n] - outliers[i] for i in range(n)]
        diffs_delta = [diffs[i + 1] - diffs[i] for i in range(n - 1)]
        return check_offsets[np.argmax(diffs_delta, keepdims=True)[-1] + 1]
    
    @staticmethod
    def calc_metric(times, offsets, refractor_velocities, threshold_times=50, offset_step=100,
                    tol=0.03, max_offset=None, **kwargs):
        metric = []
        fb_metric = FirstBreaksOutliers()
        outliers = fb_metric.calc_metric(times, offsets, refractor_velocities, threshold_times=threshold_times)
        
        max_offset = np.max(np.concatenate(offsets)) if max_offset is None else max_offset
        metric = np.where(np.array(outliers) < tol, max_offset, np.nan)
        tol_idx = np.nonzero(np.array(outliers) >= tol)[0]
        for idx in tol_idx:
            metric[idx] = DivergencePoint.calc_(times[idx], offsets[idx], refractor_velocities[idx], fb_metric,
                                                threshold_times, offset_step)
        return metric
        
    def plot_refractor_velocity(self, coords, ax, **kwargs):
        x_coord = kwargs.pop('map_data')[coords]
        ax.axvline(x=x_coord, color='k', linestyle='--')
        super().plot_refractor_velocity(coords, ax, **kwargs)
        
REFRACTOR_VELOCITY_QC_METRICS = [FirstBreaksOutliers, DivergencePoint]
        