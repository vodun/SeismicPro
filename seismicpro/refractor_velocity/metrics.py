import numpy as np
from numba import njit, prange, objmode
from numba.core import types
from numba.typed import Dict
from matplotlib import patches
from scipy.signal import hilbert
from scipy.optimize import minimize
import torch

from ..metrics import Metric, ScatterMapPlot, MetricMap
from ..const import HDR_FIRST_BREAK


class RefractorVelocityMetric(Metric):
    views = ("plot_gather", "plot_refractor_velocity")

    def __init__(self, survey=None, field=None, first_breaks_col=HDR_FIRST_BREAK, **kwargs):
        super().__init__(**kwargs)
        self.survey = survey
        self.field = field
        self.first_breaks_col = first_breaks_col

    def _plot_gather(self, gather, ax, sort_by=None, event_headers=None, **kwargs):
        if sort_by is not None:
            gather = gather.sort(by=sort_by)
        if event_headers is None:
            event_headers = {'headers': self.first_breaks_col}
        gather.plot(event_headers=event_headers, ax=ax, **kwargs)
        
    def plot_gather(self, coords, ax, **kwargs):
        gather = self.survey.get_gather(self.field._coords_to_index[coords])
        self._plot_gather(gather, ax=ax, **kwargs)
        
    def plot_refractor_velocity(self, coords, ax, threshold_times=50, **kwargs):
        refractor_velocity = self.field(coords)
        if not refractor_velocity.is_fit:
            gather = self.survey.get_gather(self.field._coords_to_index[coords])
            refractor_velocity.times = gather[self.first_breaks_col]
            refractor_velocity.offsets = gather['offset']
        refractor_velocity.plot(threshold_times=threshold_times, ax=ax, **kwargs)


class FirstBreaksOutliers(RefractorVelocityMetric):
    name = "first_breaks_outliers"
    vmin = 0
    vmax = 0.05
    is_lower_better = True

    @staticmethod    
    def calc(gather, refractor_velocity, first_breaks_col=HDR_FIRST_BREAK, threshold_times=50):
        rv_times = refractor_velocity(gather['offset'])
        return np.abs(rv_times - gather[first_breaks_col]) > threshold_times


class FirstBreaksAmplitudes(RefractorVelocityMetric):
    name = "first_breaks_amplitudes"
    min_value = -3
    max_value = 3
    is_lower_better = None
    
    @staticmethod
    def calc(gather, refractor_velocity, first_breaks_col=HDR_FIRST_BREAK):
        _ = refractor_velocity
        times = gather[first_breaks_col]
        ix = (times / gather.sample_rate).astype(np.int64)
        res = gather.data[range(len(ix)), ix]
        return res
    
    def plot_gather(self, coords, ax, **kwargs):
        gather = self.survey.get_gather(self.field._coords_to_index[coords]).scale_standard()
        gather['amps'] = self.calc(gather=gather, first_breaks_col=self.first_breaks_col, refractor_velocity=None)
        kwargs['top_header'] = 'amps'
        super()._plot_gather(gather, ax=ax, **kwargs)


class FirstBreaksPhases(RefractorVelocityMetric):
    name = 'first_breaks_phases'
    min_value = -np.pi
    max_value = np.pi
    is_lower_better = None
    
    @staticmethod
    def calc(gather, refractor_velocity, first_breaks_col=HDR_FIRST_BREAK):
        _ = refractor_velocity
        times = gather[first_breaks_col]
        ix = (times / gather.sample_rate).astype(np.int64)
        phases = hilbert(gather.data, axis=1)
        angles = np.angle(phases)
        res = angles[range(len(ix)), ix]
        return res

    def plot_gather(self, coords, ax, **kwargs):
        gather = self.survey.get_gather(self.field._coords_to_index[coords]).scale_standard()
        gather['phase'] = self.calc(gather=gather, first_breaks_col=self.first_breaks_col, refractor_velocity=None)
        kwargs['top_header'] = 'phase'
        super()._plot_gather(gather, ax=ax, **kwargs)
        return ax


class FirstBreaksCorrelations(RefractorVelocityMetric):
    name = "first_breaks_correlations"
    min_value = -1
    max_value = 1
    is_lower_better = False
    
    @staticmethod
    def calc(gather, refractor_velocity, first_breaks_col=HDR_FIRST_BREAK, corr_window=10):
        _ = refractor_velocity
        times = gather[first_breaks_col]
        ix = (times / gather.sample_rate).astype(np.int64)
        mean_cols = np.linspace(ix - corr_window, ix + corr_window, num=corr_window * 2, dtype=np.int64).T

        traces_windows = gather.data[np.arange(len(gather.data)).reshape(-1, 1), mean_cols]
        mean_trace = traces_windows.mean(axis=0)

        traces_centered = traces_windows - traces_windows.mean(axis=1).reshape(-1, 1)
        mean_trace_centered = (mean_trace - mean_trace.mean()).reshape(1, -1)

        corrs = (traces_centered * mean_trace_centered).sum(axis=1)
        corrs /= np.sqrt((traces_centered**2).sum(axis=1) * (mean_trace_centered**2).sum(axis=1))
        return corrs
    
    def plot_gather(self, coords, ax, **kwargs):
        gather = self.survey.get_gather(self.field._coords_to_index[coords]).scale_standard()
        corr_window = self.corr_window if hasattr(self, 'corr_window') else 10
        gather['corr'] = self.calc(gather=gather, refractor_velocity=None, first_breaks_col=self.first_breaks_col,
                                   corr_window=corr_window)
        kwargs['top_header'] = 'corr'
        super()._plot_gather(gather, ax=ax, **kwargs)


class GeometryError(RefractorVelocityMetric):
    name = "geometry_error"
    min_value = 0
    is_lower_better = True
    
    @staticmethod
    def sin(x, amp, phase):
        return amp * np.sin(x + phase)

    @classmethod
    def loss(cls, params, x, y, reg=0.01):
        return np.abs(y - cls.sin(x, *params)).mean() + reg * params[0]**2

    @classmethod
    def fit(cls, azimuth, diff, reg=0.01):
        fit_result = minimize(cls.loss, x0=[0, 0], args=(azimuth, diff - diff.mean(), reg),
                              bounds=((None, None), (-np.pi, np.pi)), method="Nelder-Mead", tol=1e-5)
        return fit_result.x

    
    @classmethod
    def calc(cls, gather, refractor_velocity, first_breaks_col=HDR_FIRST_BREAK, reg=0.01):
        times = gather[first_breaks_col]
        rv_times = refractor_velocity(gather['offset'])
        shot_coords = gather[['SourceX', 'SourceY']]
        receiver_coords = gather[['GroupX', 'GroupY']]
        diff = times - rv_times
        x, y = (receiver_coords - shot_coords).T
        azimuth = np.arctan2(y, x)
        params = cls.fit(azimuth, diff, reg=reg)
        w0 = params[0]
        return np.array(abs(w0), dtype=np.float32)

    def plot_refractor_velocity(self, coords, ax, **kwargs):
        gather = self.survey.get_gather(self.field._coords_to_index[coords])
        rv = self.field(coords)
        shot_coords, receiver_coords = gather[['SourceX', 'SourceY']], gather[['GroupX', 'GroupY']]
        x, y = (receiver_coords - shot_coords).T
        times, rv_times = gather[self.first_breaks_col], rv(gather['offset'])
        diff = times - rv_times
        reg = self.reg if hasattr(self, 'reg') else 0.01
        azimuth = np.arctan2(y, x)
        params = self.fit(azimuth, diff, reg=reg)
        ax.scatter(azimuth, diff)
        x = np.linspace(-np.pi, np.pi, 100)
        ax.plot(x, params[0] * np.sin(x + params[1]), c='r')


class DivergencePoint(RefractorVelocityMetric):
    """Find offset after that first breaks are most likely to significantly diverge from expected time.
    Such an offset is found as one with the maximum increase of outliers after it(actually, maximum increase of 
    increase of outliers, as it seems to be more robust).
    Divergence point is set to be the maximum offset when the overall fraction of outliers does not exceeds given tolerance.
    """
    name = "divergence_point"
    min_value = 0
    is_lower_better = False
    
    def __init__(self, survey=None, **kwargs):
        super().__init__(survey, **kwargs)
        self.vmax = survey['offset'].max() if survey is not None else None
        self.vmin = survey['offset'].min() if survey is not None else None
    
    @staticmethod
    def calc(gather, refractor_velocity, first_breaks_col=HDR_FIRST_BREAK, threshold_times=50,
             tol=0.03, offset_step=100):
        times = gather[first_breaks_col]
        offsets = gather['offset']
        rv_times = refractor_velocity(offsets)
        outliers = np.mean((np.abs(rv_times - times) > threshold_times))
        if outliers <= tol:
            return np.array(max(offsets), dtype=np.float32)
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
        rv_windows = [rv_times[:idx] for idx in check_indices] + [rv_times[idx:] for idx in check_indices]
        outliers = [np.mean(np.abs(rv_window - time_window) > threshold_times) 
                    for time_window, rv_window in zip(time_windows, rv_windows)]

        diffs = [outliers[i + n] - outliers[i] for i in range(n)]
        diffs_delta = [diffs[i + 1] - diffs[i] for i in range(n - 1)]
        return np.array(check_offsets[np.argmax(diffs_delta) + 1], dtype=np.float32)
        
    def plot_refractor_velocity(self, coords, ax, **kwargs):
        gather = self.survey.get_gather(self.field._coords_to_index[coords])
        rv = self.field(coords)
        tol = self.tol if hasattr(self, 'tol') else 0.03
        offset_step = self.offset_step if hasattr(self, 'offset_step') else 100
        threshold_times = self.threshold_times if hasattr(self, 'threshold_times') else 50

        divergence_offset = self.calc(gather=gather, refractor_velocity=rv, first_breaks_col=self.first_breaks_col,
                                      threshold_times=threshold_times, offset_step=offset_step, tol=tol)
        ax.axvline(x=divergence_offset, color='k', linestyle='--')
        super().plot_refractor_velocity(coords, ax=ax, **kwargs)
        
REFRACTOR_VELOCITY_QC_METRICS = [FirstBreaksOutliers, FirstBreaksAmplitudes, FirstBreaksPhases,
                                 FirstBreaksCorrelations, GeometryError, DivergencePoint]
