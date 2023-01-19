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
    headers = "offset"
    data = False

    def __init__(self, survey=None, field=None, coords_to_index=None,
                    first_breaks_col=HDR_FIRST_BREAK, threshold_times=50, **kwargs):
        super().__init__(**kwargs)
        self.survey = survey
        self.field = field
        self.coords_to_index = coords_to_index
        self.first_breaks_col = first_breaks_col
        self.threshold_times = threshold_times

    def _plot_gather(self, gather, ax, **kwargs):
        event_headers = kwargs.pop('event_headers', {'headers': self.first_breaks_col})
        gather.plot(event_headers=event_headers, ax=ax, **kwargs)
        
    def plot_gather(self, coords, ax, **kwargs):
        gather = self.survey.get_gather(self.coords_to_index[coords])
        self._plot_gather(gather, ax=ax, **kwargs)
        
    def plot_refractor_velocity(self, coords, ax, **kwargs):
        refractor_velocity = self.field(coords)
        if not refractor_velocity.is_fit:
            gather_headers = self.survey.get_gather(self.coords_to_index[coords])
            refractor_velocity.times = gather_headers[self.first_breaks_col]
            refractor_velocity.offsets = gather_headers['offset']
        refractor_velocity.plot(threshold_times=self.threshold_times, ax=ax, **kwargs)


class FirstBreaksOutliers(RefractorVelocityMetric):
    name = "first_breaks_outliers"
    vmin = 0
    vmax = 0.05
    is_lower_better = True

    @staticmethod
    @njit(nogil=True)    
    def calc(gather_data, gather_headers, headers_names, rv_times, sample_rate, kwargs):
        threshold_times = kwargs.get('threshold_times', 50)
        return (np.abs(rv_times - gather_headers[:, 0]) > threshold_times).astype(np.float32)


class FirstBreaksAmplitudes(RefractorVelocityMetric):
    name = "first_breaks_amplitudes"
    min_value = -3
    max_value = 3
    is_lower_better = None
    data = True
    
    @staticmethod
    @njit(nogil=True)
    def calc(gather_data, gather_headers, headers_names, rv_times, sample_rate, kwargs):
        times = gather_headers[:, 0]
        ix = (times / sample_rate).astype(np.int64)
        res = np.zeros(len(times), dtype=np.float32)
        for i in range(0, len(ix)):
            res[i] = gather_data[i, ix[i]]
        return res
    
    def plot_gather(self, coords, ax, **kwargs):
        gather = self.survey.get_gather(self.coords_to_index[coords])
        gather['amps'] = self.calc(gather_data=gather.data, gather_headers=gather[[self.first_breaks_col]], headers_names=None,
                                   sample_rate=gather.sample_rate,
                                   rv_times=None, kwargs=None)
        kwargs['top_header'] = 'amps'
        super()._plot_gather(gather, ax=ax, **kwargs)


class FirstBreaksPhases(RefractorVelocityMetric):
    name = 'first_breaks_phases'
    min_value = -np.pi
    max_value = np.pi
    is_lower_better = None
    data = True
    
    @staticmethod
    @njit(nogil=True)
    def calc(gather_data, gather_headers, headers_names, rv_times, sample_rate, kwargs):
        times = gather_headers[:, 0]
        ix = (times / sample_rate).astype(np.int64)
        with objmode(phases='complex128[:, :]'):
            phases = np.ascontiguousarray(hilbert(gather_data, axis=1))
        angles = np.angle(phases)
        res = np.zeros(len(gather_headers[:, 0]), dtype=np.float32)
        for i in range(0, len(ix)):
            res[i] = angles[i, ix[i]]
        return res

    def plot_gather(self, coords, ax, **kwargs):
        gather = self.survey.get_gather(self.coords_to_index[coords])
        rv = None
        gather['phase'] = self.calc(gather_data=gather.data, gather_headers=gather[[self.first_breaks_col]], headers_names=None,
                                    sample_rate=gather.sample_rate, rv_times=None, kwargs=None)
        kwargs['top_header'] = 'phase'
        super()._plot_gather(gather, ax=ax, **kwargs)


class FirstBreaksCorrelations(RefractorVelocityMetric):
    name = "first_breaks_correlations"
    min_value = -1
    max_value = 1
    is_lower_better = None
    data = True
    
    @staticmethod
    @njit(nogil=True)
    def calc(gather_data, gather_headers, headers_names, rv_times, sample_rate, kwargs):
        corr_window = int(kwargs.get('corr_window', 10))
        times = gather_headers[:, 0]
        ix = (times / sample_rate).astype(np.int64)
        mean_trace = np.zeros(corr_window * 2, dtype=np.float32)
        for trace, pick in zip(gather_data, ix):
            current = trace[max(0, pick - corr_window): min(pick + corr_window, gather_data.shape[1])]
            left_pad = max(corr_window - pick, 0)
            mean_trace[left_pad: left_pad + len(current)] += current
        mean_trace /= gather_data.shape[0]
        corrs = np.zeros(len(times), dtype=np.float32)

        i = 0
        for trace, pick in zip(gather_data, ix):
            current = trace[max(0, pick - corr_window): min(pick + corr_window, gather_data.shape[1])]
            left_pad = max(corr_window - pick, 0)
            current_padded = np.zeros(corr_window *2, dtype=np.float32)
            current_padded[left_pad: left_pad + len(current)] = current
            coef = np.corrcoef(current_padded, mean_trace)[0, 1]
            corrs[i] = coef
            i += 1
        return corrs
    
    def plot_gather(self, coords, ax, **kwargs):
        gather = self.survey.get_gather(self.coords_to_index[coords])
        rv = None
        calc_kwargs = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
        corr_window = self.corr_window if hasattr(self, 'corr_window') else 10
        calc_kwargs['corr_window'] = corr_window
        gather['corr'] = self.calc(gather_data=gather.data, gather_headers=gather[[self.first_breaks_col]], headers_names=None,
                                   sample_rate=gather.sample_rate, kwargs=calc_kwargs, rv_times=None)
        kwargs['top_header'] = 'corr'
        super()._plot_gather(gather, ax=ax, **kwargs)


class GeometryError(RefractorVelocityMetric):
    name = "geometry_error"
    min_value = 0
    is_lower_better = True
    headers = ("offset", "SourceX", "SourceY", "GroupX", "GroupY")
    data = False
    
    @staticmethod
    def sin(x, amp, phase):
        return amp * np.sin(x + phase)

    @staticmethod
    def loss(params, x, y, reg=0.01):
        return np.abs(y - GeometryError.sin(x, *params)).mean() + reg * params[0]**2

    @staticmethod
    def fit(azimuth, diff, reg=0.01):
        fit_result = minimize(GeometryError.loss, x0=[0, 0], args=(azimuth, diff - diff.mean(), reg),
                              bounds=((None, None), (-np.pi, np.pi)), method="Nelder-Mead", tol=1e-5)
        return fit_result.x

    
    @staticmethod
    @njit(nogil=True)
    def calc(gather_data, gather_headers, headers_names, rv_times, sample_rate, kwargs):
        reg = kwargs['reg'] if 'reg' in kwargs else 0.01 # int-float err
        times = gather_headers[:, 0]
        shot_coords = np.stack([gather_headers[:, headers_names.index('SourceX')], gather_headers[:, headers_names.index('SourceY')]]).T
        receiver_coords = np.stack([gather_headers[:, headers_names.index('GroupX')], gather_headers[:, headers_names.index('GroupY')]]).T
        diff = times - rv_times
        x, y = np.ascontiguousarray((receiver_coords - shot_coords).T)
        azimuth = np.arctan2(y, x)
        with objmode(w0='float32'):
            params = GeometryError.fit(azimuth, diff, reg=reg)
            w0 = params[0]
        return np.array(abs(w0), dtype=np.float32)

    def plot_refractor_velocity(self, coords, ax, **kwargs):
        gather = self.survey.get_gather(self.coords_to_index[coords])
        rv = self.field(coords)
        shot_coords, receiver_coords = gather['SourceX', 'SourceY'], gather['GroupX', 'GroupY']
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
    data = False
    
    def __init__(self, survey=None, **kwargs):
        super().__init__(survey, **kwargs)
        self.vmax = survey['offset'].max() if survey is not None else None
        self.vmin = survey['offset'].min() if survey is not None else None
    
    @staticmethod
    @njit(nogil=True)
    def calc(gather_data, gather_headers, headers_names, rv_times, sample_rate, kwargs):
        threshold_times = int(kwargs.get('threshold_times', 50))
        tol = kwargs['tol'] if 'tol' in kwargs else 0.03
        offset_step = int(kwargs.get('offset_step', 100))

        times = gather_headers[:, 0]
        offsets = gather_headers[:, headers_names.index('offset')]
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
        
        diffs = np.zeros(n, dtype=np.float32)
        for i in range(n):
            diffs[i] = outliers[i + n] - outliers[i]
        diffs_delta = np.zeros(n - 1, dtype=np.float32)
        for i in range(n - 1):
            diffs_delta[i] = diffs[i + 1] - diffs[i]
        return np.array(check_offsets[np.argmax(diffs_delta) + 1], dtype=np.float32)
        
    def plot_refractor_velocity(self, coords, ax, **kwargs):
        gather = self.survey.get_gather(self.coords_to_index[coords])
        rv = self.field(coords)
        tol = self.tol if hasattr(self, 'tol') else 0.03
        offset_step = self.offset_step if hasattr(self, 'offset_step') else 100
        threshold_times = self.threshold_times if hasattr(self, 'threshold_times') else 50
        
        calc_kwargs = Dict.empty(key_type=types.unicode_type, value_type=types.float32)
        calc_kwargs['tol'] = tol
        calc_kwargs['offset_step'] = offset_step
        calc_kwargs['threshold_times'] = threshold_times
        divergence_offset = self.calc(gather_data=gather.data, rv_times=rv(gather['offset']), gather_headers=gather[[self.first_breaks_col, 'offset']],
                                      headers_names=(self.firstBreaks_col, 'offset'),
                                     sample_rate=None,  kwargs=calc_kwargs)
        ax.axvline(x=divergence_offset, color='k', linestyle='--')
        super().plot_refractor_velocity(coords, ax=ax, **kwargs)
        
REFRACTOR_VELOCITY_QC_METRICS = [FirstBreaksOutliers, DivergencePoint, FirstBreaksAmplitudes, FirstBreaksPhases,
                                 FirstBreaksCorrelations, GeometryError]
