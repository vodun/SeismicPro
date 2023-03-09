import numpy as np
from scipy.signal import hilbert
from scipy.optimize import minimize

from ..metrics import Metric
from ..const import HDR_FIRST_BREAK
from ..utils import times_to_indices, get_first_defined

class RefractorVelocityMetric(Metric):
    views = ("plot_gather", "plot_refractor_velocity")

    def __init__(self, name=None):
        super().__init__(name=name)
        self.survey = None
        self.field = None
        self.first_breaks_col = None
        self.threshold_times = None
        
    def bind_context(self, metric_map, survey, field, first_breaks_col, threshold_times=50):
        _ = metric_map
        self.survey = survey
        self.field = field
        self.first_breaks_col = first_breaks_col
        self.threshold_times = threshold_times
        
    def __call__(self, *args, **kwargs):
        return np.mean(self.calc(*args, **kwargs))

    def plot_gather(self, coords, ax, sort_by=None, event_headers=None, mask=True, top_header=True, **kwargs):
        gather = self.survey.get_gather(coords)
        if sort_by is not None:
            gather = gather.sort(by=sort_by)
        if event_headers is None:
            event_headers = {'headers': self.first_breaks_col}
        if top_header or mask:
            refractor_velocity = self.field(gather.coords)
            metric_values = self.calc(gather=gather, refractor_velocity=refractor_velocity)
            if mask:
                mask_kwargs = kwargs.get('masks', {})
                invert_mask = -1 if self.is_lower_better is False else 1
                mask_threshold = get_first_defined(mask_kwargs.get('threshold', None),
                                                   self.vmax if invert_mask==1 else self.vmin,
                                                   metric_values.mean())
                mask_kwargs.update({'masks': metric_values * invert_mask,
                                    "threshold": mask_threshold})
                kwargs['masks'] = mask_kwargs
            if top_header:
                gather[self.name] = metric_values
                kwargs['top_header'] = self.name
        kwargs.pop('index')
        gather.plot(event_headers=event_headers, ax=ax, **kwargs)

    def plot_refractor_velocity(self, coords, ax, **kwargs):
        refractor_velocity = self.field(coords)
        gather = self.survey.get_gather(coords)
        refractor_velocity.times = gather[self.first_breaks_col]
        refractor_velocity.offsets = gather['offset']
        kwargs.pop('index')
        refractor_velocity.plot(threshold_times=self.threshold_times, ax=ax, **kwargs)


class FirstBreaksOutliers(RefractorVelocityMetric):
    name = "first_breaks_outliers"
    vmin = 0
    vmax = 0.05
    is_lower_better = True

    def calc(self, gather, refractor_velocity):
        rv_times = refractor_velocity(gather['offset'])
        return np.abs(rv_times - gather[self.first_breaks_col]) > self.threshold_times
    
    def plot_gather(self, coords, ax, *args, **kwargs):
        super().plot_gather(coords, ax, *args, top_header=False, **kwargs)


class FirstBreaksAmplitudes(RefractorVelocityMetric):
    name = "first_breaks_amplitudes"
    vmin = 0
    vmax = 0.5
    is_lower_better = True
    
    def bind_context(self, target=0, **context):
        self.target = target
        super().bind_context(**context)

    def calc(self, gather, refractor_velocity):
        _ = refractor_velocity
        g = gather.copy()
        g.scale_maxabs()
        ix = times_to_indices(g[self.first_breaks_col], g.samples).astype(np.int64)
        res = g.data[range(len(ix)), ix]
        return res

    def __call__(self, *args, **kwargs):
        amps = self.calc(*args, **kwargs)
        return np.mean(abs((amps - self.target)))


class FirstBreaksPhases(RefractorVelocityMetric):
    name = 'first_breaks_phases'
    vmin = 0
    vmax = np.pi / 2
    is_lower_better = True

    def bind_context(self, target='max', **context):
        if isinstance(target, str):
            target = {'max': 0, 'min': np.pi, 'transition': np.pi / 2}[target]
        self.target = target
        super().bind_context(**context)

    def calc(self, gather, refractor_velocity):
        _ = refractor_velocity
        g = gather.copy()
        ix = times_to_indices(g[self.first_breaks_col], g.samples).astype(np.int64)
        phases = hilbert(g.data, axis=1)[range(len(ix)), ix]
        res = abs(np.angle(phases))
        return res

    def __call__(self, *args, **kwargs):
        phases = self.calc(*args, **kwargs)
        return np.mean(abs((phases - self.target)))


class FirstBreaksCorrelations(RefractorVelocityMetric):
    name = "first_breaks_correlations"
    views = ("plot_gather", "plot_mean_hodograph")
    vmin = 0
    vmax = 1
    is_lower_better = False
    
    def bind_context(self, win_size=20, **context):
        self.win_size = win_size
        super().bind_context(**context)

    def calc(self, gather, refractor_velocity):
        _ = refractor_velocity
        g = gather.copy()
        g.scale_standard()
        ix = times_to_indices(g[self.first_breaks_col], g.samples).astype(np.int64)
        mean_cols = ix.reshape(-1, 1) + np.arange(-self.win_size // g.sample_rate, self.win_size // g.sample_rate).reshape(1, -1)
        mean_cols = np.clip(mean_cols, 0, g.data.shape[1] - 1).astype(np.int64)

        traces_windows = g.data[np.arange(len(g.data)).reshape(-1, 1), mean_cols]
        mean_trace = traces_windows.mean(axis=0)

        traces_centered = traces_windows - traces_windows.mean(axis=1).reshape(-1, 1)
        mean_trace_centered = (mean_trace - mean_trace.mean()).reshape(1, -1)

        corrs = (traces_centered * mean_trace_centered).sum(axis=1)
        corrs /= np.sqrt((traces_centered**2).sum(axis=1) * (mean_trace_centered**2).sum(axis=1))
        return corrs

    def plot_mean_hodograph(self, coords, ax, **kwargs):
        gather = self.survey.get_gather(coords)
        g = gather.copy()
        g.scale_standard()
        ix = times_to_indices(g[self.first_breaks_col], g.samples).astype(np.int64)
        mean_cols = ix.reshape(-1, 1) + np.arange(-self.win_size // g.sample_rate, self.win_size // g.sample_rate).reshape(1, -1)
        mean_cols = np.clip(mean_cols, 0, g.data.shape[1] - 1).astype(np.int64)

        traces_windows = g.data[np.arange(len(g.data)).reshape(-1, 1), mean_cols]
        mean_trace = traces_windows.mean(axis=0)

        traces_centered = traces_windows - traces_windows.mean(axis=1).reshape(-1, 1)
        mean_trace_centered = (mean_trace - mean_trace.mean()).reshape(1, -1)
        g.data = mean_trace_centered
        kwargs.pop('index')
        ax.get_xaxis().set_visible(False)
        g.plot(mode='wiggle', ax=ax, **kwargs)


class GeometryError(RefractorVelocityMetric):
    name = "geometry_error"
    min_value = 0
    is_lower_better = True
    
    def bind_context(self, reg=0.01, **context):
        self.reg = reg
        super().bind_context(**context)

    @staticmethod
    def sin(x, amp, phase):
        return amp * np.sin(x + phase)

    def loss(self, params, x, y):
        return np.abs(y - self.sin(x, *params)).mean() + self.reg * params[0]**2

    def fit(self, azimuth, diff):
        fit_result = minimize(self.loss, x0=[0, 0], args=(azimuth, diff - diff.mean()),
                              bounds=((None, None), (-np.pi, np.pi)), method="Nelder-Mead", tol=1e-5)
        return fit_result.x

    def calc(self, gather, refractor_velocity):
        times = gather[self.first_breaks_col]
        rv_times = refractor_velocity(gather['offset'])
        shot_coords = gather[['SourceX', 'SourceY']]
        receiver_coords = gather[['GroupX', 'GroupY']]
        diff = times - rv_times
        x, y = (receiver_coords - shot_coords).T
        azimuth = np.arctan2(y, x)
        params = self.fit(azimuth, diff)
        w0 = params[0]
        return np.array(abs(w0))
    
    def plot_gather(self, coords, ax, *args, **kwargs):
        super().plot_gather(coords, ax, *args, mask=False, top_header=False, **kwargs)

    def plot_refractor_velocity(self, coords, ax, **kwargs):
        gather = self.survey.get_gather(coords)
        rv = self.field(coords)
        shot_coords, receiver_coords = gather[['SourceX', 'SourceY']], gather[['GroupX', 'GroupY']]
        x, y = (receiver_coords - shot_coords).T
        times, rv_times = gather[self.first_breaks_col], rv(gather['offset'])
        diff = times - rv_times
        azimuth = np.arctan2(y, x)
        params = self.fit(azimuth, diff)
        ax.scatter(azimuth, diff)
        x = np.linspace(-np.pi, np.pi, 100)
        ax.plot(x, params[0] * np.sin(x + params[1]), c='r')


class DivergencePoint(RefractorVelocityMetric):
    """Find offset after that first breaks are most likely to diverge from expected time.
    Such an offset is defined as one with the maximum outliers in window of `step` times after it.
    Outliers are points that deviate from expected time more than `threshold_times` ms., see `FirstBreaksOutliers`.
    Divergence point is set to be the maximum offset when the overall fraction of outliers is zero.
    """
    name = "divergence_point"
    is_lower_better = False

    def __init__(self, survey=None, **kwargs):
        super().__init__(survey, **kwargs)
        self.vmax = survey['offset'].max() if survey is not None else None
        self.vmin = survey['offset'].min() if survey is not None else None
        
    def bind_context(self, threshold_times=50, step=100, **context):
        self.threshold_times = threshold_times
        self.step = step
        super().bind_context(**context)

    def calc(self, gather, refractor_velocity):
        times = gather[self.first_breaks_col]
        offsets = gather['offset']
        rv_times = refractor_velocity(offsets)
        outliers = np.abs(rv_times - times) > self.threshold_times
        if np.isclose(np.mean(outliers), 0) or self.step >= len(offsets) - len(offsets) % self.step:
            return np.array(max(offsets))

        sorted_offsets_idx = np.argsort(offsets)
        offsets = offsets[sorted_offsets_idx]
        outliers = outliers[sorted_offsets_idx]

        split_idxs = np.arange(self.step, len(offsets) - len(offsets) % self.step, self.step)
        outliers_splits = np.split(outliers, split_idxs)

        outliers_fractions = [outliers_window.mean() for outliers_window in outliers_splits]
        return np.array(offsets[split_idxs[np.argmax(outliers_fractions) - 1]])

    def plot_gather(self, coords, ax, *args, **kwargs):
        super().plot_gather(coords, ax, *args, mask=False, top_header=False, **kwargs)

    def plot_refractor_velocity(self, coords, ax, **kwargs):
        gather = self.survey.get_gather(coords)
        rv = self.field(coords)
        divergence_offset = self.calc(gather, rv)
        ax.axvline(x=divergence_offset, color='k', linestyle='--')
        super().plot_refractor_velocity(coords, ax=ax, **kwargs)

REFRACTOR_VELOCITY_QC_METRICS = [FirstBreaksOutliers, FirstBreaksAmplitudes, FirstBreaksPhases,
                                 FirstBreaksCorrelations, GeometryError, DivergencePoint]
