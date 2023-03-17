import numpy as np
from scipy.signal import hilbert
from scipy.optimize import minimize

from ..metrics import Metric
from ..const import HDR_FIRST_BREAK
from ..utils import times_to_indices, get_first_defined

class RefractorVelocityMetric(Metric):
    views = ("plot_gather", "plot_refractor_velocity")

    def __init__(self, name=None, first_breaks_col=HDR_FIRST_BREAK, threshold_times=50, correct_uphole=True):
        super().__init__(name=name)
        self.survey = None
        self.field = None
        self.first_breaks_col = first_breaks_col
        self.threshold_times = threshold_times
        self.correct_uphole = correct_uphole
        
    def bind_context(self, metric_map, survey, field):
        _ = metric_map
        self.survey = survey
        self.field = field
        
    def calc(self, *args, **kwargs):
        """Calculate the metric. Must be overridden in child classes."""
        _ = args, kwargs
        raise NotImplementedError
        
    def __call__(self, *args, **kwargs):
        return np.mean(self.calc(*args, **kwargs))

    def plot_gather(self, coords, ax, index, sort_by=None, mask=True, top_header=True, **kwargs):
        _ = coords
        gather = self.survey.get_gather(index).copy()
        if sort_by is not None:
            gather = gather.sort(by=sort_by)
        event_headers = kwargs.pop('event_headers', {'headers': self.first_breaks_col})
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
        gather.plot(event_headers=event_headers, ax=ax, **kwargs)

    def plot_refractor_velocity(self, coords, ax, index, **kwargs):
        refractor_velocity = self.field(coords)
        gather = self.survey.get_gather(index)
        refractor_velocity.times = gather[self.first_breaks_col]
        if self.correct_uphole:
            refractor_velocity.times += gather["SourceUpholeTime"]

        refractor_velocity.max_offset = gather["offset"].max()
        refractor_velocity.offsets = gather['offset']
        refractor_velocity.plot(threshold_times=self.threshold_times, ax=ax, **kwargs)


class FirstBreaksOutliers(RefractorVelocityMetric):
    name = "first_breaks_outliers"
    vmin = 0
    vmax = 0.05
    is_lower_better = True

    def calc(self, gather, refractor_velocity):
        g = gather.copy()
        rv_times = refractor_velocity(gather['offset'])
        gather_times = g[self.first_breaks_col]
        if self.correct_uphole:
            gather_times += g["SourceUpholeTime"]
        return np.abs(rv_times - gather_times) > self.threshold_times
    
    def plot_gather(self, *args, **kwargs):
        super().plot_gather(*args, top_header=False, **kwargs)


class FirstBreaksAmplitudes(RefractorVelocityMetric):
    name = "first_breaks_amplitudes"
    vmin = 0
    vmax = 0.5
    is_lower_better = None

    def calc(self, gather, refractor_velocity):
        _ = refractor_velocity
        g = gather.copy()
        g.scale_maxabs()
        ix = times_to_indices(g[self.first_breaks_col], g.samples).astype(np.int64)
        res = g.data[range(len(ix)), ix]
        return res

    def __call__(self, *args, **kwargs):
        amps = self.calc(*args, **kwargs)
        return amps.mean()


class FirstBreaksPhases(RefractorVelocityMetric):
    name = 'first_breaks_phases'
    vmin = 0
    vmax = np.pi / 2
    is_lower_better = True

    def __init__(self, target='max', **kwargs):
        if isinstance(target, str):
            target = {'max': 0, 'min': np.pi, 'transition': np.pi / 2}[target]
        self.target = target
        super().__init__(**kwargs)

    def calc(self, gather, refractor_velocity):
        _ = refractor_velocity
        ix = times_to_indices(gather[self.first_breaks_col], gather.samples).astype(np.int64), 
        phases = hilbert(gather.data, axis=1)[range(len(ix)), ix]
        res = np.angle(phases).reshape(-1)
        return res

    def __call__(self, gather, refractor_velocity):
        phases = abs(self.calc(gather=gather, refractor_velocity=refractor_velocity))
        return np.mean(abs(phases - self.target))
    
    def plot_gather(self, coords, ax, index, sort_by=None, **kwargs):
        _ = coords
        gather = self.survey.get_gather(index).copy()
        if sort_by is not None:
            gather = gather.sort(by=sort_by)
        event_headers =  kwargs.pop('event_headers', {'headers': self.first_breaks_col})
        phases = self.calc(gather=gather, refractor_velocity=None)
        gather[self.name] = phases
        kwargs['top_header'] = self.name

        metric_values = abs(abs(phases) - self.target)
        mask_kwargs = kwargs.get('masks', {})
        mask_threshold = get_first_defined(mask_kwargs.get('threshold', None), self.vmax)
        mask_kwargs.update({'masks': metric_values,
                            "threshold": mask_threshold})
        kwargs['masks'] = mask_kwargs
        gather.plot(event_headers=event_headers, ax=ax, **kwargs)


class FirstBreaksCorrelations(RefractorVelocityMetric):
    name = "first_breaks_correlations"
    views = ("plot_gather", "plot_mean_hodograph")
    vmin = 0
    vmax = 1
    is_lower_better = False
    
    def __init__(self, win_size=20, **kwargs):
        self.win_size = win_size
        super().__init__(**kwargs)

    def calc(self, gather, refractor_velocity):
        _ = refractor_velocity
        g = gather.copy()
        g.scale_maxabs(clip=True)
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

    def plot_mean_hodograph(self, coords, ax, index, **kwargs):
        _ = coords
        gather = self.survey.get_gather(index)
        g = gather.copy()
        g.scale_maxabs(clip=True)
        ix = times_to_indices(g[self.first_breaks_col], g.samples).astype(np.int64)
        mean_cols = ix.reshape(-1, 1) + np.arange(-self.win_size // g.sample_rate, self.win_size // g.sample_rate).reshape(1, -1)
        mean_cols = np.clip(mean_cols, 0, g.data.shape[1] - 1).astype(np.int64)

        traces_windows = g.data[np.arange(len(g.data)).reshape(-1, 1), mean_cols]
        mean_trace = traces_windows.mean(axis=0)

        traces_centered = traces_windows - traces_windows.mean(axis=1).reshape(-1, 1)
        mean_trace_centered = (mean_trace - mean_trace.mean()).reshape(1, -1)
        g.data = mean_trace_centered

        y_min = mean_cols[0, :].min()
        g.plot(mode='wiggle', ax=ax, **kwargs)
        ax.set_xlabel('Amplitude')
        ax.set_xticks(ticks=[-1, 0, 1])
        ax.set_yticks(ticks=np.arange(self.win_size)[::5], labels=np.arange(self.win_size)[::5] + y_min)

class GeometryError(RefractorVelocityMetric):
    name = "geometry_error"
    min_value = 0
    is_lower_better = True
    
    def __init__(self, reg=0.01, **kwargs):
        self.reg = reg
        super().__init__(**kwargs)

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
        g = gather.copy()
        times = g[self.first_breaks_col]
        if self.correct_uphole:
            times += g["SourceUpholeTime"]
        rv_times = refractor_velocity(g['offset'])
        shot_coords = g[['SourceX', 'SourceY']]
        receiver_coords = g[['GroupX', 'GroupY']]
        diff = times - rv_times
        x, y = (receiver_coords - shot_coords).T
        azimuth = np.arctan2(y, x)
        params = self.fit(azimuth, diff)
        w0 = params[0]
        return abs(w0)
    
    def plot_gather(self, *args, **kwargs):
        super().plot_gather(*args, mask=False, top_header=False, **kwargs)

    def plot_refractor_velocity(self, coords, ax, index, **kwargs):
        gather = self.survey.get_gather(index).copy()
        rv = self.field(coords)
        shot_coords, receiver_coords = gather[['SourceX', 'SourceY']], gather[['GroupX', 'GroupY']]
        x, y = (receiver_coords - shot_coords).T
        times, rv_times = gather[self.first_breaks_col], rv(gather['offset'])
        if self.correct_uphole:
            times += gather["SourceUpholeTime"]
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
      
    def __init__(self, threshold_times=50, step=100, **kwargs):
        super().__init__(**kwargs)
        self.threshold_times = threshold_times
        self.step = step
        
    def bind_context(self, *args, **kwargs):
        super().bind_context(*args, **kwargs)
        self.vmax = self.survey['offset'].max() if self.survey is not None else None
        self.vmin = self.survey['offset'].min() if self.survey is not None else None

    def calc(self, gather, refractor_velocity):
        g = gather.copy()
        times = g[self.first_breaks_col]
        if self.correct_uphole:
            times += g["SourceUpholeTime"]
        offsets = g['offset']
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

    def plot_gather(self, *args, **kwargs):
        super().plot_gather(*args, mask=False, top_header=False, **kwargs)

    def plot_refractor_velocity(self, coords, ax, index, **kwargs):
        gather = self.survey.get_gather(index)
        rv = self.field(coords)
        divergence_offset = self.calc(gather, rv)
        ax.axvline(x=divergence_offset, color='k', linestyle='--')
        super().plot_refractor_velocity(coords, ax, index, **kwargs)

REFRACTOR_VELOCITY_QC_METRICS = [FirstBreaksOutliers, FirstBreaksAmplitudes, FirstBreaksPhases,
                                 FirstBreaksCorrelations, GeometryError, DivergencePoint]
