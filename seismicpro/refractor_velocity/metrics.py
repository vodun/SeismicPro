import numpy as np
from scipy.signal import hilbert
from scipy.optimize import minimize

from ..metrics import Metric
from ..const import HDR_FIRST_BREAK
from ..utils import times_to_indices

class RefractorVelocityMetric(Metric):
    views = ("plot_gather", "plot_refractor_velocity")

    def __init__(self, survey=None, field=None, first_breaks_col=HDR_FIRST_BREAK, target=None, **kwargs):
        super().__init__(**kwargs)
        self.survey = survey
        self.field = field
        self.first_breaks_col = first_breaks_col

    def _plot_gather(self, gather, ax, sort_by=None, event_headers=None, **kwargs):
        if sort_by is not None:
            gather = gather.sort(by=sort_by)
        if event_headers is None:
            event_headers = {'headers': self.first_breaks_col}
        gather.apply_agc(110).plot(event_headers=event_headers, ax=ax, **kwargs)

    def plot_gather(self, coords, ax, **kwargs):
        gather = self.survey.get_gather(coords)
        self._plot_gather(gather, ax=ax, **kwargs)

    def plot_refractor_velocity(self, coords, ax, **kwargs):
        refractor_velocity = self.field(coords)
        gather = self.survey.get_gather(coords)
        refractor_velocity.times = gather[self.first_breaks_col]
        refractor_velocity.offsets = gather['offset']
        threshold_times = getattr(self, 'threshold_times', 50)
        refractor_velocity.plot(threshold_times=threshold_times, ax=ax, **kwargs)

class FirstBreaksOutliers(RefractorVelocityMetric):
    name = "first_breaks_outliers"
    vmin = 0
    vmax = 0.05
    is_lower_better = True

    @staticmethod
    def _calc(gather, refractor_velocity, first_breaks_col=HDR_FIRST_BREAK, threshold_times=50):
        rv_times = refractor_velocity(gather['offset'])
        return np.abs(rv_times - gather[first_breaks_col]) > threshold_times
    
    @classmethod
    def calc(cls, *args, **kwargs):
        return np.mean(cls._calc(*args, **kwargs))

class FirstBreaksAmplitudes(RefractorVelocityMetric):
    name = "first_breaks_amplitudes"
    vmin = -3
    vmax = 3
    is_lower_better = None

    @staticmethod
    def _calc(gather, refractor_velocity, first_breaks_col=HDR_FIRST_BREAK):
        _ = refractor_velocity
        g = gather.copy()
        g.scale_standard()
        ix = times_to_indices(g[first_breaks_col], g.samples).astype(np.int64)
        res = g.data[range(len(ix)), ix]
        return res

    @classmethod
    def calc(cls, target=0, *args, **kwargs):
        amps = cls._calc(*args, **kwargs)
        return np.mean(abs(amps - target))

    def plot_gather(self, coords, ax, **kwargs):
        gather = self.survey.get_gather(coords)
        gather['amps'] = self._calc(gather=gather, first_breaks_col=self.first_breaks_col, refractor_velocity=None)
        kwargs['top_header'] = 'amps'
        target = getattr(self, 'target', 0)
        cbar_std = max(abs(gather['amps'] - target))
        super()._plot_gather(gather, ax=ax,  **kwargs)

class FirstBreaksPhases(RefractorVelocityMetric):
    name = 'first_breaks_phases'
    min_value = -np.pi
    max_value = np.pi
    is_lower_better = None

    @staticmethod
    def calc(gather, refractor_velocity, first_breaks_col=HDR_FIRST_BREAK):
        _ = refractor_velocity
        g = gather.copy()
        g.scale_standard()
        ix = times_to_indices(g[first_breaks_col], g.samples).astype(np.int64)
        phases = hilbert(g.data, axis=1)[range(len(ix)), ix]
        res = np.angle(phases)
        return res

    def plot_gather(self, coords, ax, **kwargs):
        gather = self.survey.get_gather(coords)
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
    def calc(gather, refractor_velocity, first_breaks_col=HDR_FIRST_BREAK, win_size=10):
        _ = refractor_velocity
        g = gather.copy()
        g.scale_standard()
        ix = times_to_indices(g[first_breaks_col], g.samples).astype(np.int64)
        mean_cols = ix.reshape(-1, 1) + np.arange(-win_size // g.sample_rate, win_size // g.sample_rate).reshape(1, -1)
        mean_cols = np.clip(mean_cols, 0, g.data.shape[1] - 1).astype(np.int64)

        traces_windows = g.data[np.arange(len(g.data)).reshape(-1, 1), mean_cols]
        mean_trace = traces_windows.mean(axis=0)

        traces_centered = traces_windows - traces_windows.mean(axis=1).reshape(-1, 1)
        mean_trace_centered = (mean_trace - mean_trace.mean()).reshape(1, -1)

        corrs = (traces_centered * mean_trace_centered).sum(axis=1)
        corrs /= np.sqrt((traces_centered**2).sum(axis=1) * (mean_trace_centered**2).sum(axis=1))
        return corrs

    def plot_gather(self, coords, ax, **kwargs):
        gather = self.survey.get_gather(coords)
        win_size = getattr(self, 'win_size', 10)
        gather['corr'] = self.calc(gather=gather, refractor_velocity=None, first_breaks_col=self.first_breaks_col,
                                   win_size=win_size)
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
        return np.array(abs(w0))

    def plot_refractor_velocity(self, coords, ax, **kwargs):
        gather = self.survey.get_gather(coords)
        rv = self.field(coords)
        shot_coords, receiver_coords = gather[['SourceX', 'SourceY']], gather[['GroupX', 'GroupY']]
        x, y = (receiver_coords - shot_coords).T
        times, rv_times = gather[self.first_breaks_col], rv(gather['offset'])
        diff = times - rv_times
        reg = getattr(self, 'reg', 0.01)
        azimuth = np.arctan2(y, x)
        params = self.fit(azimuth, diff, reg=reg)
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

    @staticmethod
    def calc(gather, refractor_velocity, first_breaks_col=HDR_FIRST_BREAK, threshold_times=50, step=100):
        times = gather[first_breaks_col]
        offsets = gather['offset']
        rv_times = refractor_velocity(offsets)
        outliers = np.abs(rv_times - times) > threshold_times
        if np.isclose(np.mean(outliers), 0) or step >= len(offsets) - len(offsets) % step:
            return np.array(max(offsets))

        sorted_offsets_idx = np.argsort(offsets)
        offsets = offsets[sorted_offsets_idx]
        outliers = outliers[sorted_offsets_idx]

        split_idxs = np.arange(step, len(offsets) - len(offsets) % step, step)
        outliers_splits = np.split(outliers, split_idxs)

        outliers_fractions = [outliers_window.mean() for outliers_window in outliers_splits]
        return np.array(offsets[split_idxs[np.argmax(outliers_fractions) - 1]])

    def plot_refractor_velocity(self, coords, ax, **kwargs):
        gather = self.survey.get_gather(coords)
        rv = self.field(coords)
        step = getattr(self, 'step', 100)
        threshold_times = getattr(self, 'threshold_times', 50)

        divergence_offset = self.calc(gather=gather, refractor_velocity=rv, first_breaks_col=self.first_breaks_col,
                                      threshold_times=threshold_times, step=step)
        ax.axvline(x=divergence_offset, color='k', linestyle='--')
        super().plot_refractor_velocity(coords, ax=ax, threshold_times=threshold_times, **kwargs)

REFRACTOR_VELOCITY_QC_METRICS = [FirstBreaksOutliers, FirstBreaksAmplitudes, FirstBreaksPhases,
                                 FirstBreaksCorrelations, GeometryError, DivergencePoint]
