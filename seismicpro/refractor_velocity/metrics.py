"""Implements metrics for quality control of refractor velocity field, particularly focused on estimated times of first breaks examination."""
import numpy as np
from scipy.signal import hilbert
from scipy.optimize import minimize

from ..metrics import Metric
from ..const import HDR_FIRST_BREAK
from ..utils import times_to_indices, get_first_defined

class RefractorVelocityMetric(Metric):
    """Base metric class for quality control of refractor velocity field.
    In most cases, implements the following logic: `calc` method returns either iterable of trase-wise metric values
    or single metric value for the whole gather. Then `__call__` is used for gather-wise metric aggregation.
    `plot_gather` view is adjustable for plotting metric values on top of gather plot if `calc` implements trase-wise calculation.
    Parameters needed for metric calculation and view plotting should be set as attributes, e.g. `first_breaks_col`.
    
    Parameters
    ----------
    first_breaks_col : str, optional, defaults to :const:`~const.HDR_FIRST_BREAK`.
        Column name in context survey where times of first breaks are stored.
    threshold_times: float, optional, defaults to 50.
        Threshold for the refractor_velocity plot. Measured in milliseconds.
    correct_uphole : bool, optional.
        Whether to perform uphole correction in refractor_velocity plot.
    """
    views = ("plot_gather", "plot_refractor_velocity")

    def __init__(self, name=None, first_breaks_col=HDR_FIRST_BREAK, threshold_times=50, correct_uphole=None):
        super().__init__(name=name)
        self.first_breaks_col = first_breaks_col
        self.threshold_times = threshold_times
        self.correct_uphole = correct_uphole
        # Attributes set after context binding
        self.survey = None
        self.field = None
        
    def bind_context(self, metric_map, survey, field):
        """Process metric evaluation context, needed for interactive plotting: namely survey and field instances."""
        _ = metric_map
        self.survey = survey
        self.field = field
        if self.correct_uphole is None:
            self.correct_uphole = self.survey.is_uphole and self.field.is_uphole_corrected
        
    def calc(self, *args, **kwargs):
        """Calculate the metric. Must be overridden in child classes."""
        _ = args, kwargs
        raise NotImplementedError
        
    def __call__(self, *args, **kwargs):
        """Aggregate the metric. If not overriden, takes mean value of `calc`."""
        return np.mean(self.calc(*args, **kwargs))

    def plot_gather(self, coords, ax, index, sort_by=None, mask=True, top_header=True, **kwargs):
        """Base view for gather plotting. Plot the gather by its index in bounded survey and its first breaks.
        By default also recalculates metric in order to display `top_header` with metric values above gather traces
        and mark trases with metric greater than threshold. Threshold is either aquired from `kwargs` if given,
        or metric's colorbar margin if defined, or simply by mean value.
        
        Parameters
        ----------
        sort_by : str or iterable of str, optional.
            Headers names to sort the gather by.
        mask : bool, optional, defaults to True.
            Whether to plot mask defined by metric values on top of the gather plot.
        top_header : bool, optional, defaults to True.
            Whether to show a header with metric values above the gather plot.
        kwargs : misc, optional.
            Additional keyword arguments to `gather.plot`
        """
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
        """Plot the refractor velocity curve and show the threshold area used for metric calculation."""
        refractor_velocity = self.field(coords)
        gather = self.survey.get_gather(index)
        refractor_velocity.times = gather[self.first_breaks_col]
        if self.correct_uphole:
            refractor_velocity.times += gather["SourceUpholeTime"]
        refractor_velocity.max_offset = gather["offset"].max()
        refractor_velocity.offsets = gather['offset']
        refractor_velocity.plot(threshold_times=self.threshold_times, ax=ax, **kwargs)


class FirstBreaksOutliers(RefractorVelocityMetric):
    """The first break outliers metric.
    A first break time is considered to be an outlier if it differs from the expected arrival time defined by
    an offset-traveltime curve by more than a given threshold. Returns the fraction of outliers in the gather.
    
    Parameters
    ----------
    first_breaks_col : str, optional, defaults to :const:`~const.HDR_FIRST_BREAK`.
        Column name from `gather.headers` where times of first breaks are stored.
    threshold_times: float, optional, defaults to 50.
        Threshold for the first breaks outliers metric calculation. Measured in milliseconds.
    correct_uphole : bool, optional
        Whether to perform uphole correction by adding values of "SourceUpholeTime" header to times of first breaks.
    """
    name = "first_breaks_outliers"
    vmin = 0
    vmax = 0.05
    is_lower_better = True

    def calc(self, gather, refractor_velocity):
        """Calculate the first break outliers metric.
        Returns whether first break of each trace in the gather differs from those estimated by
        a near-surface velocity model by more than `threshold_times`.

        Parameters
        ----------
        gather : Gather
            A seismic gather to get offsets and times of first breaks from.
        refractor_velocity : RefractorVelocity
            Near-surface velocity model to estimate the expected first break times at `gather` offsets.
            
        Returns 
        -------
        metric : np.ndarray of bool.
            Array indicating whether each trace in the gather represents an outlier.
        """
        g = gather.copy()
        rv_times = refractor_velocity(gather['offset'])
        gather_times = g[self.first_breaks_col]
        correct_uphole = self.correct_uphole if self.correct_uphole is not None else (g.survey.is_uphole and
                                                                                      refractor_velocity.is_uphole_corrected)
        if correct_uphole:
            gather_times += g["SourceUpholeTime"]
        return np.abs(rv_times - gather_times) > self.threshold_times
    
    def plot_gather(self, *args, **kwargs):
        """Plot the gather with highlighted outliers on top of the gather plot."""
        super().plot_gather(*args, top_header=False, **kwargs)


class FirstBreaksAmplitudes(RefractorVelocityMetric):
    """Mean amplitude of the signal in the moment of first break after gather scaling."""
    name = "first_breaks_amplitudes"
    vmin = 0
    vmax = 0.5
    is_lower_better = None

    def calc(self, gather, refractor_velocity):
        """Return signal amplitudes at first break times.
        
        Returns
        -------
        metric : np.ndarray of float
            Signal amplitudes for each trace in the gather.
        """
        _ = refractor_velocity
        g = gather.copy()
        g.scale_maxabs()
        ix = times_to_indices(g[self.first_breaks_col], g.samples).astype(np.int64)
        res = g.data[range(len(ix)), ix]
        return res


class FirstBreaksPhases(RefractorVelocityMetric):
    """Mean absolute deviation of the signal phase from target value in the moment of first break.
    
    Parameters
    ----------
    target : float in range (-pi, pi] or str from {'max', 'min', 'transition'}.
        Target phase value in the moment of first break, see `np.angle`.
    """
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
        """Return signal phases at first break times.
        
        Returns
        -------
        metric : np.ndarray of float
            Signal phase value at first break time for each trace in the gather.
        """
        _ = refractor_velocity
        ix = times_to_indices(gather[self.first_breaks_col], gather.samples).astype(np.int64), 
        phases = hilbert(gather.data, axis=1)[range(len(ix)), ix]
        res = np.angle(phases).reshape(-1)
        return res

    def __call__(self, gather, refractor_velocity):
        """Return mean absolute deviation of the signal phase from target value in the moment of first break
        in the gather."""
        phases = abs(self.calc(gather=gather, refractor_velocity=refractor_velocity))
        return np.mean(abs(phases - self.target))
    
    def plot_gather(self, coords, ax, index, sort_by=None, **kwargs):
        """Plot the gather with phase values at first break times above the seismogram
        and highlight traces whose metric value differs from the target more than given threshold.
        """
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
    """Mean Pearson correlation coeffitient of trace with mean hodograph in window around the first break.
    
    Parameters
    ----------
    win_size : int, optional, defaults to 20.
        Size of the window to calculate the correlation coeffitient in. Measured in milliseconds.
    """
    name = "first_breaks_correlations"
    views = ("plot_gather", "plot_mean_hodograph")
    vmin = 0
    vmax = 1
    is_lower_better = False
    
    def __init__(self, win_size=20, **kwargs):
        self.win_size = win_size
        super().__init__(**kwargs)

    def calc(self, gather, refractor_velocity):
        """Return signal correlation with mean hodograph in the given window around first break times
        for a scaled gather.

        Returns
        -------
        metric : np.ndarray of float
            Window correlation with mean hodograph for each trace in the gather.
        """
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
        """Plot mean trace in the scaled gather around the first break with length of the given window size."""
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


class DivergencePoint(RefractorVelocityMetric):
    """The divergence point metric for first breaks.
    Find offset after that first breaks are most likely to diverge from expected time.
    Such an offset is defined as one with the maximum number of outliers in window of `step` times after it.
    
    Parameters
    ----------
    threshold_times: float, optional, defaults to 50.
        Threshold to define the first breaks outliers with, see `FirstBreaksOutliers`. Measured in milliseconds.
    step : int, optional, defaults to 100.
        Size of the offset window to count outliers in. Measured in meters.
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
        """Return the offset that defines a divergence point of first break times.

        Returns
        -------
        metric : int
            Metric value. Set to be the maximum offset when the overall fraction of outliers is close to zero.
        """
        g = gather.copy()
        times = g[self.first_breaks_col]
        correct_uphole = self.correct_uphole if self.correct_uphole is not None else (g.survey.is_uphole and
                                                                                      refractor_velocity.is_uphole_corrected)
        if correct_uphole:
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
        return offsets[split_idxs[np.argmax(outliers_fractions) - 1]]

    def plot_gather(self, *args, **kwargs):
        """Plot the gather and its first breaks."""
        super().plot_gather(*args, mask=False, top_header=False, **kwargs)

    def plot_refractor_velocity(self, coords, ax, index, **kwargs):
        """Plot the refractor velocity curve, show the divergence offset
         and threshold area used for metric calculation."""
        gather = self.survey.get_gather(index)
        rv = self.field(coords)
        divergence_offset = self.calc(gather, rv)
        ax.axvline(x=divergence_offset, color='k', linestyle='--')
        super().plot_refractor_velocity(coords, ax, index, **kwargs)

REFRACTOR_VELOCITY_QC_METRICS = [FirstBreaksOutliers, FirstBreaksAmplitudes, FirstBreaksPhases,
                                 FirstBreaksCorrelations, DivergencePoint]
