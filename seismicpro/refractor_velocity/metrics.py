"""Implements metrics for quality control of first breaks given the near-surface velocity model."""
from math import ceil, floor

import numpy as np
from numba import njit
from scipy.signal import hilbert

from ..gather.utils.normalization import scale_maxabs
from ..metrics import Metric
from ..utils import get_first_defined, to_list


class RefractorVelocityMetric(Metric):
    """Base metric class for quality control of refractor velocity field.
    Implements the following logic: `calc` method returns iterable of tracewise metric values,
    then `__call__` is used for gatherwise metric aggregation.
    `plot_gather` view is adjustable for plotting metric values on top of gather plot.
    Parameters needed for metric calculation and view plotting should be set as attributes, e.g. `first_breaks_col`.
    """

    views = ("plot_gather", "plot_refractor_velocity")

    def __init__(self, name=None):
        super().__init__(name=name)
        # Attributes set after context binding
        self.survey = None
        self.field = None
        self.first_breaks_col = None
        self.correct_uphole = None
        self.max_offset = None

    def bind_context(self, metric_map, survey, field, first_breaks_col, correct_uphole):
        """Process metric evaluation context."""
        _ = metric_map
        self.survey = survey
        self.field = field
        self.max_offset = survey["offset"].max()
        self.first_breaks_col = first_breaks_col
        if correct_uphole is None:
            self.correct_uphole = ("SourceUpholeTime" in self.survey.available_headers
                                    and self.field.is_uphole_corrected)
        else:
            self.correct_uphole = correct_uphole

    def calc(self, *args, **kwargs):
        """Calculate the metric. Must be overridden in child classes."""
        _ = args, kwargs
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """Aggregate the metric. If not overriden, takes mean value of `calc`."""
        return np.mean(self.calc(*args, **kwargs))

    def get_views(self, **kwargs):
        """Return plotters of the metric views and add kwargs for `gather_plot` to an interactive map plotter."""
        gather_plot_kwargs = {kwarg: kwargs.pop(kwarg) for kwarg in ["sort_by", "mask", "top_header"]
                              if kwarg in kwargs}
        plot_on_click_kwargs = kwargs.pop('plot_on_click_kwargs', [{}, {}])                      
        plot_on_click_kwargs[0].update(gather_plot_kwargs)
        kwargs['plot_on_click_kwargs'] = plot_on_click_kwargs
        return [getattr(self, view) for view in to_list(self.views)], kwargs

    def plot_gather(self, coords, ax, index, sort_by=None, mask=True, top_header=True, **kwargs):
        """Base view for gather plotting. Plot the gather by its index in bounded survey and its first breaks.
        By default also recalculates metric in order to display `top_header` with metric values above gather traces
        and mark traces with metric greater than threshold. Threshold is either aquired from `kwargs` if given,
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
        gather = self.survey.get_gather(index)
        if sort_by is not None:
            gather = gather.sort(by=sort_by)
        event_headers = kwargs.pop("event_headers", {"headers": self.first_breaks_col})
        if top_header or mask:
            refractor_velocity = self.field(gather.coords)
            metric_values = self.calc(gather=gather, refractor_velocity=refractor_velocity,
                                      first_breaks_col=self.first_breaks_col, correct_uphole=self.correct_uphole)
            if mask:
                mask_kwargs = kwargs.get("masks", {})
                invert_mask = -1 if self.is_lower_better is False else 1
                mask_threshold = get_first_defined(mask_kwargs.get("threshold", None),
                                                   self.vmax if invert_mask == 1 else self.vmin,
                                                   metric_values.mean())
                mask_kwargs.update({"masks": metric_values * invert_mask,
                                    "threshold": mask_threshold * invert_mask})
                kwargs["masks"] = mask_kwargs
            if top_header:
                kwargs["top_header"] = metric_values
        gather.plot(event_headers=event_headers, ax=ax, **kwargs)

    def plot_refractor_velocity(self, coords, ax, index, **kwargs):
        """Plot the refractor velocity curve."""
        refractor_velocity = self.field(coords)
        gather = self.survey.get_gather(index)
        refractor_velocity.times = gather[self.first_breaks_col]
        if self.correct_uphole:
            refractor_velocity.times = refractor_velocity.times + gather["SourceUpholeTime"]
        refractor_velocity.offsets = gather["offset"]
        refractor_velocity.plot(ax=ax, max_offset=self.max_offset, **kwargs)


class FirstBreaksOutliers(RefractorVelocityMetric):
    """The first break outliers metric.
    A first break time is considered to be an outlier if it differs from the expected arrival time defined by
    an offset-traveltime curve by more than a given threshold. Evaluates the fraction of outliers in the gather.

    Parameters
    ----------
    threshold_times: float, optional, defaults to 50.
        Threshold for the first breaks outliers metric calculation. Measured in milliseconds.
    """

    name = "first_breaks_outliers"
    vmin = 0
    vmax = 0.05
    is_lower_better = True

    def __init__(self, name=None, threshold_times=50):
        super().__init__(name)
        self.threshold_times = threshold_times

    @staticmethod
    @njit(nogil=True)
    def _calc(rv_times, picking_times, threshold_times):
        """Calculate the first break outliers."""
        return np.abs(rv_times - picking_times) > threshold_times

    def calc(self, gather, refractor_velocity, first_breaks_col, correct_uphole):
        """Calculate the first break outliers.
        Returns whether first break of each trace in the gather differs from those estimated by
        a near-surface velocity model by more than `threshold_times`.

        Parameters
        ----------
        gather : Gather
            A seismic gather to get offsets and times of first breaks from.
        refractor_velocity : RefractorVelocity
            Near-surface velocity model to estimate the expected first break times at `gather` offsets.
        first_breaks_col : str, optional, defaults to :const:`~const.HDR_FIRST_BREAK`
            Column name from `survey.headers` where times of first break are stored.
        correct_uphole : bool, optional
            Whether to perform uphole correction by adding values of "SourceUpholeTime" header to times of first breaks
            emulating the case when sources are located on the surface. If not given, correction is performed if
            "SourceUpholeTime" header is loaded.

        Returns
        -------
        metric : np.ndarray of bool.
            Array indicating whether each trace in the gather represents an outlier.
        """
        rv_times = refractor_velocity(gather["offset"])
        picking_times = gather[first_breaks_col]
        correct_uphole = (correct_uphole if correct_uphole is not None
                          else ("SourceUpholeTime" in gather.available_headers
                                and refractor_velocity.is_uphole_corrected))
        if correct_uphole:
            picking_times = picking_times + gather["SourceUpholeTime"]
        return self._calc(rv_times, picking_times, self.threshold_times)

    def plot_gather(self, *args, top_header=False, **kwargs):
        """Plot the gather with highlighted outliers on top of the gather plot."""
        super().plot_gather(*args, top_header=top_header, **kwargs)

    def plot_refractor_velocity(self, coords, ax, index, **kwargs):
        """Plot the refractor velocity curve and show the threshold area used for metric calculation."""
        threshold_times = kwargs.get("threshold_times", self.threshold_times)
        return super().plot_refractor_velocity(coords, ax, index, threshold_times=threshold_times, **kwargs)


class FirstBreaksAmplitudes(RefractorVelocityMetric):
    """Mean amplitude of the signal in the moment of first break after maxabs scaling."""

    name = "first_breaks_amplitudes"
    vmin = 0
    vmax = 0.5
    is_lower_better = None

    @staticmethod
    @njit(nogil=True)
    def _calc(gather_data, picking_times, sample_interval, start_time):
        """Calculate signal amplitudes at first break times."""
        gather_data = scale_maxabs(gather_data, min_value=None, max_value=None, q_min=0, q_max=1,
                                   clip=False, tracewise=True, eps=1e-10)
        ix = ((picking_times - start_time) // sample_interval).astype(np.int64)
        if np.any(gather_data.shape[1] < ix) or np.any(ix < 0):
            ix = np.clip(ix, 0, gather_data.shape[1] - 1)
            # warnings.warn("First breaks are out of bounds", RuntimeWarning)
        res = np.empty_like(ix, dtype=gather_data.dtype)
        for i, idx in enumerate(ix):
            prev_idx, next_idx = floor(idx), ceil(idx)
            weight = next_idx - idx
            res[i] = gather_data[i, prev_idx] * weight + gather_data[i, next_idx] * (1 - weight)
        return res

    def calc(self, gather, refractor_velocity, first_breaks_col, correct_uphole):
        """Return signal amplitudes at first break times.

        Parameters
        ----------
        gather : Gather
            A seismic gather to get offsets and times of first breaks from.
        first_breaks_col : str, optional, defaults to :const:`~const.HDR_FIRST_BREAK`
            Column name from `survey.headers` where times of first break are stored.

        Returns
        -------
        metric : np.ndarray of float
            Signal amplitudes for each trace in the gather.
        """
        _ = refractor_velocity, correct_uphole
        res = self._calc(gather.data, gather[first_breaks_col], gather.sample_interval, gather.times[0])
        return res


class FirstBreaksPhases(RefractorVelocityMetric):
    """Mean absolute deviation of the signal phase from target value in the moment of first break.

    Parameters
    ----------
    target : float in range (-pi, pi] or str from {'max', 'min', 'transition'}, optional, defaults to 'max'
        Target phase value in the moment of first break: 0, pi, pi / 2 for `max`, `min` and `transition` respectively.
    """

    name = "first_breaks_phases"
    vmin = 0
    vmax = np.pi / 2
    is_lower_better = True

    def __init__(self, target="max", **kwargs):
        if isinstance(target, str):
            if target not in {"max", "min", "transition"}:
                raise KeyError("target should be one of {'max', 'min', 'transition'} or float.")
            target = {"max": 0, "min": np.pi, "transition": np.pi / 2}[target]
        self.target = target
        super().__init__(**kwargs)

    def calc(self, gather, refractor_velocity, first_breaks_col, correct_uphole):
        """Return absolute deviation of the signal phase from target value in the moment of first break.

        Parameters
        ----------
        gather : Gather
            A seismic gather to get offsets and times of first breaks from.
        first_breaks_col : str, optional, defaults to :const:`~const.HDR_FIRST_BREAK`
            Column name from `survey.headers` where times of first break are stored.

        Returns
        -------
        metric : np.ndarray of float
            Signal phase value at first break time for each trace in the gather.
        """
        _ = refractor_velocity, correct_uphole
        ix = ((gather[first_breaks_col] - gather.times[0]) // gather.sample_interval).astype(np.int64)
        if np.any(gather.data.shape[1] < ix) or np.any(ix < 0):
            ix = np.clip(ix, 0, gather.data.shape[1] - 1)
            # warnings.warn("First breaks are out of bounds", RuntimeWarning)
        phases = hilbert(gather.data, axis=1)[range(len(ix)), ix]
        angles = np.angle(phases)
        # Map angles to range (target - pi, target + pi]
        if self.target > 0:
            angles = np.where(angles > self.target - np.pi, angles, angles + (2 * np.pi))
        else:
            angles = np.where(angles < self.target + np.pi, angles, angles - (2 * np.pi))
        res = angles - self.target
        return res

    def __call__(self, *args, **kwargs):
        """Return mean absolute deviation of the signal phase from target value in the moment of first break
        in the gather."""
        deltas = self.calc(*args, **kwargs)
        return np.mean(np.abs(deltas))

    def plot_gather(self, coords, ax, index, sort_by=None, mask=True, top_header=True, **kwargs):
        """Plot the gather with phase values at first break times above the seismogram
        and highlight traces whose metric value differs from the target more than 90 degrees.
        """
        _ = coords
        gather = self.survey.get_gather(index)
        if sort_by is not None:
            gather = gather.sort(by=sort_by)
        event_headers = kwargs.pop("event_headers", {"headers": self.first_breaks_col})
        signed_deltas = self.calc(gather=gather, refractor_velocity=None, first_breaks_col=self.first_breaks_col,
                                  correct_uphole=self.correct_uphole)
        kwargs["top_header"] = signed_deltas if top_header else None

        metric_values = np.abs(signed_deltas)
        mask_kwargs = kwargs.get("masks", {})
        mask_threshold = get_first_defined(mask_kwargs.get("threshold", None), self.vmax)
        mask_kwargs.update({"masks": metric_values, "threshold": mask_threshold})
        kwargs["masks"] = mask_kwargs if mask else None
        gather.plot(event_headers=event_headers, ax=ax, **kwargs)


class FirstBreaksCorrelations(RefractorVelocityMetric):
    """Mean Pearson correlation coeffitient of trace with mean hodograph in window around the first break.

    Parameters
    ----------
    window_size : int, optional, defaults to 40
        Size of the window to calculate the correlation coefficient in. Measured in milliseconds.
    """

    name = "first_breaks_correlations"
    views = ("plot_gather", "plot_mean_hodograph")
    vmin = 0
    vmax = 1
    is_lower_better = False

    def __init__(self, window_size=40, **kwargs):
        self.window_size = window_size
        super().__init__(**kwargs)

    @staticmethod
    @njit(nogil=True)
    def _make_windows(times, data, window_size, sample_interval, start_time):
        ix = ((times - start_time) // sample_interval).reshape(-1, 1).astype(np.int64)
        if np.any(data.shape[1] < ix) or np.any(ix < 0):
            ix = np.clip(ix, 0, data.shape[1] - 1)
        mean_cols = ix + np.arange(-window_size // (2 * sample_interval),
                                    window_size // (2 * sample_interval)).reshape(1, -1)
        mean_cols = np.clip(mean_cols, 0, data.shape[1] - 1).astype(np.int64)

        trace_len = mean_cols.shape[1]
        n_traces = len(data)
        traces_windows = np.empty((n_traces, trace_len), dtype=np.float32)
        for i in range(n_traces):
            traces_windows[i] = data[i][mean_cols[i]]
        return traces_windows

    @staticmethod
    @njit(nogil=True)
    def _calc(traces_windows):
        """Calculate signal correlation with mean hodograph"""
        n_traces, trace_len = traces_windows.shape
        mean_hodograph = np.empty((1, trace_len), dtype=traces_windows.dtype)
        for i in range(trace_len):
            mean_hodograph[:, i] = np.sum(traces_windows[:, i]) / (n_traces)
        mean_hodograph_centered = (mean_hodograph - np.mean(mean_hodograph)) / np.std(mean_hodograph)

        mean_amplitudes = np.empty((n_traces, 1), dtype=traces_windows.dtype)
        std_amplitudes = np.empty((n_traces, 1), dtype=traces_windows.dtype)
        for i in range(n_traces):
            mean_amplitudes[i] = np.sum(traces_windows[i]) / (trace_len)
            std_amplitudes[i] = np.sqrt((np.sum((traces_windows[i] - mean_amplitudes[i])**2) / trace_len))
        traces_windows_centered = (traces_windows - mean_amplitudes) / std_amplitudes

        corrs = np.empty(n_traces, dtype=traces_windows.dtype)
        for i in range(n_traces):
            corrs[i] = np.sum(traces_windows_centered[i] * mean_hodograph_centered) / trace_len
        return corrs

    def calc(self, gather, refractor_velocity, first_breaks_col, correct_uphole):
        """Return signal correlation with mean hodograph in the given window around first break times
        for a scaled gather.

        Parameters
        ----------
        gather : Gather
            A seismic gather to get offsets and times of first breaks from.
        first_breaks_col : str, optional, defaults to :const:`~const.HDR_FIRST_BREAK`
            Column name from `survey.headers` where times of first break are stored.

        Returns
        -------
        metric : np.ndarray of float
            Window correlation with mean hodograph for each trace in the gather.
        """
        _ = refractor_velocity, correct_uphole
        traces_windows = self._make_windows(gather[first_breaks_col], gather.data, self.window_size,
                         gather.sample_interval, gather.times[0])
        res = self._calc(traces_windows)
        return res

    def plot_mean_hodograph(self, coords, ax, index, **kwargs):
        """Plot mean trace in the scaled gather around the first break with length of the given window size."""
        _ = coords
        gather = self.survey.get_gather(index)
        g = gather.copy()
        g.scale_maxabs(clip=True)

        traces_windows = self._make_windows(g[self.first_breaks_col], g.data, self.window_size,
                                            g.sample_interval, g.times[0])
        mean_hodograph = traces_windows.mean(axis=0)
        mean_hodograph_centered = ((mean_hodograph - mean_hodograph.mean()) / mean_hodograph.std()).reshape(1, -1)
        g.data = mean_hodograph_centered

        g.plot(mode="wiggle", ax=ax, **kwargs)
        ax.set_xlabel("Amplitude")
        ax.set_xticks(ticks=[-1, 0, 1])
        ax.set_yticks(ticks=np.arange(self.window_size)[::5],labels=np.arange(self.window_size)[::5])


class DivergencePoint(RefractorVelocityMetric):
    """The divergence point metric for first breaks.
    Find an offset after that first breaks are most likely to diverge from expected time.
    Such an offset is defined as one with the maximum increase of outliers in between bins of `step` times.

    Parameters
    ----------
    threshold_times: float, optional, defaults to 50
        Threshold to define the first breaks outliers with, see `FirstBreaksOutliers`. Measured in milliseconds.
    step : int, optional, defaults to 100
        Size of the offset window to count outliers in. Measured in meters.
    """

    name = "divergence_point"
    is_lower_better = False

    def __init__(self, threshold_times=50, step=100, **kwargs):
        super().__init__(**kwargs)
        self.threshold_times = threshold_times
        self.step = step

    def bind_context(self, *args, **kwargs):
        """Set map attributes according to provided context."""
        super().bind_context(*args, **kwargs)
        self.vmax = self.survey["offset"].max()
        self.vmin = self.survey["offset"].min()

    @staticmethod
    @njit(nogil=True)
    def _calc(times, rv_times, offsets, threshold_times, step):
        """Calculate whether first break time is diverged from expected for each trace."""
        outliers = np.abs(rv_times - times) > threshold_times
        if np.mean(outliers) < 1e-3 or step >= len(offsets) - len(offsets) % step:
            return np.zeros_like(outliers)

        sorted_offsets_idx = np.argsort(offsets)
        sorted_offsets = offsets[sorted_offsets_idx]
        sorted_outliers = outliers[sorted_offsets_idx]

        split_idxs = np.arange(step, len(offsets) - len(offsets) % step, step)
        outliers_splits = np.split(sorted_outliers, split_idxs)
        n_splits = len(outliers_splits)

        outliers_fractions = np.array([outliers_window.mean() for outliers_window in outliers_splits])
        diffs = np.empty(n_splits - 1, dtype=outliers_fractions.dtype)
        for i in range(n_splits - 1):
            diffs[i] = outliers_fractions[i + 1] - outliers_fractions[i]
        div_idx = split_idxs[np.argmax(diffs)]
        div_offset = sorted_offsets[div_idx]
        return sorted_outliers * (sorted_offsets >= div_offset)

    def calc(self, gather, refractor_velocity, first_breaks_col, correct_uphole):
        """Return whether first break time is diverged from expected for each trace.
        First break is named diverged if it is an outlier after the divergence offset.

        Parameters
        ----------
        gather : Gather
            A seismic gather to get offsets and times of first breaks from.
        refractor_velocity : RefractorVelocity
            Near-surface velocity model to estimate the expected first break times at `gather` offsets.
        first_breaks_col : str, optional, defaults to :const:`~const.HDR_FIRST_BREAK`
            Column name from `survey.headers` where times of first break are stored.
        correct_uphole : bool, optional
            Whether to perform uphole correction by adding values of "SourceUpholeTime" header to times of first breaks
            emulating the case when sources are located on the surface. If not given, correction is performed if
            "SourceUpholeTime" header is loaded.

        Returns
        -------
        metric : np.ndarray of bool
            Array indicating whether first break of each trace in the gather is diverged.
        """
        times = gather[first_breaks_col]
        correct_uphole = (correct_uphole if correct_uphole is not None
                          else ("SourceUpholeTime" in gather.available_headers
                          and refractor_velocity.is_uphole_corrected))
        if correct_uphole:
            times = times + gather["SourceUpholeTime"]
        offsets = gather["offset"]
        rv_times = refractor_velocity(offsets)
        return self._calc(times, rv_times, offsets, self.threshold_times, self.step)

    def __call__(self, gather, *args, **kwargs):
        """Return the offset that defines a divergence point of first break times.

        Returns
        -------
        metric : int
            Metric value. Set to be the maximum offset when the overall fraction of outliers is close to zero.
        """
        diverged_outliers = self.calc(gather, *args, **kwargs)
        if np.allclose(diverged_outliers, 0):
            return gather['offset'].max()
        return np.sort(gather['offset'])[diverged_outliers.argmax()]

    def plot_gather(self, coords, ax, index, sort_by='offset', mask=True, top_header=False, **kwargs):
        """Plot the gather sorted by offset and hilight diverged traces."""
        _ = coords
        gather = self.survey.get_gather(index).sort(by=sort_by)
        event_headers = kwargs.pop("event_headers", {"headers": self.first_breaks_col})
        refractor_velocity = self.field(gather.coords)
        metric_values = self.calc(gather=gather, refractor_velocity=refractor_velocity,
                                  first_breaks_col=self.first_breaks_col, correct_uphole=self.correct_uphole)
        masks = metric_values if mask else None
        top_header = metric_values if top_header else None
        gather.plot(event_headers=event_headers, ax=ax, masks=masks, top_header=top_header, **kwargs)

    def plot_refractor_velocity(self, coords, ax, index, **kwargs):
        """Plot the refractor velocity curve, show the divergence offset
        and threshold area used for metric calculation."""
        gather = self.survey.get_gather(index)
        rv = self.field(coords)
        divergence_offset = self(gather=gather, refractor_velocity=rv, first_breaks_col=self.first_breaks_col,
                                 correct_uphole=self.correct_uphole)
        ax.axvline(x=divergence_offset, color="k", linestyle="--")
        threshold_times = kwargs.get("threshold_times", self.threshold_times)
        title = f"Divergence point: {divergence_offset} m"
        super().plot_refractor_velocity(coords, ax, index, title=title, threshold_times=threshold_times, **kwargs)

REFRACTOR_VELOCITY_QC_METRICS = [FirstBreaksOutliers, FirstBreaksAmplitudes, FirstBreaksPhases,
                                 FirstBreaksCorrelations, DivergencePoint]
