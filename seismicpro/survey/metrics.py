"""Implements survey metrics"""

from functools import partial

import numpy as np
from numba import njit, prange
from matplotlib import patches

from ..metrics import Metric
from ..utils import to_list
from ..gather.utils import times_to_indices


class SurveyAttribute(Metric):
    """A utility metric class that reindexes given survey by `index_cols` and allows for plotting gathers by their
    indices. Does not implement any calculation logic."""
    def __init__(self, name=None):
        super().__init__(name=name)

        # Attributes set after context binding
        self.survey = None

    def bind_context(self, metric_map, survey):
        """Process metric evaluation context: memorize the parent survey, reindexed by `index_cols`."""
        self.survey = survey.reindex(metric_map.index_cols)

    def plot(self, ax, coords, index, sort_by=None, **kwargs):
        """Plot a gather by its `index`. Optionally sort it."""
        _ = coords
        gather = self.survey.get_gather(index)
        if sort_by is not None:
            gather = gather.sort(by=sort_by)
        gather.plot(ax=ax, **kwargs)

    def get_views(self, sort_by=None, **kwargs):
        """Return a single view, that plots a gather sorted by `sort_by` by click coordinates."""
        return [partial(self.plot, sort_by=sort_by)], kwargs


class TracewiseMetric(SurveyAttribute):
    """Base class for tracewise metrics with addidional plotters and aggregations. Child classes should redefine
    `get_mask` method, and optionnaly `preprocess`."""

    min_value = None
    max_value = None
    is_lower_better = None
    threshold = None
    top_ax_y_scale = "linear"

    def __init__(self, name=None):
        super().__init__(name=name)

    def __call__(self, gather):
        """Return an already calculated metric."""
        gather = self.preprocess(gather)
        mask = self.get_mask(gather)
        return self.aggregate(mask)

    @property
    def header_cols(self):
        return self.name

    def preprocess(self, gather):
        """Preprocess gather for calculating metric. Identity by default."""
        return gather

    def get_mask(self, gather):
        """QC indicator implementation. Takes a gather as an argument and returns either a samplewise qc indicator with
        shape equal to `gather.shape` or a tracewize indicator with shape (`gather.n_traces`,)."""
        raise NotImplementedError

    def aggregate_headers(self, headers, index_cols, coords_cols):
        index = headers[to_list(index_cols)] if index_cols is not None else None
        return headers[to_list(coords_cols)], headers[self.name], index

    def aggregate(self, mask):
        """Aggregate input mask depending on `self.is_lower_better` to select the worst mask value for each trace"""
        agg_fn = np.nanmax if self.is_lower_better else np.nanmin
        return mask if mask.ndim == 1 else agg_fn(mask, axis=1)

    def binarize(self, mask, threshold=None):
        """Binarize input mask by `threshold`. Depending on `self.is_lower_better` values greater or less than the
        `threshold` will be taken. If `threshold` is None, `self.threshold` is used."""
        bin_fn = np.greater_equal if self.is_lower_better else np.less_equal
        if threshold is None and self.threshold is None:
            raise ValueError("Either `threshold` or `self.threshold` must be non None.")
        return bin_fn(mask, self.threshold if threshold is None else threshold)

    def plot(self, coords, ax, index, sort_by=None, threshold=None, top_ax_y_scale=None,  bad_only=False, **kwargs):
        """Gather plot where samples with indicator above/below `.threshold` are highlited."""
        threshold = self.threshold if threshold is None else threshold
        top_ax_y_scale = self.top_ax_y_scale if top_ax_y_scale is None else top_ax_y_scale
        _ = coords

        gather = self.survey.get_gather(index)
        if sort_by is not None:
            gather = gather.sort(sort_by)
        gather = self.preprocess(gather)

        # TODO: Can we do only single copy here? (first copy done in self.preprocess)
        # We need to copy gather since some metrics changes gather in get_mask, but we want to plot gather unchanged
        mask = self.get_mask(gather.copy())
        metric_vals = self.aggregate(mask)
        bin_mask = self.binarize(mask, threshold)
        if bad_only:
            gather.data[self.aggregate(bin_mask) == 0] = np.nan

        mode = kwargs.pop("mode", "wiggle")
        masks_dict = {"masks": bin_mask, "alpha": 0.8, "label": self.name or "metric", **kwargs.pop("masks", {})}
        gather.plot(ax=ax, mode=mode, top_header=metric_vals, masks=masks_dict, **kwargs)
        ax.figure.axes[1].axhline(threshold, alpha=0.5)
        ax.figure.axes[1].set_yscale(top_ax_y_scale)

    def get_views(self, sort_by=None, threshold=None, top_ax_y_scale=None, **kwargs):
        """Return plotters of the metric views and those `kwargs` that should be passed further to an interactive map
        plotter."""
        plot_kwargs = {"sort_by": sort_by, "threshold": threshold, "top_ax_y_scale": top_ax_y_scale}
        return [partial(self.plot, **plot_kwargs), partial(self.plot, bad_only=True, **plot_kwargs)], kwargs


class MuteTracewiseMetric(TracewiseMetric):
    """Base class for tracewise metric with implemented `self.preprocess` method which applies muting and standatd
    scaling to the input gather. Child classes should redefine `get_mask` method."""

    def __init__(self, muter, name=None):
        super().__init__(name=name)
        self.muter = muter

    def __repr__(self):
        """String representation of the metric."""
        return f"{type(self).__name__}(name='{self.name}', muter='{self.muter}')"

    def preprocess(self, gather):
        return gather.copy().mute(muter=self.muter, fill_value=np.nan).scale_standard()


class Spikes(MuteTracewiseMetric):
    """Spikes detection. The metric reacts to drastic changes in traces ampliutes in 1-width window around each
    amplitude value. The resulted 2d mask shows the deviation of the ampluteds of an input gather.

    The metric is highly depends on muter, if muter isn't strong enough, the metric will overreact to the first breaks.

    Parameters
    ----------
    muter : Muter
    A muter to use.
    name : str, optional, defaults to "spikes"
    Metrics name.

    Attributes
    ----------
    ?? Do we want to describe them ??
    """

    name = "spikes"
    min_value = 0
    max_value = None
    is_lower_better = True
    threshold = 2

    def get_mask(self, gather):
        """QC indicator implementation."""
        traces = gather.data
        self.fill_leading_nulls(traces)

        res = np.abs(traces[:, 2:] + traces[:, :-2] - 2*traces[:, 1:-1]) / 3
        return np.pad(res, ((0, 0), (1, 1)))

    @staticmethod
    @njit(parallel=True, nogil=True)
    def fill_leading_nulls(arr):
        """"Fill leading null values of array's row with the first non null value in a row."""
        for i in prange(arr.shape[0]):
            nan_indices = np.nonzero(np.isnan(arr[i]))[0]
            if len(nan_indices) > 0:
                j = nan_indices[-1] + 1
                if j < arr.shape[1]:
                    arr[i, :j] = arr[i, j]


class Autocorrelation(MuteTracewiseMetric):
    """Autocorrelation with shift 1"""
    name = "autocorrelation"
    min_value = -1
    max_value = 1
    is_lower_better = False
    threshold = 0.8

    def get_mask(self, gather):
        """QC indicator implementation."""
        # TODO: descide what to do with almost nan traces (in 98% in trace are nan, it almost always will have -1 val)
        return np.nanmean(gather.data[..., 1:] * gather.data[..., :-1], axis=1)


class TraceAbsMean(TracewiseMetric):
    """Absolute value of the trace's mean scaled by trace's std."""
    name = "trace_absmean"
    is_lower_better = True
    threshold = 0.1

    def get_mask(self, gather):
        """QC indicator implementation."""
        return np.abs(gather.data.mean(axis=1) / (gather.data.std(axis=1) + 1e-10))


class TraceMaxAbs(TracewiseMetric):
    """Maximun absolute amplitude value scaled by trace's std."""
    name = "trace_maxabs"
    is_lower_better = True
    threshold = 15

    def get_mask(self, gather):
        """QC indicator implementation."""
        return np.max(np.abs(gather.data), axis=1) / (gather.data.std(axis=1) + 1e-10)


class MaxLenMetric(TracewiseMetric):
    """Base class for metrics that calculates length of continuous sequence of 1."""

    @staticmethod
    @njit(nogil=True, parallel=True)
    def compute_indicators_length(indicators, counter_init, old_shape):
        for i in prange(len(indicators)):
            counter = counter_init
            indicator = indicators[i]
            for j in range(len(indicator)):
                if indicator[j] == 1:
                    counter += 1
                else:
                    if counter > 1:
                        indicators[i, j - counter: j] = counter
                    counter = counter_init

            if counter > 1:
                indicators[i, -counter:] = counter
        return indicators.reshape(old_shape)


class MaxClipsLen(MaxLenMetric):
    """Detecting minimum/maximun clips"""
    name = "max_clips_len"
    min_value = 1
    max_value = None
    is_lower_better = True
    threshold = 3

    def get_mask(self, gather):
        """QC indicator implementation."""
        traces = gather.data

        maxes = traces.max(axis=-1, keepdims=True)
        mins = traces.min(axis=-1, keepdims=True)

        res_plus = self.get_val_subseq(traces, maxes)
        res_minus = self.get_val_subseq(traces, mins)

        return (res_plus + res_minus).astype(np.float32)

    def get_val_subseq(self, traces, val):
        old_shape = traces.shape
        traces = np.atleast_2d(traces)
        indicators = (traces == val).astype(np.int16)
        return self.compute_indicators_length(indicators, 0, old_shape)


class MaxConstLen(MaxLenMetric):
    """Detecting constant subsequences"""
    name = "const_len"
    is_lower_better = True
    threshold = 4

    def get_mask(self, gather):
        """QC indicator implementation."""
        traces = np.atleast_2d(gather.data)
        indicators = np.zeros_like(traces, dtype=np.int16)
        indicators[:, 1:] = (traces[:, 1:] == traces[:, :-1]).astype(np.int16)
        return self.compute_indicators_length(indicators, 1, gather.data.shape).astype(np.float32)


class DeadTrace(TracewiseMetric):
    """Detects constant traces."""
    name = "dead_trace"
    min_value = 0
    max_value = 1
    is_lower_better = True
    threshold = 0.5

    def get_mask(self, gather):
        """Return QC indicator."""
        return (np.max(gather.data, axis=1) - np.min(gather.data, axis=1) < 1e-10).astype(np.float32)


class WindowRMS(TracewiseMetric):
    """ RMS computed for provided window

    Parameters
    ----------
    offsets : tuple of 2 ints
        offset range to use for calcualtion.
    times : tuple of 2 ints
        time range to use for calcualtion, measured in ms.
    """
    name = "RMS"
    is_lower_better = False # TODO: think what should it be?
    # What treshold to use? Leave it none?
    threshold = None

    def __init__(self, offsets=None, times=None, name=None):
        super().__init__(name=name)
        self.offsets = offsets
        self.times = times

    def __call__(self, gather):
        """Return an already calculated metric."""
        gather = self.preprocess(gather)
        return self.get_mask(gather)

    @property
    def header_cols(self):
        return [self.name+"_sum", self.name+"_n"]

    @staticmethod
    def _get_times(gather, times):
        return (min(gather.samples), max(gather.samples)) if times is None else times

    @staticmethod
    def _get_offsets(gather, offsets):
        return (min(gather.offsets), max(gather.offsets)) if offsets is None else offsets

    @staticmethod
    def _get_indices(gather, times):
        times_range = np.asarray([max(gather.samples[0], times[0]), min(gather.samples[-1], times[1])])
        return times_to_indices(times_range, gather.samples).astype(np.int16)

    def get_mask(self, gather):
        """QC indicator implementation."""
        times = self._get_times(gather, self.times)
        offsets = self._get_offsets(gather, self.offsets)
        w_beg, w_end = self._get_indices(gather, times)

        window_ixs = np.nonzero((gather.offsets >= offsets[0]) & (gather.offsets <= offsets[1]))[0]
        square_amps, win_sizes = self.rms_win(gather.data, window_ixs, w_beg, w_end)
        return np.vstack((square_amps, win_sizes)).T

    @staticmethod
    @njit(nogil=True, parallel=True)
    def rms_win(data, ixs, w_beg, w_end):
        """Compute sum of square amplitudes for a window defined by its starting and end sample indices."""
        square_amps = np.full(data.shape[0], fill_value=np.nan)
        win_sizes = np.full(data.shape[0], fill_value=0)
        for i in prange(len(ixs)):
            ix = ixs[i]
            trace = data[ix, w_beg: w_end]
            square_amps[ix] = sum(trace**2)
            win_sizes[ix] = len(trace)

        return square_amps, win_sizes

    def aggregate_headers(self, headers, index_cols, coords_cols):
        groupby_cols = self.header_cols + coords_cols if index_cols != coords_cols else []
        groupby = headers.groupby(index_cols)[groupby_cols]
        aggregated_gb = groupby.agg({groupby_cols[0]: lambda x: np.sqrt(np.sum(x)),
                                     groupby_cols[1]: "sum",
                                     **{groupby_cols[i]: "mean" for i in range(2, len(groupby_cols))}})
        aggregated_gb.reset_index(inplace=True)
        coords = aggregated_gb[coords_cols]
        value = aggregated_gb[groupby_cols[0]] / aggregated_gb[groupby_cols[1]]
        index = aggregated_gb[index_cols]
        return coords, value, index

    def plot(self, coords, ax, index, threshold=None, top_ax_y_scale=None, bad_only=False, **kwargs):
        """Gather plot sorted by offset with tracewise indicator on a separate axis and signal and noise windows"""
        _ = coords
        # TODO: add threshold processing and marking traces inside rectangle
        gather = self.survey.get_gather(index).sort(by='offset')
        amp_sum, amp_num = self.get_mask(gather).T
        tracewise_metric = np.sqrt(amp_sum) / amp_num
        tracewise_metric[tracewise_metric==0] = np.nan
        if bad_only:
            bin_mask = self.binarize(tracewise_metric, threshold)
            gather.data[self.aggregate(bin_mask) == 0] = np.nan

        mode = kwargs.pop("mode", "wiggle")
        gather.plot(ax=ax, mode=mode, top_header=tracewise_metric, **kwargs)
        ax.figure.axes[1].axhline(threshold, alpha=0.5)
        top_ax_y_scale = self.top_ax_y_scale if top_ax_y_scale is None else top_ax_y_scale
        ax.figure.axes[1].set_yscale(top_ax_y_scale)

        times = self._get_times(gather, self.times)
        offsets = self._get_offsets(gather, self.offsets)
        w_beg, w_end = self._get_indices(gather, times)

        offs_ind = np.nonzero((gather.offsets >= offsets[0]) & (gather.offsets <= offsets[1]))[0]
        if len(offs_ind) > 0:
            n_rec = (offs_ind[0], w_beg), len(offs_ind), (w_end - w_beg)
            ax.add_patch(patches.Rectangle(*n_rec, linewidth=2, edgecolor='magenta', facecolor='none'))

    def get_views(self, threshold=None, top_ax_y_scale=None, **kwargs):
        """Return plotters of the metric views and those `kwargs` that should be passed further to an interactive map
        plotter."""
        plot_kwargs = {"threshold": threshold, "top_ax_y_scale": top_ax_y_scale}
        return [partial(self.plot, **plot_kwargs), partial(self.plot, bad_only=True, **plot_kwargs)], kwargs


DEFAULT_TRACEWISE_METRICS = [TraceAbsMean, TraceMaxAbs, MaxClipsLen, MaxConstLen, DeadTrace]
