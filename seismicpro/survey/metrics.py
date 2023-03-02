"""Implements survey metrics"""

from functools import partial

import numpy as np
from numba import njit, prange

from ..metrics import Metric


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
    """Base class for tracewise metrics with addidional plotters and aggregations
    Child classes should redefine `_get_res` method, and optionnaly `preprocess`.
    """

    min_value = None
    max_value = None
    is_lower_better = None
    threshold = None
    top_ax_y_scale = "linear"

    preprocess_kwargs = []

    def __init__(self, name=None):
        super().__init__(name=name)

    def __call__(self, gather, **kwargs):
        """Return an already calculated metric."""
        gather = self.preprocess(gather, **{**self.get_preprocess_kwargs, **kwargs})
        mask = self.get_mask(gather)
        return self.aggregate(mask)

    def aggregate(self, mask):
        """Aggregate input mask depending on `cls.is_lower_better."""
        agg_fn = np.nanmax if self.is_lower_better else np.nanmin
        return mask if mask.ndim == 1 else agg_fn(mask, axis=1)

    @classmethod
    def preprocess(cls, gather, **kwargs):
        """Preprocess gather for calculating metric. Identity by default."""
        _ = kwargs
        return gather

    @classmethod
    def get_mask(cls, gather):
        """QC indicator implementation.
        Takes a gather as an argument and returns either a samplewise qc indicator with shape
        (`gather.n_traces`, `gather.n_samples - d`), where `d >= 0`,
        or a tracewize indicator with shape (`gather.n_traces`,).
        For a 2d output with 2-d axis smaller than `gather.n_samples`,
        it will be padded with zeros at the beggining in `get_res`. Return QC indicator with zero traces masked
        with `np.nan` and output shape either `gater.data.shape`, or (`gather.n_traces`,)."""
        raise NotImplementedError

    @property
    def get_preprocess_kwargs(self):
        """Returns all args to self.preprocess method."""
        return {name: getattr(self, name) for name in self.preprocess_kwargs}

    def plot(self, coords, ax, index, sort_by=None, threshold=None, top_ax_y_scale=None,  bad_only=False, **kwargs):
        """Gather plot where samples with indicator above/below `cls.threshold` are highlited."""
        threshold = self.threshold if threshold is None else threshold
        top_ax_y_scale = self.top_ax_y_scale if top_ax_y_scale is None else top_ax_y_scale
        _ = coords

        gather = self.survey.get_gather(index)
        if sort_by is not None:
            gather = gather.sort(sort_by)
        gather = self.preprocess(gather, **self.get_preprocess_kwargs)

        # TODO: Can we do only single copy here? (first copy done in self.preprocess)
        # We need to copy gather since some metrics changes gather in get_mask, but we want to plot gather unchanged
        mask = self.get_mask(gather.copy())
        metric_vals = self.aggregate(mask)

        bin_fn = np.greater_equal if self.is_lower_better else np.less_equal
        bin_mask = bin_fn(mask, threshold)
        if bad_only:
            gather.data[self.aggregate(bin_mask) == 0] = np.nan

        mode = kwargs.pop("mode", "wiggle")
        masks_dict = {"masks": bin_mask, "alpha":0.8, "label": self.name or "metric", **kwargs.pop("masks", {})}
        gather.plot(ax=ax, mode=mode, top_header=metric_vals, masks=masks_dict, **kwargs)
        ax.figure.axes[1].axhline(threshold, alpha=0.5)
        ax.figure.axes[1].set_yscale(top_ax_y_scale)

    def get_views(self, sort_by=None, threshold=None, top_ax_y_scale=None, **kwargs):
        plot_kwargs = {"sort_by": sort_by, "threshold": threshold, "top_ax_y_scale": top_ax_y_scale}
        return [partial(self.plot, **plot_kwargs), partial(self.plot, bad_only=True, **plot_kwargs)], kwargs


class MuteTracewiseMetric(TracewiseMetric):
    preprocess_kwargs = ['muter']

    def __init__(self, muter, name=None):
        super().__init__(name=name)
        self.muter = muter

    def __repr__(self):
        """String representation of the metric."""
        return f"{type(self).__name__}(name='{self.name}', muter='{self.muter}')"

    @classmethod
    def preprocess(cls, gather, muter):
        return gather.copy().mute(muter=muter, fill_value=np.nan).scale_standard()


class Spikes(MuteTracewiseMetric):
    """Spikes detection."""
    name = "spikes"
    min_value = 0
    max_value = None
    is_lower_better = True
    threshold = 2
    top_ax_y_scale = "log"

    @classmethod
    def get_mask(cls, gather):
        """QC indicator implementation."""
        traces = gather.data
        cls.fill_leading_nulls(traces)

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


class Autocorr(MuteTracewiseMetric):
    """Autocorrelation with shift 1"""
    name = "autocorr"
    min_value = -1
    max_value = 1
    is_lower_better = False
    threshold = 0.8
    top_ax_y_scale = "log"

    @classmethod
    def get_mask(cls, gather):
        """QC indicator implementation."""
        return np.nanmean(gather.data[..., 1:] * gather.data[..., :-1], axis=1)


class TraceAbsMean(TracewiseMetric):
    """Absolute value of the trace's mean scaled by trace's std."""
    name = "trace_absmean"
    is_lower_better = True
    threshold = 0.1
    top_ax_y_scale = "log"

    @classmethod
    def get_mask(cls, gather):
        """QC indicator implementation."""
        return np.abs(gather.data.mean(axis=1) / (gather.data.std(axis=1) + 1e-10))


class TraceMaxAbs(TracewiseMetric):
    """Maximun absolute amplitude value scaled by trace's std."""
    name = "trace_maxabs"
    is_lower_better = True
    threshold = 15
    top_ax_y_scale = "log"

    @classmethod
    def get_mask(cls, gather):
        """QC indicator implementation."""
        return np.max(np.abs(gather.data), axis=1) / (gather.data.std(axis=1) + 1e-10)


class MaxClipsLen(TracewiseMetric):
    """Detecting minimum/maximun clips"""
    name = "max_clips_len"
    min_value = 1
    max_value = None
    is_lower_better = True
    threshold = 3

    @classmethod
    def get_mask(cls, gather, **kwargs):
        """QC indicator implementation."""
        _ = kwargs
        traces = gather.data

        maxes = traces.max(axis=-1, keepdims=True)
        mins = traces.min(axis=-1, keepdims=True)

        res_plus = cls.get_val_subseq(traces, maxes)
        res_minus = cls.get_val_subseq(traces, mins)

        return (res_plus + res_minus).astype(np.float32)

    @staticmethod
    @njit(nogil=True)
    def get_val_subseq(traces, cmpval):
        """Indicator of constant subsequences equal to given value."""

        old_shape = traces.shape
        traces = np.atleast_2d(traces)

        indicators = (traces == cmpval).astype(np.int16)

        for t, indicator in enumerate(indicators):
            counter = 0
            for i, sample in enumerate(indicator):
                if sample == 1:
                    counter += 1
                else:
                    if counter > 1:
                        indicators[t, i - counter: i] = counter
                    counter = 0

            if counter > 1:
                indicators[t, -counter:] = counter

        return indicators.reshape(*old_shape)


class MaxConstLen(TracewiseMetric):
    """Detecting constant subsequences"""
    name = "const_len"
    is_lower_better = True
    threshold = 4

    @classmethod
    def get_mask(cls, gather, **kwargs):
        """QC indicator implementation."""
        _ = kwargs
        res = cls.get_const_subseq(gather.data)
        return res.astype(np.float32)

    @staticmethod
    @njit(nogil=True)
    def get_const_subseq(traces):
        """Indicator of constant subsequences."""

        old_shape = traces.shape
        traces = np.atleast_2d(traces)

        indicators = np.zeros_like(traces, dtype=np.int16)
        indicators[:, 1:] = (traces[:, 1:] == traces[:, :-1]).astype(np.int16)
        for t, indicator in enumerate(indicators):
            counter = 0
            for i, sample in enumerate(indicator):
                if sample == 1:
                    counter += 1
                else:
                    if counter > 0:
                        indicators[t, i - counter - 1:i] = counter + 1
                    counter = 0

            if counter > 0:
                indicators[t, -counter - 1:] = counter + 1

        return indicators.reshape(*old_shape)

class DeadTrace(TracewiseMetric):
    """Detects constant traces."""
    name = "DeadTrace"
    min_value = 0
    max_value = 1
    is_lower_better = True
    threshold = 0.5

    @classmethod
    def get_mask(cls, gather, **kwargs):
        """Return QC indicator."""
        _ = kwargs
        return (np.max(gather.data, axis=1) - np.min(gather.data, axis=1) < 1e-10).astype(np.float32)
