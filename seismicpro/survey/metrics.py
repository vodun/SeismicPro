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
    top_ax_y_scale = 'linear'

    params = []

    def __call__(self, gather, **kwargs):
        """Return an already calculated metric."""
        gather = self.preprocess(gather, **kwargs)
        mask = self.get_mask(gather, **kwargs)
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
    def get_mask(cls, gather, **kwargs):
        """QC indicator implementation.
        Takes a gather as an argument and returns either a samplewise qc indicator with shape
        (`gather.n_traces`, `gather.n_samples - d`), where `d >= 0`,
        or a tracewize indicator with shape (`gather.n_traces`,).
        For a 2d output with 2-d axis smaller than `gather.n_samples`,
        it will be padded with zeros at the beggining in `get_res`. Return QC indicator with zero traces masked
        with `np.nan` and output shape either `gater.data.shape`, or (`gather.n_traces`,)."""
        return NotImplementedError

    @property
    def kwargs(self):
        """returns metric kwargs"""
        return {'threshold': self.threshold, 'top_ax_y_scale': self.top_ax_y_scale,
                **{name: getattr(self, name) for name in self.params}}

    def plot(self, coords, ax, index, sort_by=None, bad_only=False, **kwargs):
        """Gather plot where samples with indicator above/below `cls.threshold` are highlited."""
        _ = coords
        gather = self.survey.get_gather(index)
        if sort_by is not None:
            gather = gather.sort(sort_by)
        gather = self.preprocess(gather, **self.kwargs)

        mask = self.get_mask(gather, **self.kwargs)
        metric_vals = self.aggregate(mask)

        bin_fn = np.greater_equal if self.is_lower_better else np.less_equal
        bin_mask = bin_fn(mask, self.threshold)
        if bad_only:
            gather.data[self.aggregate(bin_mask) == 0] = np.nan

        mode = kwargs.pop("mode", "wiggle")
        masks_dict = {"masks": bin_mask, "alpha":0.8, "label": self.name or "metric", **kwargs.pop("masks", {})}
        gather.plot(ax=ax, mode=mode, top_header=metric_vals, masks=masks_dict, **kwargs)
        ax.figure.axes[1].axhline(self.threshold, alpha=0.5)

    def get_views(self, sort_by=None, **kwargs):
        return [partial(self.plot, sort_by=sort_by), partial(self.plot, sort_by=sort_by, bad_only=True)], kwargs


class MuteTracewiseMetric(TracewiseMetric):
    params = ['muter']

    def __init__(self, muter, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.muter = muter

    def __repr__(self):
        """String representation of the metric."""
        return f"{type(self).__name__}(name='{self.name}', muter='{self.muter}')"

    @classmethod
    def preprocess(cls, gather, muter, **kwargs):
        _ = kwargs
        return gather.mute(muter=muter, fill_value=np.nan).scale_standard()


class Spikes(MuteTracewiseMetric):
    """Spikes detection."""
    name = "spikes"
    min_value = 0
    max_value = None
    is_lower_better = True
    threshold = 2
    top_ax_y_scale = 'log'

    @classmethod
    def get_mask(cls, gather, **kwargs):
        """QC indicator implementation."""
        _ = kwargs
        traces = gather.data.copy()
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
