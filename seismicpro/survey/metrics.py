# pylint: disable=not-an-iterable
"""Implements survey metrics"""

import warnings
from functools import partial

import numpy as np
from numba import njit
from matplotlib import patches

from ..metrics import Metric
from ..utils import times_to_indices, isclose

# Ignore all warnings related to empty slices or dividing by zero
warnings.simplefilter("ignore", category=RuntimeWarning)


class SurveyAttribute(Metric):
    """A utility metric class that reindexes given survey by `index_cols` and allows for plotting gathers by their
    indices. Does not implement any calculation logic."""
    def __init__(self, name=None):
        super().__init__(name=name)

        # Attributes set after context binding
        self.survey = None

    @property
    def header_cols(self):
        """Column names in survey.headers to store the metrics results in."""
        return self.name

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
    """Base class for tracewise metrics with additional plotters and aggregations. Child classes should redefine
    `get_mask` or `numba_get_mask` methods, and optionally `preprocess` and `description`."""
    threshold = None
    top_ax_y_scale = "linear"

    def __call__(self, gather):
        """Compute qc metric by applying `self.preprocess`, `self.get_mask` and `self.aggregate` to provided gather."""
        gather = self.preprocess(gather)
        mask = self.get_mask(gather)
        return self.aggregate(mask)

    @property
    def description(self):
        """String description of the tracewise metric. Mainly used in `survey.info` when describing the number of bad
        traces detected by the metric."""
        return NotImplementedError

    def preprocess(self, gather):
        """Preprocess gather before either calling `self.get_mask` method to calculate metric or to plot the gather.
        Identity by default."""
        _ = self
        return gather

    def get_mask(self, gather):
        """Compute QC indicator.

        For a provided gather returns either a samplewise qc indicator with the same shape as `gather` or a tracewize
        indicator with a shape of (`gather.n_traces`,).

        The method redirects the call to njitted static `numba_get_mask` method. Either this method or `numba_get_mask`
        must be overridden in child classes.
        """
        return self.numba_get_mask(gather.data)

    @staticmethod
    @njit(nogil=True)
    def numba_get_mask(traces):
        """Compute QC indicator in parallel."""
        raise NotImplementedError

    def aggregate(self, mask):
        """Columnwise `mask` aggregation depending on `self.is_lower_better` to select the worst values."""
        if self.is_lower_better is None:
            agg_fn = np.nanmean
        elif self.is_lower_better:
            agg_fn = np.nanmax
        else:
            agg_fn = np.nanmin
        return mask if mask.ndim == 1 else agg_fn(mask, axis=1)

    def binarize(self, mask, threshold=None):
        """Binarize a given mask by a `threshold`.

        Parameters
        ----------
        mask : 1d ndarray or 2d ndarray
            Array with computed metric values to be converted to a binary mask.
        threshold : int, float, array-like with 2 elements, optional, defaults to None
            Threshold used to binarize the mask.
            If int or float, depending on `self.is_lower_better` values greater or less than the `threshold` will be
            treated as a bad value and marked as True. If array, two numbers indicate the boundaries within which the
            metric values are treated as False, outside inclusive - as True. If None, self.threshold will be used.

        Returns
        -------
        bin_mask : 1d ndarray or 2d ndarray
            Binary mask obtained by comparing the mask with threshold.

        Raises
        ------
        ValueError
            If threshold is not provided and self.threshold is None.
            If threshold is a single number and self.is_lower_better is None.
            If threshold is iterable but does not contain exactly 2 elements.
        """
        threshold = self.threshold if threshold is None else threshold
        if threshold is None:
            raise ValueError("Either `threshold` or `self.threshold` must be non None")

        nan_ixs = np.isnan(mask)
        if isinstance(threshold, (int, float, np.number)):
            if self.is_lower_better is None:
                raise ValueError("`threshold` cannot be single number if `is_lower_better` is None")
            bin_fn = np.greater_equal if self.is_lower_better else np.less_equal
            bin_mask = bin_fn(mask, threshold)
        elif len(threshold) != 2:
            raise ValueError(f"`threshold` should contain exactly 2 elements, not {len(threshold)}")
        else:
            bin_mask = (mask <= threshold[0]) | (mask >= threshold[1])
        bin_mask[nan_ixs] = False
        return bin_mask

    def plot(self, ax, coords, index, sort_by=None, threshold=None, top_ax_y_scale=None,  bad_only=False, **kwargs):
        """Plot gather by its `index` with highlighted traces with metric value above or below the `self.threshold`.

        Tracewise metric values will be shown on top of the gather plot. Also, the area with `good` metric values based
        on threshold values and `self.is_lower_better` will be highlighted in blue. If `self.is_lower_better` is None
        and threshold is a number, only the threshold line will be displayed.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes of the figure to plot on.
        coords : array-like with 2 elements
            Gather coordinates.
        index : array-like with 2 elements
            Gather index.
        sort_by : str or array-like, optional, defaults to None
            Headers names to sort the gather by.
        threshold : int, float, array-like with 2 elements, optional, defaults to None
            Threshold used to binarize the metric values.
            If None, `self.threshold` will be used. See `self.binarize` for more details.
        top_ax_scale : str, optional, defaults to None
            Scale type for top header plot, see `matplotlib.axes.Axes.set_yscale` for avalible options.
        bad_only : bool, optional, defaults to False
            Show only traces that are considered as bad based on provided threshold and `self.is_lower_better`.
        kwargs : misc, optional
            Additional keyword arguments to the `gather.plot`.
        """
        threshold = self.threshold if threshold is None else threshold
        top_ax_y_scale = self.top_ax_y_scale if top_ax_y_scale is None else top_ax_y_scale
        _ = coords

        gather = self.survey.get_gather(index)
        if sort_by is not None:
            gather = gather.sort(sort_by)
        gather = self.preprocess(gather)

        mask = self.get_mask(gather)
        metric_vals = self.aggregate(mask)
        bin_mask = self.binarize(mask, threshold)

        mode = kwargs.pop("mode", "wiggle")
        masks_dict = {"masks": bin_mask, "alpha": 0.8, "label": self.name or "metric", **kwargs.pop("masks", {})}

        if bad_only:
            gather.data[self.aggregate(bin_mask) == 0] = np.nan
            masks_dict = {}  # Don't need to plot the mask since only bad traces will be plotted.

        gather.plot(ax=ax, mode=mode, top_header=metric_vals, masks=masks_dict, **kwargs)
        top_ax = ax.figure.axes[1]
        top_ax.set_yscale(top_ax_y_scale)
        if threshold is not None:
            self._plot_threshold(ax=top_ax, threshold=threshold)

    def _plot_threshold(self, ax, threshold):
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        if isinstance(threshold, (int, float, np.number)):
            if self.is_lower_better is None:
                ax.axhline(threshold, alpha=0.5, color="blue")
                return
            threshold = [threshold, y_max] if self.is_lower_better else [y_min, threshold]
        ax.fill_between(np.arange(x_min, x_max), *threshold, alpha=0.3, color="blue")

    def get_views(self, sort_by=None, threshold=None, top_ax_y_scale=None, **kwargs):
        """Return two plotters of the metric views. Each view plots a gather sorted by `sort_by` with a metric values
        shown on top of the gather plot. The y-axis of the metric plot is scaled by `top_ax_y_scale`. The first view
        plots full gather with bad traces highlighted based on the `threshold` and the `self.is_lower_better`
        attribute. The second view only displays the traces defined by the metric as bad ones."""
        plot_kwargs = {"sort_by": sort_by, "threshold": threshold, "top_ax_y_scale": top_ax_y_scale}
        return [partial(self.plot, **plot_kwargs), partial(self.plot, bad_only=True, **plot_kwargs)], kwargs


class MuteTracewiseMetric(TracewiseMetric):  # pylint: disable=abstract-method
    """Base class for tracewise metric with implemented `self.preprocess` method which applies muting and standard
    scaling to the input gather. Child classes should redefine `get_mask` or `numba_get_mask` methods.

    Parameters
    ----------
    muter : Muter
        A muter to use.
    name : str, optional, defaults to None
        A metirc name.
    """
    def __init__(self, muter, name=None):
        super().__init__(name=name)
        self.muter = muter

    def __repr__(self):
        """String representation of the metric."""
        return f"{type(self).__name__}(name='{self.name}', muter='{self.muter}')"

    def preprocess(self, gather):
        """Apply muting with np.nan as a fill value and standard scaling to provided gather."""
        return gather.copy().mute(muter=self.muter, fill_value=np.nan).scale_standard()


class Spikes(MuteTracewiseMetric):
    """Spikes detection. The metric reacts to drastic changes in traces ampliutes within a 1-width window around each
    amplitude value.

    `get_mask` returns 2d mask that shows the deviation of the ampluteds of an input gather.

    The metric is highly dependent on the muter being used; if muter is not strong enough, the metric will overreact
    to the first breaks.
    """
    name = "spikes"
    min_value = 0
    max_value = None
    is_lower_better = True
    threshold = 2

    @property
    def description(self):
        """String description of tracewise metric."""
        return "Traces with spikes"

    @staticmethod
    @njit(nogil=True)
    def numba_get_mask(traces):
        """Compute QC indicator in parallel."""
        traces = traces.copy()
        res = np.zeros_like(traces)
        for i in range(traces.shape[0]):
            nan_indices = np.nonzero(np.isnan(traces[i]))[0]
            if len(nan_indices) > 0:
                j = nan_indices[-1] + 1
                if j < traces.shape[1]:
                    traces[i, :j] = traces[i, j]
                else:
                    traces[i, :] = 0
            res[i, 1: -1] = np.abs(traces[i, 2:] + traces[i, :-2] - 2*traces[i, 1:-1]) / 3
        return res


class Autocorrelation(MuteTracewiseMetric):
    """Trace correlation with itself shifted by 1.

    `get_mask` returns 1d mask with mean trace autocorrelation. If proportion of nans in the trace is greater than
    `nan_ratio`, then the metric value for the trace will be nan.

    The metric is highly dependent on the muter being used; if muter is not strong enough, the metric will overreact
    to the first breaks.

    Parameters
    ----------
    muter : Muter
        A muter to use.
    name : str, optional, defaults to "autocorrelation"
        A metric name.
    nan_ratio : float, optional, defaults to 0.95
        The maximum proportion of nan values allowed in a trace.
    """
    name = "autocorrelation"
    min_value = -1
    max_value = 1
    is_lower_better = False
    threshold = 0.8

    def __init__(self, muter, name=None, nan_ratio=0.95):
        super().__init__(muter=muter, name=name)
        self.nan_ratio = nan_ratio

    @property
    def description(self):
        """String description of tracewise metric."""
        return f"Traces with autocorrelation less than {self.threshold}"

    def get_mask(self, gather):
        return self.numba_get_mask(gather.data, nan_ratio=self.nan_ratio)

    @staticmethod
    @njit(nogil=True)
    def numba_get_mask(traces, nan_ratio):
        """Compute QC indicator in parallel."""
        res = np.empty_like(traces[:, 0])
        for i in range(traces.shape[0]):
            if np.isnan(traces[i]).sum() > nan_ratio*len(traces[i]):
                res[i] = np.nan
            else:
                res[i] = np.nanmean(traces[i, 1:] * traces[i, :-1])
        return res

class TraceAbsMean(TracewiseMetric):
    """Calculate absolute value of the trace's mean scaled by trace's std.

    `get_mask` returns 1d array wtih computed metric values for the gather.
    """
    name = "trace_absmean"
    is_lower_better = True
    threshold = 0.1

    @property
    def description(self):
        """String description of tracewise metric."""
        return f"Traces with mean divided by std greater than {self.threshold}"

    @staticmethod
    @njit(nogil=True)
    def numba_get_mask(traces):
        """Compute QC indicator in parallel."""
        res = np.empty_like(traces[:, 0])
        for i in range(traces.shape[0]):
            res[i] = np.abs(traces[i].mean() / (traces[i].std() + 1e-10))
        return res


class TraceMaxAbs(TracewiseMetric):
    """Find a maximum absolute amplitude value scaled by trace's std.

    `get_mask` returns 1d array wtih computed metric values for the gather.
    """
    name = "trace_maxabs"
    is_lower_better = True
    threshold = 15

    @property
    def description(self):
        """String description of tracewise metric"""
        return f"Traces with max abs to std ratio greater than {self.threshold}"

    @staticmethod
    @njit(nogil=True)
    def numba_get_mask(traces):
        """Compute QC indicator in parallel."""
        res = np.empty_like(traces[:, 0])
        for i in range(traces.shape[0]):
            res[i] = np.max(np.abs(traces[i])) / (traces[i].std() + 1e-10)
        return res


class MaxClipsLen(TracewiseMetric):
    """Calculate the length of consecutive minimum or maximun ampliuteds clips.

    `get_mask` returns 2d mask indicating the length of consecutive maximum or minimum amplitudes for each trace in the
    input gather.
    """
    name = "max_clips_len"
    min_value = 1
    max_value = None
    is_lower_better = True
    threshold = 3

    @property
    def description(self):
        """String description of tracewise metric."""
        return f"Traces with more than {self.threshold} clipped samples in a row"

    @staticmethod
    @njit(nogil=True)
    def numba_get_mask(traces):
        """Compute QC indicator in parallel."""
        def _update_counters(trace, i, j, value, counter, container):
            if isclose(trace, value):
                counter += 1
            else:
                if counter > 1:
                    container[i, j - counter: j] = counter
                counter = 0
            return counter

        maxes = np.zeros_like(traces)
        mins = np.zeros_like(traces)
        for i in range(traces.shape[0]):
            trace = traces[i]
            max_val = max(trace)
            max_counter = 0
            min_val = min(trace)
            min_counter = 0
            for j in range(trace.shape[0]):  # pylint: disable=consider-using-enumerate
                max_counter = _update_counters(trace[j], i, j, max_val, max_counter, maxes)
                min_counter = _update_counters(trace[j], i, j, min_val, min_counter, mins)

            if max_counter > 1:
                maxes[i, -max_counter:] = max_counter
            if min_counter > 1:
                mins[i, -min_counter:] = min_counter
        return maxes + mins


class MaxConstLen(TracewiseMetric):
    """Calcualte the number of consecutive identical amplitudes.

    `get_mask` returns 2d mask indicating the length of consecutive identical values in each trace in the input gather.
    """
    name = "const_len"
    is_lower_better = True
    threshold = 4

    @property
    def description(self):
        """String description of tracewise metric"""
        return f"Traces with more than {self.threshold} identical values in a row"

    @staticmethod
    @njit(nogil=True)
    def numba_get_mask(traces):
        """Compute QC indicator in parallel."""
        indicator = np.zeros_like(traces)
        for i in range(traces.shape[0]):
            trace = traces[i]
            counter = 1
            for j in range(1, trace.shape[0]):  # pylint: disable=consider-using-enumerate
                if isclose(trace[j], trace[j-1]):
                    counter += 1
                else:
                    if counter > 1:
                        indicator[i, j - counter: j] = counter
                    counter = 1

            if counter > 1:
                indicator[i, -counter:] = counter
        return indicator


class DeadTrace(TracewiseMetric):
    """Detect constant traces.

    `get_mask` returns 1d binary mask where each dead trace is marked with one.
    """
    name = "dead_trace"
    min_value = 0
    max_value = 1
    is_lower_better = True
    threshold = 0.5

    @property
    def description(self):
        return "Number of dead traces"

    @staticmethod
    @njit(nogil=True)
    def numba_get_mask(traces):
        """Compute QC indicator in parallel."""
        res = np.empty_like(traces[:, 0])
        for i in range(traces.shape[0]):
            res[i] = isclose(max(traces[i]), min(traces[i]))
        return res


class BaseWindowRMSMetric(TracewiseMetric):
    """Base class for the tracewise metrics that computes RMS in windows defined by two arrays with start and end
    indices for each trace in provided gather. Child classes should redefine `get_mask` or `numba_get_mask` methods."""

    def __call__(self, gather, return_rms=True):
        """Compute the metric by applying `self.preprocess` and `self.get_mask` to provided gather.
        If `return_rms` is True, the RMS value for provided gather will be returned.
        Otherwise, two 1d arrays will be returned:
            1. Sum of squares of amplitudes in the defined window for each trace,
            2. Number of amplitues in a specified window for each trace.
        """
        gather = self.preprocess(gather)
        squares, nums = self.get_mask(gather)
        if return_rms:
            return self.compute_rms(squares, nums)
        return squares, nums

    @property
    def header_cols(self):
        """Column names in survey.headers to store the metrics results in."""
        return [self.name+"_sum", self.name+"_n"]

    @staticmethod
    def compute_rms(squares, nums):
        """Compute RMS using provided squares of amplitues and the number of amplitudes used for square calculation."""
        return np.sqrt(np.sum(squares) / np.sum(nums))

    @staticmethod
    @njit(nogil=True)
    def compute_stats_by_ixs(data, start_ixs, end_ixs):
        """Compute the sum of squares and the number of elements in a window specified by `start_ixs` and `end_ixs`
        for each trace in provided data."""
        sum_squares = np.empty_like(data[:, 0])
        nums = np.empty_like(data[:, 0])

        for i in range(data.shape[0]):
            trace = data[i]
            start_ix = start_ixs[i]
            end_ix = end_ixs[i]
            sum_squares[i] = sum(trace[start_ix: end_ix] ** 2)
            nums[i] = len(trace[start_ix: end_ix])
        return sum_squares, nums

    def construct_map(self, coords, values, *, coords_cols=None, index=None, index_cols=None, agg=None, bin_size=None,
                      calculate_immediately=True):
        """Construct a metric map with computed RMS values for gathers indexed by `index`."""
        sum_square_map = super().construct_map(coords, values.iloc[:, 0], coords_cols=coords_cols, index=index,
                                               index_cols=index_cols, agg="sum")
        nums_map = super().construct_map(coords, values.iloc[:, 1], coords_cols=coords_cols, index=index,
                                         index_cols=index_cols, agg="sum")
        sum_square_map.index_data.drop(columns=sum_square_map.coords_cols, inplace=True)
        sum_df = sum_square_map.index_data.merge(nums_map.index_data, on=nums_map.index_cols)
        sum_df[self.name] = np.sqrt(sum_df[self.name+"_x"] / sum_df[self.name+"_y"])
        return super().construct_map(sum_df[coords.columns], sum_df[self.name], index=sum_df[index.columns], agg=agg,
                                     bin_size=bin_size, calculate_immediately=calculate_immediately)

    def plot(self, ax, coords, index, threshold=None, top_ax_y_scale=None, bad_only=False, color="lime",
             **kwargs):
        """Plot the gather sorted by offset with tracewise indicator on the top of the gather plot. Any mask can be
        displayied over the gather plot using `self.add_mask_on_plot`."""
        threshold = self.threshold if threshold is None else threshold
        top_ax_y_scale = self.top_ax_y_scale if top_ax_y_scale is None else top_ax_y_scale
        _ = coords
        gather = self.survey.get_gather(index).sort("offset")
        squares, nums = self(gather, return_rms=False)
        tracewise_metric = np.sqrt(squares / nums)
        tracewise_metric[tracewise_metric == 0] = np.nan
        if bad_only:
            bin_mask = self.binarize(tracewise_metric, threshold)
            gather.data[self.aggregate(bin_mask) == 0] = np.nan

        gather.plot(ax=ax, top_header=tracewise_metric, **kwargs)
        top_ax = ax.figure.axes[1]
        if threshold is not None:
            self._plot_threshold(ax=top_ax, threshold=threshold)
        top_ax.set_yscale(top_ax_y_scale)
        self.add_mask_on_plot(ax=ax, gather=gather, color=color)

    def add_mask_on_plot(self, ax, gather, color=None):
        """Plot any additional metric related graphs over the gather plot."""
        _ = ax, gather, color
        pass

    def get_views(self, threshold=None, top_ax_y_scale=None, **kwargs):
        """Return two plotters of the metric views. Each view plots a gather with a metric values shown on top of the
        gather plot. The y-axis of the metric plot is scaled by `top_ax_y_scale`. The first view plots full gather with
        bad traces highlighted based on the `threshold` and the `self.is_lower_better` attribute. The second view only
        displays the traces defined by the metric as bad ones."""
        plot_kwargs = {"threshold": threshold, "top_ax_y_scale": top_ax_y_scale}
        return [partial(self.plot, **plot_kwargs), partial(self.plot, bad_only=True, **plot_kwargs)], kwargs


class MetricsRatio(TracewiseMetric):
    """Calculate the ratio of two window RMS metircs.

    In the metric map, the displayed values are obtained by dividing the  value of `self.numerator` metric by the
    value of `self.denominator` metric for each gather independently.

    Parameters
    ----------
    numerator : subclass of BaseWindowRMSMetric
        Metric instance whose values will be divided by the `denominator` metirc values.
    denominator : subclass of BaseWindowRMSMetric
        Metric instance whose values will be used as a divisor for a `numerator` metric.
    """
    is_lower_better = False
    threshold = None

    def __init__(self, numerator, denominator, name=None):
        for metric in [numerator, denominator]:
            if not isinstance(metric, BaseWindowRMSMetric):
                msg = f"Metric ratio can be computed only for BaseWindowRMSMetric instances or its subclasses, but \
                       given metric has type: {type(metric)}."
                raise ValueError(msg)

        name = f"{numerator.name} to {denominator.name} ratio" if name is None else name
        super().__init__(name=name)

        self.numerator = numerator
        self.denominator = denominator

    @property
    def header_cols(self):
        """Column names in survey.headers to store the metrics results in."""
        return self.numerator.header_cols + self.denominator.header_cols

    def construct_map(self, coords, values, *, coords_cols=None, index=None, index_cols=None, agg=None, bin_size=None,
                      calculate_immediately=True):
        """Construct a metric map with `self.numerator` and `self.denominator` ratio for gathers indexed by `index`."""
        mmaps_1 = self.numerator.construct_map(coords, values[self.numerator.header_cols], coords_cols=coords_cols,
                                               index=index, index_cols=index_cols)
        mmaps_2 = self.denominator.construct_map(coords, values[self.denominator.header_cols], coords_cols=coords_cols,
                                                 index=index, index_cols=index_cols)

        mmaps_1.index_data.drop(columns=mmaps_1.coords_cols, inplace=True)
        ratio_df = mmaps_1.index_data.merge(mmaps_2.index_data, on=mmaps_1.index_cols)
        ratio_df[self.name] = ratio_df[self.numerator.name] / ratio_df[self.denominator.name]
        coords = ratio_df[coords.columns]
        values = ratio_df[self.name]
        index = ratio_df[index.columns]
        return super().construct_map(coords, values, index=index, agg=agg, bin_size=bin_size,
                                     calculate_immediately=calculate_immediately)

    def plot(self, ax, coords, index, threshold=None, top_ax_y_scale=None, bad_only=False, **kwargs):
        """Plot the gather sorted by offset with two masks over the gather plot. The lime-colored mask represents the
        window where `self.numerator` metric was calculated, while the magenta-colored mask is intended for a
        `self.denominator` window. Additionally, tracewise ratio of `self.numerator` by `self.denominator` indicators
        values is displayed on the top of the gather plot."""
        threshold = self.threshold if threshold is None else threshold
        top_ax_y_scale = self.top_ax_y_scale if top_ax_y_scale is None else top_ax_y_scale
        _ = coords
        gather = self.survey.get_gather(index).sort("offset")

        squares_numerator, nums_numerator = self.numerator(gather, return_rms=False)
        squares_denominator, nums_denominator = self.denominator(gather, return_rms=False)

        tracewise_numerator = np.sqrt(squares_numerator / nums_numerator)
        tracewise_denominator = np.sqrt(squares_denominator / nums_denominator)
        tracewise_metric = tracewise_numerator / tracewise_denominator
        tracewise_metric[tracewise_metric == 0] = np.nan

        if bad_only:
            bin_mask = self.binarize(tracewise_metric, threshold)
            gather.data[self.aggregate(bin_mask) == 0] = np.nan

        gather.plot(ax=ax, top_header=tracewise_metric, **kwargs)
        top_ax = ax.figure.axes[1]
        if threshold is not None:
            self._plot_threshold(ax=top_ax, threshold=threshold)
        top_ax.set_yscale(top_ax_y_scale)

        self.numerator.add_mask_on_plot(ax=ax, gather=gather, color="lime", legend="numerator window")
        self.denominator.add_mask_on_plot(ax=ax, gather=gather, color="magenta", legend="denominator window")
        ax.legend()

    def get_views(self, threshold=None, top_ax_y_scale=None, **kwargs):
        """Return two plotters of the metric views. Each view plots a gather with a metric values shown on top of the
        gather plot. The y-axis of the metric plot is scaled by `top_ax_y_scale`. The first view plots full gather with
        bad traces highlighted based on the `threshold` and the `self.is_lower_better` attribute. The second view only
        displays the traces defined by the metric as bad ones."""
        plot_kwargs = {"threshold": threshold, "top_ax_y_scale": top_ax_y_scale}
        return [partial(self.plot, **plot_kwargs), partial(self.plot, bad_only=True, **plot_kwargs)], kwargs


class WindowRMS(BaseWindowRMSMetric):
    """Compute traces RMS for provided window by offsets and times.

    Parameters
    ----------
    offsets : array-like with 2 int
        Offset range to use for calcualtion, measured in meters.
    times : array-like with 2 int
        Time range to use for calcualtion, measured in ms.
    name : str, optional, defaults to "window_rms"
        Metrics name.
    """
    name = "window_rms"
    is_lower_better = None
    threshold = None

    def __init__(self, offsets, times, name=None):
        if len(offsets) != 2:
            raise ValueError(f"`offsets` must contain 2 elements, not {len(offsets)}")

        if len(times) != 2:
            raise ValueError(f"`times` must contain 2 elements, not {len(times)}")

        super().__init__(name=name)
        self.offsets = np.array(offsets)
        self.times = np.array(times)

    def __repr__(self):
        """String representation of the metric."""
        return f"{type(self).__name__}(name='{self.name}', offsets='{self.offsets}', times='{self.times}')"

    def get_mask(self, gather):
        """Compute QC indicator."""
        return self.numba_get_mask(gather.data, gather.samples, gather.offsets, self.times, self.offsets,
                                   self._get_time_ixs, self.compute_stats_by_ixs)

    @staticmethod
    @njit(nogil=True)
    def _get_time_ixs(times, gather_samples):
        """Convert times into indices using samples from provided gather."""
        times = np.asarray([max(gather_samples[0], times[0]), min(gather_samples[-1], times[1])])
        time_ixs = times_to_indices(times, gather_samples, round=True).astype(np.int16)
        # Include the next index to mimic the behavior of traditional software
        time_ixs[1] += 1
        return time_ixs

    @staticmethod
    @njit(nogil=True)
    def numba_get_mask(traces, gather_samples, gather_offests, times, offsets, _get_time_ixs,
                       compute_stats_by_ixs):
        """Compute QC indicator in parallel."""
        time_ixs = _get_time_ixs(times, gather_samples)

        window_ixs = (gather_offests >= offsets[0]) & (gather_offests <= offsets[1])
        start_ixs = np.full(sum(window_ixs), fill_value=time_ixs[0], dtype=np.int16)
        end_ixs = np.full(sum(window_ixs), fill_value=time_ixs[1], dtype=np.int16)
        squares = np.zeros_like(traces[:, 0])
        nums = np.zeros_like(traces[:, 0])
        window_squares, window_nums = compute_stats_by_ixs(traces[window_ixs], start_ixs, end_ixs)
        squares[window_ixs] = window_squares
        nums[window_ixs] = window_nums
        return squares, nums

    def add_mask_on_plot(self, ax, gather, color="lime", legend=None):
        """Plot a rectangle path over the gather plot in a place where metric was computed."""
        times = self._get_time_ixs(self.times, gather.samples)

        offs_ind = np.nonzero((gather.offsets >= self.offsets[0]) & (gather.offsets <= self.offsets[1]))[0]
        if len(offs_ind) > 0:
            n_rec = (offs_ind[0], times[0]), len(offs_ind), (times[1] - times[0])
            ax.add_patch(patches.Rectangle(*n_rec, linewidth=2, edgecolor=color, facecolor='none', label=legend))


class AdaptiveWindowRMS(BaseWindowRMSMetric):
    """TODO: rewrite Signal to Noise RMS ratio computed in sliding windows along provided refractor velocity.
    RMS will be computed in two windows for every gather:
    1. Window shifted up from refractor velocity by `shift_up` ms. RMS in this window represents the noise value.
    2. WIndow shifted down from refractor velocity by `shift_down` ms`. RMS in this window represents the signal value.

    Only traces that contain noise and signal windows of the provided `window_size` are considered,
    the metric is 0 for other traces.


    Parameters
    ----------
    window_size : int
        Length of the windows for computing signam and noise RMS amplitudes measured in ms.
    shift_up : int
        The delta between noise window end and first breaks, measured in ms.
    shift_down : int
        The delta between signal window beginning and first breaks, measured in ms.
    refractor_velocity: RefractorVelocity
        Refractor velocity object to find times along witch
    name : str, optional, defaults to "adaptive_rms"
        Metrics name.
    """
    name = "adaptive_rms"
    is_lower_better = False
    threshold = None

    def __init__(self, window_size, shift, refractor_velocity, name=None):
        super().__init__(name=name)
        self.window_size = window_size
        self.shift = shift
        self.refractor_velocity = refractor_velocity

    def __repr__(self):
        """String representation of the metric."""
        repr_str = f"(name='{self.name}', window_size='{self.window_size}', shift='{self.shift}', "\
                   f"refractor_velocity='{self.refractor_velocity}')"
        return f"{type(self).__name__}" + repr_str

    def get_mask(self, gather):
        """Compute QC indicator."""
        fbp_times = self.refractor_velocity(gather.offsets)
        return self.numba_get_mask(gather.data, self._get_indices, self.compute_stats_by_ixs,
                                   window_size=self.window_size, shift=self.shift, samples=gather.samples,
                                   fbp_times=fbp_times, times_to_indices=times_to_indices)

    @staticmethod
    @njit(nogil=True)
    def numba_get_mask(traces, _get_indices, compute_stats_by_ixs, window_size, shift, samples, fbp_times,
                       times_to_indices):
        """Compute QC indicator in parallel."""
        start_ixs, end_ixs = _get_indices(window_size, shift, samples, fbp_times, times_to_indices)
        return compute_stats_by_ixs(traces, start_ixs, end_ixs)

    @staticmethod
    @njit(nogil=True)
    def _get_indices(window_size, shift, samples, fbp_times, times_to_indices):
        """Convert times to use for noise and signal windows into indices"""
        mid_samples = times_to_indices(fbp_times + shift, samples, round=True).astype(np.int16)
        window_size = int(times_to_indices(np.array([window_size]), samples, round=True)[0])

        start_ixs = np.clip(mid_samples - (window_size - window_size // 2), 0, len(samples))
        end_ixs = np.clip(mid_samples + (window_size // 2), 0, len(samples))
        return start_ixs, end_ixs

    def add_mask_on_plot(self, ax, gather, color="lime", legend=None):
        """Gather plot sorted by offset with tracewise indicator on a separate axis and signal and noise windows."""
        fbp_times = self.refractor_velocity(gather.offsets)
        indices = self._get_indices(self.window_size, self.shift, gather.samples, fbp_times, times_to_indices)
        indices = np.where(np.asarray(indices) == 0, np.nan, indices)
        indices = np.where(np.asarray(indices) == np.nanmax(indices), np.nan, indices)

        ax.plot(np.arange(gather.n_traces), indices[0], color=color, label=legend)
        ax.plot(np.arange(gather.n_traces), indices[1], color=color)

DEFAULT_TRACEWISE_METRICS = [TraceAbsMean, TraceMaxAbs, MaxClipsLen, MaxConstLen, DeadTrace]
