# pylint: disable=not-an-iterable
"""Implements metrics for quality control of the survey.

These metrics are supposed to be used in :func:`~survey.qc` method, which iterates over traces or group of traces and
automatically provides metrics with all required context for interactive plotting.

To define your own metric, you need to inherit a new class from either `TracewiseMetric` or
`MuteTracewiseMetric`, or `BaseWindowRMSMetric` depending on the purpose, and do the following:
* Redefine `get_values` or `numba_get_values` method. Since all metrics are calculated in threads, it is important to
  release GIL as early as possible. Try not to use `get_values` at all, or only use it to calculate the part that can't
  be done without releasing the GIL. The main part should be implemented in `numba_get_values` with njit decorator and
  flag `nogil=True`.
* Redefine `description` method, which describes what traces were detected by the metric.
* Set a `threshold` class attribute to a number above or below which the trace will be considered bad. If an
  `is_lower_better` class attribute is `True`, the values greater or equal to the `threshold` will be considered bad
  and lower or equal otherwise.
* Optionally redefine `preprocess` method, which accepts the gather and applies any preprocess procedures such as
  muting or scaling. This gather will later used in `get_values` and showed on the right side of the interactive plot.
* Optionally define all other class attributes of `Metric` for future convenience.
* Optionally redefine `plot` method which will be used to plot gather with tracewise metric on top when a metric map is
  clicked in interactive mode. It should accept an instance of `matplotlib.axes.Axes` to plot on, `coords` and `index`
  for gather that will be plotted, all arguments defined in `self.get_views` and kwargs for gather.plot. By default the
  method plots the gather with tracewise metric on top and a red mask on top of the gather plot, highlighted bad parts
  of the gather according to the metric.

If you want the created metric to be calculated by :func:`~survey.qc` method by default, it should also be appended to
a `DEFAULT_TRACEWISE_METRICS` list.
"""

import warnings
from textwrap import wrap
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
        """Column names in `survey.headers` for which metric was created."""
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
    """Base class for tracewise metrics with plotters and aggregations. Child classes should redefine `get_values` or
    `numba_get_values` and `description` methods, and optionally `preprocess`."""
    threshold = None
    top_y_ax_scale = "linear"

    def __call__(self, gather):
        """Compute qc metric by applying sequentially `self.preprocess`, `self.get_values` and `self.aggregate` to
        provided gather."""
        gather = self.preprocess(gather)
        mask = self.get_values(gather)
        return self.aggregate(mask)

    @property
    def description(self):
        """String description of the tracewise metric. Mainly used in `self.describe` when describing the number of bad
        traces detected by the metric."""
        return NotImplementedError

    def describe(self, metric_values, line_width=55, separator="\n"):
        """Provide a description about the number of bad values for the passed metric values in a string format. Each
        line in the resulting string will not exceed `line_width` and will be separated by `separator`."""
        bin_values = self.binarize(metric_values)
        # Process multiline descriptions and set the distance from the last line to the result to `line_width` symbols
        desc_list = wrap(self.description, width=line_width-1)
        last_line = f"{desc_list[-1]+':':<{line_width}}"
        description = separator.join(desc_list[:-1]) + separator + last_line if len(desc_list) > 1 else last_line
        return f"{description}{bin_values.sum()} ({100 * bin_values.mean():.3f}%)"

    def preprocess(self, gather):
        """Preprocess gather before either calling `self.get_values` method to calculate metric or to plot the gather.
        Identity by default."""
        _ = self
        return gather

    def get_values(self, gather):  # get_values, compute_metric
        """Compute QC indicator.

        There are two possible outputs for the provided gather:
            1. Samplewise indicator with the same shape as `gather`.
            2. Tracewise indicator with a shape of (`gather.n_traces`,).

        The method redirects the call to njitted static `numba_get_values` method. Either this method or
        `numba_get_values` must be overridden in child classes.
        """
        return self.numba_get_values(gather.data)

    @staticmethod
    @njit(nogil=True)
    def numba_get_values(traces):
        """Compute njitted QC indicator."""
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
        """Binarize a given mask based on the provided threshold.

        Parameters
        ----------
        mask : 1d ndarray or 2d ndarray
            Array with computed metric values to be converted to a binary mask.
        threshold : int, float, array-like with 2 elements, optional, defaults to None
            Value to use as a threshold for binarizing the mask.
            If a single number and `self.is_lower_better` is True, mask values greater or equal than the `threshold`
            will be treated as bad and marked as True, otherwise, if `self.is_lower_better` is False, mask values lower
            or equal then the `threshold` will be treated as bad and marked as True.
            If array, two numbers indicate the boundaries within which the metric values are treated as False, outside
            inclusive - as True.
            If None, self.threshold will be used.

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

        if isinstance(threshold, (int, float, np.number)):
            if self.is_lower_better is None:
                raise ValueError("`threshold` cannot be single number if `is_lower_better` is None")
            bin_fn = np.greater_equal if self.is_lower_better else np.less_equal
            bin_mask = bin_fn(mask, threshold)
        elif len(threshold) != 2:
            raise ValueError(f"`threshold` should contain exactly 2 elements, not {len(threshold)}")
        else:
            bin_mask = (mask <= threshold[0]) | (mask >= threshold[1])
        bin_mask[np.isnan(mask)] = False
        return bin_mask

    def plot(self, ax, coords, index, sort_by=None, threshold=None, top_y_ax_scale=None,  bad_only=False, **kwargs):
        """Plot gather by its `index` with highlighted traces with metric value above or below the `self.threshold`.

        Tracewise metric values will be shown on top of the gather plot. Also, the area with `good` metric values based
        on threshold values and `self.is_lower_better` will be highlighted in blue.

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
            Scale type for top header plot, see `matplotlib.axes.Axes.set_yscale` for available options.
        bad_only : bool, optional, defaults to False
            Show only traces that are considered bad based on provided threshold and `self.is_lower_better`.
        kwargs : misc, optional
            Additional keyword arguments to the `gather.plot`.
        """
        threshold = self.threshold if threshold is None else threshold
        top_y_ax_scale = self.top_y_ax_scale if top_y_ax_scale is None else top_y_ax_scale
        _ = coords

        gather = self.survey.get_gather(index)
        if sort_by is not None:
            gather = gather.sort(sort_by)
        gather = self.preprocess(gather)

        metric_res = self.get_values(gather)
        metric_vals = self.aggregate(metric_res)
        mask = self.binarize(metric_res, threshold)

        mode = kwargs.pop("mode", "wiggle")
        masks_dict = {"masks": mask, "alpha": 0.8, "label": self.name or "metric", **kwargs.pop("masks", {})}

        if bad_only:
            bad_mask = self.aggregate(mask) == 0
            gather.data[bad_mask] = np.nan
            metric_vals[bad_mask] = np.nan
            # Don't need to plot 1d mask if only bad traces will be plotted.
            if mask.ndim == 1:
                masks_dict = None

        gather.plot(ax=ax, mode=mode, top_header=metric_vals, masks=masks_dict, **kwargs)
        top_ax = ax.figure.axes[1]
        top_ax.set_yscale(top_y_ax_scale)
        if threshold is not None:
            self._plot_threshold(ax=top_ax, threshold=threshold)

    def _plot_threshold(self, ax, threshold):
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        if isinstance(threshold, (int, float, np.number)):
            if self.is_lower_better is None:
                raise ValueError("`threshold` cannot be single number if `is_lower_better` is None")
            threshold = [threshold, y_max] if self.is_lower_better else [y_min, threshold]
        ax.fill_between(np.arange(x_min, x_max), *threshold, alpha=0.3, color="blue")

    def get_views(self, sort_by=None, threshold=None, top_y_ax_scale=None, **kwargs):
        """Return two plotters of the metric views. Each view plots a gather sorted by `sort_by` with a metric values
        shown on top of the gather plot. The y-axis of the metric plot is scaled by `top_y_ax_scale`. The first view
        plots full gather with bad traces highlighted based on the `threshold` and the `self.is_lower_better`
        attribute. The second view only displays the traces defined by the metric as bad ones."""
        plot_kwargs = {"sort_by": sort_by, "threshold": threshold, "top_y_ax_scale": top_y_ax_scale}
        return [partial(self.plot, **plot_kwargs), partial(self.plot, bad_only=True, **plot_kwargs)], kwargs


class DeadTrace(TracewiseMetric):
    """Detect constant traces.

    `get_values` returns 1d binary mask where each constant trace is marked with one.
    """
    min_value = 0
    max_value = 1
    is_lower_better = True
    threshold = 0.5

    @property
    def description(self):
        """String description of the tracewise metric."""
        return "Number of dead traces"

    @staticmethod
    @njit(nogil=True)
    def numba_get_values(traces):
        """Compute njitted QC indicator."""
        res = np.empty_like(traces[:, 0])
        for i in range(traces.shape[0]):
            res[i] = isclose(max(traces[i]), min(traces[i]))
        return res


class TraceAbsMean(TracewiseMetric):
    """Calculate absolute value of the trace's mean scaled by trace's std.

    `get_values` returns 1d array with computed metric values for the gather.
    """
    is_lower_better = True
    threshold = 0.1

    @property
    def description(self):
        """String description of the tracewise metric."""
        return f"Traces with mean divided by std greater than {self.threshold}"

    @staticmethod
    @njit(nogil=True)
    def numba_get_values(traces):
        """Compute njitted QC indicator."""
        res = np.empty_like(traces[:, 0])
        for i in range(traces.shape[0]):
            res[i] = np.abs(traces[i].mean() / (traces[i].std() + 1e-10))
        return res


class TraceMaxAbs(TracewiseMetric):
    """Find a maximum absolute amplitude value scaled by trace's std.

    `get_values` returns 1d array with computed metric values for the gather.
    """
    is_lower_better = True
    threshold = 15

    @property
    def description(self):
        """String description of tracewise metric"""
        return f"Traces with max abs to std ratio greater than {self.threshold}"

    @staticmethod
    @njit(nogil=True)
    def numba_get_values(traces):
        """Compute njitted QC indicator."""
        res = np.empty_like(traces[:, 0])
        for i in range(traces.shape[0]):
            res[i] = np.max(np.abs(traces[i])) / (traces[i].std() + 1e-10)
        return res


class MaxClipsLen(TracewiseMetric):
    """Calculate the length of consecutive amplitudes equals to trace minimum or maximum amplitudes.

    `get_values` returns 2d array with values indicating the length of consecutive maximum or minimum amplitudes for
    each trace in the input gather.
    """
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
    def numba_get_values(traces):
        """Compute njitted QC indicator."""
        def _update_counters(sample, counter, container, value):
            if isclose(sample, value):
                counter += 1
            else:
                if counter > 1:
                    container[:] = counter
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
                max_counter = _update_counters(trace[j], max_counter, maxes[i, j - max_counter: j], max_val)
                min_counter = _update_counters(trace[j], min_counter, mins[i, j - min_counter: j], min_val)

            if max_counter > 1:
                maxes[i, -max_counter:] = max_counter
            if min_counter > 1:
                mins[i, -min_counter:] = min_counter
        return maxes + mins


class MaxConstLen(TracewiseMetric):
    """Calculate the number of consecutive identical amplitudes.

    `get_values` returns a 2d array with values indicating the length of consecutive identical values in each trace in
    the input gather.
    """
    is_lower_better = True
    threshold = 4

    @property
    def description(self):
        """String description of tracewise metric"""
        return f"Traces with more than {self.threshold} identical values in a row"

    @staticmethod
    @njit(nogil=True)
    def numba_get_values(traces):
        """Compute njitted QC indicator."""
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


class MuteTracewiseMetric(TracewiseMetric):  # pylint: disable=abstract-method
    """Base class for tracewise metric with implemented `self.preprocess` method which applies muting and standard
    scaling to the input gather. Child classes should redefine `get_values` or `numba_get_values` methods.

    Parameters
    ----------
    muter : Muter
        A muter to use.
    name : str, optional, defaults to None
        A metric name.
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
    """Spikes detection. The metric reacts to drastic changes in traces amplitudes within a 1-width window around each
    amplitude value.

    `get_values` returns 2d array with values that show the deviation of the amplitudes of an input gather.

    The metric is highly dependent on the muter being used; if muter is not strong enough, the metric will overreact
    to the first breaks.
    """
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
    def numba_get_values(traces):
        """Compute njitted QC indicator."""
        res = np.zeros_like(traces)
        for i in range(traces.shape[0]):
            nan_indices = np.nonzero(np.isnan(traces[i]))[0]
            j = nan_indices[-1] + 1 if len(nan_indices) > 0 else 0
            res[i, j+1: -1] = np.abs(traces[i, j+2:] + traces[i, j:-2] - 2*traces[i, j+1:-1]) / 3
        return res


class Autocorrelation(MuteTracewiseMetric):
    """Trace correlation with itself shifted by 1.

    `get_values` returns 1d array with values that show mean trace autocorrelation. If proportion of nans in the trace
    is greater than `nan_ratio`, then the metric value for this trace will be nan.

    The metric is highly dependent on the muter being used; if muter is not strong enough, the metric will overreact
    to the first breaks.

    Parameters
    ----------
    muter : Muter
        A muter to use.
    name : str, optional, defaults to "Autocorrelation"
        A metric name.
    nan_ratio : float, optional, defaults to 0.95
        The maximum proportion of nan values allowed in a trace.
    """
    min_value = -1
    max_value = 1
    is_lower_better = False
    threshold = 0.8

    def __init__(self, muter, name=None, nan_ratio=0.95):
        super().__init__(muter=muter, name=name)
        self.nan_ratio = nan_ratio

    def __repr__(self):
        """String representation of the metric."""
        return f"{type(self).__name__}(name='{self.name}', muter='{self.muter}', nan_ratio='{self.nan_ratio}')"

    @property
    def description(self):
        """String description of tracewise metric."""
        return f"Traces with autocorrelation less than {self.threshold}"

    def get_values(self, gather):
        return self.numba_get_values(gather.data, nan_ratio=self.nan_ratio)

    @staticmethod
    @njit(nogil=True)
    def numba_get_values(traces, nan_ratio):
        """Compute njitted QC indicator."""
        res = np.empty_like(traces[:, 0])
        for i in range(traces.shape[0]):
            if np.isnan(traces[i]).sum() > nan_ratio*len(traces[i]):
                res[i] = np.nan
            else:
                res[i] = np.nanmean(traces[i, 1:] * traces[i, :-1])
        return res


class BaseWindowRMSMetric(TracewiseMetric):  # pylint: disable=abstract-method
    """Base class for the tracewise metrics that computes RMS in window defined by two arrays with start and end
    indices for each trace in provided gather.

    Child classes should redefine `get_values` or `numba_get_values` methods.
    """

    def __call__(self, gather, return_rms=True):
        """Compute the metric by applying `self.preprocess` and `self.get_values` to provided gather.
        If `return_rms` is True, the RMS value for provided gather will be returned.
        Otherwise, two 1d arrays will be returned:
            1. Sum of squares of amplitudes in the defined window for each trace,
            2. Number of amplitudes in a specified window for each trace.
        """
        gather = self.preprocess(gather)
        squares, nums = self.get_values(gather)
        if return_rms:
            return self.compute_rms(squares, nums)
        return squares, nums

    def _get_description_bundle(self):
        if isinstance(self.threshold, (int, float, np.number)):
            if self.is_lower_better is None:
                raise ValueError("`threshold` cannot be single number if `is_lower_better` is None")
            bundle = "greater or equal then" if self.is_lower_better else "less or equal then"
            return bundle + f" {self.threshold}"
        return f"between {self.threshold[0]} and {self.threshold[1]}"

    @property
    def header_cols(self):
        """Column names in `survey.headers` to store the metrics results in."""
        return [self.name+"_sum", self.name+"_n"]

    @staticmethod
    def compute_rms(squares, nums):
        """Compute the RMS using provided squares of amplitudes and the number of amplitudes used for square
        calculation."""
        return np.sqrt(np.sum(squares) / np.sum(nums))

    @staticmethod
    @njit(nogil=True)
    def compute_stats_by_ixs(data, start_ixs, end_ixs):
        """Compute the sum of squares and the number of elements in a window specified by `start_ixs` and `end_ixs`
        for each trace in provided data."""
        sum_squares = np.empty_like(data[:, 0])
        nums = np.empty_like(data[:, 0])

        for i, (trace, start_ix, end_ix) in enumerate(zip(data, start_ixs, end_ixs)):
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
        if sum_square_map.index_cols != sum_square_map.coords_cols:
            sum_square_map.index_data.drop(columns=sum_square_map.coords_cols, inplace=True)
        sum_df = sum_square_map.index_data.merge(nums_map.index_data, on=nums_map.index_cols)
        sum_df[self.name] = np.sqrt(sum_df[self.name+"_x"] / sum_df[self.name+"_y"])
        return super().construct_map(sum_df[nums_map.coords_cols], sum_df[self.name],
                                     index=sum_df[nums_map.index_cols], agg=agg, bin_size=bin_size,
                                     calculate_immediately=calculate_immediately)

    def plot(self, ax, coords, index, threshold=None, top_y_ax_scale=None, bad_only=False, color="lime", **kwargs):  # pylint: disable=arguments-renamed
        """Plot the gather sorted by offset with tracewise indicator on the top of the gather plot. Any mask can be
        displayed over the gather plot using `self.add_mask_on_plot`."""
        threshold = self.threshold if threshold is None else threshold
        top_y_ax_scale = self.top_y_ax_scale if top_y_ax_scale is None else top_y_ax_scale
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
        top_ax.set_yscale(top_y_ax_scale)
        self.add_mask_on_plot(ax=ax, gather=gather, color=color)

    def add_mask_on_plot(self, ax, gather, color=None):
        """Plot any additional metric related graphs over the gather plot."""
        _ = self, ax, gather, color
        pass

    def get_views(self, threshold=None, top_y_ax_scale=None, **kwargs):
        """Return two plotters of the metric views. Each view plots a gather sorted by `offset` with a metric values
        shown on top of the gather plot. The y-axis of the metric plot is scaled by `top_y_ax_scale`. The first view
        plots full gather with bad traces highlighted based on the `threshold` and the `self.is_lower_better`
        attribute. The second view only displays the traces defined by the metric as bad ones."""
        plot_kwargs = {"threshold": threshold, "top_y_ax_scale": top_y_ax_scale}
        return [partial(self.plot, **plot_kwargs), partial(self.plot, bad_only=True, **plot_kwargs)], kwargs


class MetricsRatio(TracewiseMetric):  # pylint: disable=abstract-method
    """Calculate the ratio of two window RMS metrics.

    In the metric map, the displayed values are obtained by dividing the  value of `self.numerator` metric by the
    value of `self.denominator` metric for each gather independently.

    Parameters
    ----------
    numerator : BaseWindowRMSMetric or its subclass
        Metric instance whose values will be divided by the `denominator` metric values.
    denominator : BaseWindowRMSMetric or its subclass
        Metric instance whose values will be used as a divisor for a `numerator` metric.
    name : str, optional, defaults to "numerator.name to denominator.name ratio"
        A metric name.
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

    def __repr__(self):
        """String representation of the metric."""
        kwargs_str = f"(name='{self.name}', numerator='{self.numerator}', denominator='{self.denominator}')"
        return f"{type(self).__name__}" + kwargs_str

    @property
    def header_cols(self):
        """Column names in `survey.headers` to store the metrics results in."""
        return self.numerator.header_cols + self.denominator.header_cols

    def construct_map(self, coords, values, *, coords_cols=None, index=None, index_cols=None, agg=None, bin_size=None,
                      calculate_immediately=True):
        """Construct a metric map with `self.numerator` and `self.denominator` ratio for gathers indexed by `index`."""
        mmaps_1 = self.numerator.construct_map(coords, values[self.numerator.header_cols], coords_cols=coords_cols,
                                               index=index, index_cols=index_cols)
        mmaps_2 = self.denominator.construct_map(coords, values[self.denominator.header_cols], coords_cols=coords_cols,
                                                 index=index, index_cols=index_cols)
        if mmaps_1.index_cols != mmaps_1.coords_cols:
            mmaps_1.index_data.drop(columns=mmaps_1.coords_cols, inplace=True)
        ratio_df = mmaps_1.index_data.merge(mmaps_2.index_data, on=mmaps_2.index_cols)
        ratio_df[self.name] = ratio_df[self.numerator.name] / ratio_df[self.denominator.name]
        coords = ratio_df[mmaps_1.coords_cols]
        values = ratio_df[self.name]
        index = ratio_df[mmaps_1.index_cols]
        return super().construct_map(coords, values, index=index, agg=agg, bin_size=bin_size,
                                     calculate_immediately=calculate_immediately)

    def plot(self, ax, coords, index, threshold=None, top_y_ax_scale=None, bad_only=False, **kwargs):
        """Plot the gather sorted by offset with two masks over the gather plot. The lime-colored mask represents the
        window where the `self.numerator` metric was calculated, while the magenta-colored mask represents the
        `self.denominator` window. Additionally, tracewise ratio of `self.numerator` by `self.denominator` is displayed
        on the top of the gather plot."""
        threshold = self.threshold if threshold is None else threshold
        top_y_ax_scale = self.top_y_ax_scale if top_y_ax_scale is None else top_y_ax_scale
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
        top_ax.set_yscale(top_y_ax_scale)

        self.numerator.add_mask_on_plot(ax=ax, gather=gather, color="lime", legend=f"{self.numerator.name} window")
        self.denominator.add_mask_on_plot(ax=ax, gather=gather, color="magenta",
                                          legend=f"{self.denominator.name} window")
        ax.legend()

    def get_views(self, threshold=None, top_y_ax_scale=None, **kwargs):
        """Return two plotters of the metric views. Each view plots a gather sorted by `offset` with a metric values
        shown on top of the gather plot. The y-axis of the metric plot is scaled by `top_y_ax_scale`. The first view
        plots full gather with bad traces highlighted based on the `threshold` and the `self.is_lower_better`
        attribute. The second view only displays the traces defined by the metric as bad ones."""
        plot_kwargs = {"threshold": threshold, "top_y_ax_scale": top_y_ax_scale}
        return [partial(self.plot, **plot_kwargs), partial(self.plot, bad_only=True, **plot_kwargs)], kwargs


class WindowRMS(BaseWindowRMSMetric):
    """Compute traces RMS for provided window by offsets and times.

    Parameters
    ----------
    offsets : array-like with 2 ints
        Offset range to use for calculation, measured in meters.
    times : array-like with 2 ints
        Time range to use for calculation, measured in ms.
    name : str, optional, defaults to "WindowRMS"
        A metric name.
    """
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

    @property
    def description(self):
        """String description of the tracewise metric."""
        bundle = self._get_description_bundle()
        return f"Traces within a window by offsets {self.offsets} and times {self.times} with RMS {bundle}"

    def describe(self, metric_values, line_width=55, separator="\n"):
        """Provide a description about the number of bad values for the passed metric values in a string format. Each
        line in the resulting string will not exceed `line_width` and will be separated by `separator`.
        If `self.threshold` is None, the average RMS for the passed values will be showed."""
        metric_value = np.sqrt(metric_values[:, 0] / metric_values[:, 1])
        if self.threshold is None:
            description = "Mean of traces RMS within a window by" + separator
            description += f"{f'offsets {self.offsets} and times {self.times}:':<{line_width}}"
            return description + f"{np.nanmean(metric_value):<.3f}"
        return super().describe(metric_value, line_width=line_width, separator=separator)

    def get_values(self, gather):
        """Compute QC indicator."""
        return self.numba_get_values(gather.data, gather.samples, gather.offsets, self.times, self.offsets,
                                   self._get_time_ixs, self.compute_stats_by_ixs)

    @staticmethod
    @njit(nogil=True)
    def _get_time_ixs(times, gather_samples):
        """Convert times into indices using samples from provided gather."""
        times = np.asarray([max(gather_samples[0], times[0]), min(gather_samples[-1], times[1])])
        time_ixs = times_to_indices(times, gather_samples, round=True).astype(np.int16)
        # Include the next index to mimic the behavior of conventional software
        time_ixs[1] += 1
        return time_ixs

    @staticmethod
    @njit(nogil=True)
    def numba_get_values(traces, gather_samples, gather_offsets, times, offsets, _get_time_ixs,
                       compute_stats_by_ixs):
        """Compute njitted QC indicator."""
        time_ixs = _get_time_ixs(times, gather_samples)

        window_ixs = (gather_offsets >= offsets[0]) & (gather_offsets <= offsets[1])
        start_ixs = np.full(sum(window_ixs), fill_value=time_ixs[0], dtype=np.int16)
        end_ixs = np.full(sum(window_ixs), fill_value=time_ixs[1], dtype=np.int16)
        squares = np.zeros_like(traces[:, 0])
        nums = np.zeros_like(traces[:, 0])
        window_squares, window_nums = compute_stats_by_ixs(traces[window_ixs], start_ixs, end_ixs)
        squares[window_ixs] = window_squares
        nums[window_ixs] = window_nums
        return squares, nums

    def add_mask_on_plot(self, ax, gather, color="lime", legend=None):
        """Plot a rectangle path over the gather plot in a place where RMS was computed."""
        times = self._get_time_ixs(self.times, gather.samples)

        offs_ind = np.nonzero((gather.offsets >= self.offsets[0]) & (gather.offsets <= self.offsets[1]))[0]
        if len(offs_ind) > 0:
            n_rec = (offs_ind[0], times[0]), len(offs_ind), (times[1] - times[0])
            ax.add_patch(patches.Rectangle(*n_rec, linewidth=2, edgecolor=color, facecolor='none', label=legend))


class AdaptiveWindowRMS(BaseWindowRMSMetric):
    """Compute traces RMS in sliding window along provided refractor velocity.

    For each gather, the RMS value will be calculated within a window of size `window_size` centered around the
    refractor velocity shifted by `shift`.

    Only the traces that contain at least single amplitude in the provided window are considered, otherwise the metric
    is nan.

    Parameters
    ----------
    window_size : int
        Length of the window for computing RMS amplitudes, measured in ms.
    shift : int
        The distance to shift the sliding window from the refractor velocity, measured in ms.
    refractor_velocity: RefractorVelocity
        Refractor velocity object to find times along witch the RMS will be calculated.
    name : str, optional, defaults to "AdaptiveWindowRMS"
        A metric name.
    """
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

    @property
    def description(self):
        """String description of the tracewise metric."""
        bundle = self._get_description_bundle()
        return f"Traces with a RMS computed along a RV with shift {self.shift} "\
               f"and window size {self.window_size} {bundle}"

    def describe(self, metric_values, line_width=55, separator="\n"):
        """Provide a description about the number of bad values for the passed metric values in a string format. Each
        line in the resulting string will not exceed `line_width` and will be separated by `separator`.
        If `self.threshold` is None, the average RMS for the passed values will be showed."""
        metric_value = np.sqrt(metric_values[:, 0] / metric_values[:, 1])
        if self.threshold is None:
            description = f"Mean of traces RMS computed along a RV with shift {self.shift}" + separator
            description += f"{f'and window size {self.window_size}:':<{line_width}}"
            return description + f"{np.nanmean(metric_value):<.3f}"
        return super().describe(metric_value, line_width=line_width, separator=separator)

    def get_values(self, gather):
        """Compute QC indicator."""
        fbp_times = self.refractor_velocity(gather.offsets)
        return self.numba_get_values(gather.data, self._get_indices, self.compute_stats_by_ixs,
                                   window_size=self.window_size, shift=self.shift, samples=gather.samples,
                                   fbp_times=fbp_times, times_to_indices=times_to_indices)

    @staticmethod
    @njit(nogil=True)
    def numba_get_values(traces, _get_indices, compute_stats_by_ixs, window_size, shift, samples, fbp_times,
                       times_to_indices):  # pylint: disable=redefined-outer-name
        """Compute njitted QC indicator."""
        start_ixs, end_ixs = _get_indices(window_size, shift, samples, fbp_times, times_to_indices)
        return compute_stats_by_ixs(traces, start_ixs, end_ixs)

    @staticmethod
    @njit(nogil=True)
    def _get_indices(window_size, shift, samples, fbp_times, times_to_indices):  # pylint: disable=redefined-outer-name
        """Calculates the start and end indices of a window of size `window_size` centered around the refractor
        velocity shifted by `shift`."""
        mid_times = fbp_times + shift
        start_times = np.clip(mid_times - (window_size - window_size // 2), 0, samples[-1])
        end_times = np.clip(mid_times + (window_size // 2), 0, samples[-1])

        start_ixs = times_to_indices(start_times, samples, round=True).astype(np.int32)
        end_ixs = times_to_indices(end_times, samples, round=True).astype(np.int32)
        return start_ixs, end_ixs

    def add_mask_on_plot(self, ax, gather, color="lime", legend=None):
        """Plot two parallel lines over the gather plot along the window where RMS was computed."""
        fbp_times = self.refractor_velocity(gather.offsets)
        indices = self._get_indices(self.window_size, self.shift, gather.samples, fbp_times, times_to_indices)
        indices = np.where(np.asarray(indices) == 0, np.nan, indices)
        indices = np.where(np.asarray(indices) == np.nanmax(indices), np.nan, indices)

        ax.plot(np.arange(gather.n_traces), indices[0], color=color, label=legend)
        ax.plot(np.arange(gather.n_traces), indices[1], color=color)

DEFAULT_TRACEWISE_METRICS = [DeadTrace, TraceAbsMean, TraceMaxAbs, MaxClipsLen, MaxConstLen]
