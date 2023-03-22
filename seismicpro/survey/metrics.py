# pylint: disable=not-an-iterable
"""Implements survey metrics"""
import warnings
from functools import partial

import numpy as np
from numba import njit, prange
from matplotlib import patches

from ..metrics import Metric
from ..utils import times_to_indices

# Ignore all warnings related to empty slices or dividing by zero
warnings.simplefilter("ignore", category=RuntimeWarning)


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
        """Compute metric by applying `self.preprocess`, `self.get_mask` and `self.aggregate` to provided gather."""
        gather = self.preprocess(gather)
        mask = self.get_mask(gather)
        return self.aggregate(mask)

    @property
    def header_cols(self):
        """Column names in survey.headers to srote metrics results."""
        return self.name

    @property
    def description(self):
        """String description of tracewise metric"""
        return self.name

    def preprocess(self, gather):
        """Preprocess gather before calculating metric. Identity by default."""
        _ = self
        return gather

    def get_mask(self, gather):
        """QC indicator implementation. Takes a gather as an argument and returns either a samplewise qc indicator with
        shape equal to `gather.shape` or a tracewize indicator with shape (`gather.n_traces`,).

        Since all metrics calculated in threads, it may be more effective to call directly numba-decorated function.
        Thus, depending on the case, implement QC indicator either here or in `self.numba_get_mask`."""
        return self.numba_get_mask(gather.data)

    @staticmethod
    @njit(nogil=True, parallel=True)
    def numba_get_mask(traces):
        """Parallel QC metric implemetation. """
        raise NotImplementedError

    def aggregate(self, mask):
        """Aggregate input mask depending on `self.is_lower_better` to select the worst mask value for each trace"""
        if self.is_lower_better is None:
            agg_fn = np.nanmean
        elif self.is_lower_better:
            agg_fn = np.nanmax
        else:
            agg_fn = np.nanmin
        return mask if mask.ndim == 1 else agg_fn(mask, axis=1)

    def binarize(self, mask, threshold=None):
        """Binarize input mask by `threshold` marking bad mask values as True. Depending on `self.is_lower_better`
        values greater or less than the `threshold` will be treated as a bad value. If `threshold` is None,
        `self.threshold` is used."""
        threshold = self.threshold if threshold is None else threshold
        if threshold is None:
            raise ValueError("Either `threshold` or `self.threshold` must be non None")

        if isinstance(threshold, (int, float, np.number)):
            if self.is_lower_better is None:
                raise ValueError("`threshold` cannot be single number if `is_lower_better` is None")
            bin_fn = np.greater_equal if self.is_lower_better else np.less_equal
            return bin_fn(mask, threshold)

        if len(threshold) != 2:
            raise ValueError(f"`threshold` should contain exactly 2 elements, not {len(threshold)}")

        return (mask < threshold[0]) | (mask > threshold[1])

    # def construct_map(self, headers, index_cols, coords_cols, **kwargs):
    #     """Construct metric map from headers base on index or coords cols and kwargs."""
    #     index = headers[index_cols] if index_cols is not None else None
    #     return super().construct_map(headers[coords_cols], headers[self.name], index=index, **kwargs)

    def plot(self, ax, coords, index, sort_by=None, threshold=None, top_ax_y_scale=None,  bad_only=False, **kwargs):
        """Gather plot where samples with indicator above/below `.threshold` are highlited."""
        threshold = self.threshold if threshold is None else threshold
        top_ax_y_scale = self.top_ax_y_scale if top_ax_y_scale is None else top_ax_y_scale
        _ = coords

        gather = self.survey.get_gather(index)
        if sort_by is not None:
            gather = gather.sort(sort_by)
        gather = self.preprocess(gather)

        # TODO: Can we do only single copy here? (first copy sometimes done in self.preprocess)
        # We need to copy gather since some metrics changes gather in get_mask, but we want to plot gather unchanged
        mask = self.get_mask(gather.copy())
        metric_vals = self.aggregate(mask)
        bin_mask = self.binarize(mask, threshold)
        if bad_only:
            gather.data[self.aggregate(bin_mask) == 0] = np.nan

        mode = kwargs.pop("mode", "wiggle")
        masks_dict = {"masks": bin_mask, "alpha": 0.8, "label": self.name or "metric", **kwargs.pop("masks", {})}
        gather.plot(ax=ax, mode=mode, top_header=metric_vals, masks=masks_dict, **kwargs)
        top_ax = ax.figure.axes[1]
        if threshold is not None:
            self._plot_threshold(ax=top_ax, threshold=threshold)
        top_ax.set_yscale(top_ax_y_scale)

    @staticmethod
    def _plot_threshold(ax, threshold):
        if isinstance(threshold, (int, float, np.number)):
            ax.axhline(threshold, alpha=0.5, color="blue")
        else:
            start, end = ax.get_xlim()
            ax.fill_between(np.arange(start+0.5, end+0.5), *threshold, alpha=0.3, color="blue")

    def get_views(self, sort_by=None, threshold=None, top_ax_y_scale=None, **kwargs):
        """Return plotters of the metric views and those `kwargs` that should be passed further to an interactive map
        plotter."""
        plot_kwargs = {"sort_by": sort_by, "threshold": threshold, "top_ax_y_scale": top_ax_y_scale}
        return [partial(self.plot, **plot_kwargs), partial(self.plot, bad_only=True, **plot_kwargs)], kwargs


class MuteTracewiseMetric(TracewiseMetric):  # pylint: disable=abstract-method
    """Base class for tracewise metric with implemented `self.preprocess` method which applies muting and standard
    scaling to the input gather. Child classes should redefine `get_mask` method."""

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
    """Spikes detection. The metric reacts to drastic changes in traces ampliutes in 1-width window around each
    amplitude value.

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

    @staticmethod
    @njit(nogil=True)
    def numba_get_mask(traces):
        """QC indicator implementation.

        The resulted 2d mask shows the deviation of the ampluteds of an input gather.
        """
        res = np.zeros_like(traces, dtype=np.float32)
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

    The metric is highly depends on muter, if muter isn't strong enough, the metric will overreact to the first breaks.

    Parameters
    ----------
    muter : Muter
        A muter to use.
    name : str, optional, defaults to "autocorrelation"
        Metrics name.
    """
    name = "autocorrelation"
    min_value = -1
    max_value = 1
    is_lower_better = False
    threshold = 0.8

    @staticmethod
    @njit(nogil=True)
    def numba_get_mask(traces):
        """QC indicator implementation."""
        # TODO: descide what to do with almost nan traces (in 98% in trace are nan, it almost always will have -1 val)
        res = np.empty(traces.shape[0], dtype=np.float32)
        for i in range(traces.shape[0]):
            res[i] = np.nanmean(traces[i, 1:] * traces[i, :-1])
        return res

class TraceAbsMean(TracewiseMetric):
    """Absolute value of the trace's mean scaled by trace's std.

    Parameters
    ----------
    name : str, optional, defaults to "trace_absmean"
        Metrics name.
    """
    name = "trace_absmean"
    is_lower_better = True
    threshold = 0.1

    @staticmethod
    @njit(nogil=True)
    def numba_get_mask(traces):
        """QC indicator implementation."""
        res = np.empty(traces.shape[0])
        for i in range(traces.shape[0]):
            res[i] = np.abs(traces[i].mean() / (traces[i].std() + 1e-10))
        return res


class TraceMaxAbs(TracewiseMetric):
    """Maximun absolute amplitude value scaled by trace's std.

    Parameters
    ----------
    name : str, optional, defaults to "trace_maxabs"
        Metrics name.
    """
    name = "trace_maxabs"
    is_lower_better = True
    threshold = 15

    @staticmethod
    @njit(nogil=True)
    def numba_get_mask(traces):
        """QC indicator implementation."""
        res = np.empty(traces.shape[0])
        for i in range(traces.shape[0]):
            res[i] = np.max(np.abs(traces[i])) / (traces[i].std() + 1e-10)
        return res


class MaxClipsLen(TracewiseMetric):
    """Detecting minimum and maximun clips.
    #TODO: describe how will look the resulted mask, either here or in `get_mask`.

    Parameters
    ----------
    name : str, optional, defaults to "max_clips_len"
        Metrics name.
    """
    name = "max_clips_len"
    min_value = 1
    max_value = None
    is_lower_better = True
    threshold = 3

    @property
    def description(self):
        """String description of tracewise metric"""
        return self.name + f"with {self.threshold} clips in a row"

    @staticmethod
    @njit(nogil=True)
    def numba_get_mask(traces):
        """QC indicator implementation."""
        def _update_counters(trace, i, j, value, counter, container):
            if trace == value:
                counter += 1
            else:
                if counter > 1:
                    container[i, j - counter: j] = counter
                    counter = 0
            return counter

        maxes = np.zeros_like(traces, dtype=np.int32)
        mins = np.zeros_like(traces, dtype=np.int32)
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
        return (maxes + mins).astype(np.float32)


class MaxConstLen(TracewiseMetric):
    """Detecting constant subsequences.

    #TODO: describe how will look the resulted mask, either here or in `get_mask`.
    Parameters
    ----------
    name : str, optional, defaults to "const_len"
        Metrics name.
    """
    name = "const_len"
    is_lower_better = True
    threshold = 4

    @property
    def description(self):
        """String description of tracewise metric"""
        return self.name + f"with {self.threshold} const value in a row"

    @staticmethod
    @njit(nogil=True)
    def numba_get_mask(traces): # TODO: can gather.data be 1d?
        """QC indicator implementation."""
        indicator = np.zeros_like(traces, dtype=np.float32)
        for i in range(traces.shape[0]):
            trace = traces[i]
            counter = 1
            for j in range(1, trace.shape[0]):  # pylint: disable=consider-using-enumerate
                if trace[j] == trace[j-1]: # TODO: or should we do here smth like abs(trace[j] - trace[j-1]) < 1e-10?
                    counter += 1
                else:
                    if counter > 1:
                        indicator[i, j - counter: j] = counter
                    counter = 1

            if counter > 1:
                indicator[i, -counter:] = counter
        return indicator.astype(np.float32)


class DeadTrace(TracewiseMetric):
    """Detects constant traces.

    Parameters
    ----------
    name : str, optional, defaults to "dead_trace"
        Metrics name.
    """
    name = "dead_trace"
    min_value = 0
    max_value = 1
    is_lower_better = True
    threshold = 0.5

    @staticmethod
    @njit(nogil=True)
    def numba_get_mask(traces):
        """QC indicator implementation."""
        res = np.empty(traces.shape[0], dtype=np.float32)
        for i in range(traces.shape[0]):
            res[i] = max(traces[i]) - min(traces[i]) < 1e-10
        return res


class BaseWindowMetric(TracewiseMetric):
    """Base class for all window based metric that provide method for computing sum of squares of traces amplitudes in
    provided windows defined by start and end indices, and length of windows for every trace. Also, provide a method
    `self.aggregate_headers` that is aggregating the results by passed `index_cols` or `coords_cols`."""

    def __call__(self, gather, return_rms=True):
        """Compute metric by applying `self.preprocess` and `self.get_mask` to provided gather."""
        gather = self.preprocess(gather)
        squares, nums = self.get_mask(gather)
        if return_rms:
            return self.compute_rms(squares, nums)
        return squares, nums

    @staticmethod
    def compute_rms(squares, nums):
        return np.sqrt(np.sum(squares) / np.sum(nums))

    @staticmethod
    @njit(nogil=True, parallel=True)
    def compute_stats_by_ixs(data, start_ixs_list, end_ixs_list):
        """TODO"""
        sum_squares = np.empty(data.shape[0], dtype=np.float32)
        nums = np.empty(data.shape[0], dtype=np.float32)

        for i in prange(data.shape[0]):
            trace = data[i]
            for ix in prange(len(start_ixs_list)):
                start_ix = start_ixs_list[ix][i]
                end_ix = end_ixs_list[ix][i]
                if start_ix >= 0 and end_ix >= 0:
                    sum_squares[i] = sum(trace[start_ix: end_ix] ** 2)
                    nums[i] = len(trace[start_ix: end_ix])
        return sum_squares, nums

    def construct_map(self, headers, index_cols, coords_cols, **kwargs):
        groupby_cols = self.header_cols + (coords_cols if index_cols != coords_cols else [])
        groupby = headers.groupby(index_cols)[groupby_cols]
        sums_func = {sum_name: lambda x: np.sqrt(np.sum(x)) for sum_name in self.header_cols[::2]}
        nums_func = {num_name: "sum" for num_name in self.header_cols[1::2]}
        coords_func = {coord_name: "mean" for coord_name in groupby_cols[len(self.header_cols):]}

        aggregated_gb = groupby.agg({**sums_func, **nums_func, **coords_func})
        aggregated_gb.reset_index(inplace=True)
        coords = aggregated_gb[coords_cols]
        value = self._calculate_metric_from_stats(aggregated_gb[self.header_cols].to_numpy())
        index = aggregated_gb[index_cols]
        return SurveyAttribute.construct_map(self, coords, value, index=index, **kwargs)

    def construct_map(self, coords, values, index=None, **kwargs):
        sum_sq = values.iloc[:, 0]
        n = values.iloc[:, 1]
        sum_sq_map = super().construct_map(coords, squm_sq, index=index, agg="sum")
        n_map = super().construct_map(coords, n, index=index, agg="sum")
        sum_sq_map.index_data.merge(n.index_data, on=squm_sq_map.index_cols)
        div
        sqrt

        groupby_cols = self.header_cols + (coords_cols if index_cols != coords_cols else [])
        groupby = headers.groupby(index_cols)[groupby_cols]
        sums_func = {sum_name: lambda x: np.sqrt(np.sum(x)) for sum_name in self.header_cols[::2]}
        nums_func = {num_name: "sum" for num_name in self.header_cols[1::2]}
        coords_func = {coord_name: "mean" for coord_name in groupby_cols[len(self.header_cols):]}

        aggregated_gb = groupby.agg({**sums_func, **nums_func, **coords_func})
        aggregated_gb.reset_index(inplace=True)
        coords = aggregated_gb[coords_cols]
        value = self._calculate_metric_from_stats(aggregated_gb[self.header_cols].to_numpy())
        index = aggregated_gb[index_cols]
        return SurveyAttribute.construct_map(self, coords, value, index=index, **kwargs)

    def plot(self, ax, coords, index, sort_by=None, threshold=None, top_ax_y_scale=None, bad_only=False, **kwargs):
        """Gather plot sorted by offset with tracewise indicator on a separate axis and signal and noise windows"""
        threshold = self.threshold if threshold is None else threshold
        top_ax_y_scale = self.top_ax_y_scale if top_ax_y_scale is None else top_ax_y_scale
        _ = coords
        gather = self.survey.get_gather(index)
        sort_by = "offset" if sort_by is None else sort_by
        gather = gather.sort(sort_by)
        stats = self.get_mask(gather)
        tracewise_metric = self._calculate_metric_from_stats(stats)
        tracewise_metric[tracewise_metric==0] = np.nan
        if bad_only:
            bin_mask = self.binarize(tracewise_metric, threshold)
            gather.data[self.aggregate(bin_mask) == 0] = np.nan

        gather.plot(ax=ax, top_header=tracewise_metric, **kwargs)
        top_ax = ax.figure.axes[1]
        if threshold is not None:
            self._plot_threshold(ax=top_ax, threshold=threshold)
        top_ax.set_yscale(top_ax_y_scale)
        self._plot(ax=ax, gather=gather)

    @staticmethod
    def _calculate_metric_from_stats(stats):
        raise NotImplementedError

    def _plot(self, ax, gather):
        """Add any additional metric related graphs on plot"""
        pass


class WindowRMS(BaseWindowMetric):
    """Computes traces RMS for provided window by offsets and times.

    Parameters
    ----------
    offsets : tuple of 2 ints
        Offset range to use for calcualtion.
    times : tuple of 2 ints
        Time range to use for calcualtion, measured in ms.
    name : str, optional, defaults to "rms"
        Metrics name.
    """
    name = "rms"
    is_lower_better = None # TODO: think what should it be?
    # What treshold to use? Leave it none?
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
    def header_cols(self):
        """Column names in survey.headers to srote metrics results."""
        return [self.name+"_sum", self.name+"_n"]

    @staticmethod
    @njit(nogil=True)
    def _get_time_ixs(gather_samples, times):
        # Deleete!
        times = np.asarray([max(gather_samples[0], times[0]), min(gather_samples[-1], times[1])])
        return times_to_indices(times, gather_samples).astype(np.int16)

    @staticmethod
    @njit(nogil=True)
    def _get_offsets(gather_offsets, offsets):
        # delete!
        min_offset, max_offset = min(gather_offsets), max(gather_offsets)
        return np.asarray([max(min_offset, offsets[0]), min(max_offset, offsets[1])])

    def get_mask(self, gather):
        """QC indicator implementation."""
        return self.numba_get_mask(gather.data, gather.samples, gather.offsets, self.times, self.offsets,
                                   self._get_time_ixs, self._get_offsets, self.compute_stats_by_ixs)

    @staticmethod
    @njit(nogil=True, parallel=True)
    def numba_get_mask(traces, gather_samples, gather_offests, times, offsets, _get_time_ixs, _get_offsets,
                       compute_stats_by_ixs):
        """QC indicator implementation."""
        times = _get_time_ixs(gather_samples, times)
        offsets = _get_offsets(gather_offests, offsets)

        window_ixs = np.nonzero((gather_offests >= offsets[0]) & (gather_offests <= offsets[1]))[0]
        start_ixs = np.full(len(window_ixs), fill_value=times[0])
        end_ixs = np.full(len(window_ixs), fill_value=times[1])
        result = np.full((traces.shape[0], 2), fill_value=np.nan)
        result[window_ixs] = compute_stats_by_ixs(traces[window_ixs], (start_ixs, ), (end_ixs, ))
        return result

    @staticmethod
    def _calculate_metric_from_stats(stats):
        return stats[:, 0] / stats[:, 1]

    def _plot(self, ax, gather):
        # TODO: do we want to plot this metric with sort_by != 'offset'?
        times = self._get_time_ixs(gather, self.times)
        offsets = self._get_offsets(gather, self.offsets)

        offs_ind = np.nonzero((gather.offsets >= offsets[0]) & (gather.offsets <= offsets[1]))[0]
        if len(offs_ind) > 0:
            n_rec = (offs_ind[0], times[0]), len(offs_ind), (times[1] - times[0])
            ax.add_patch(patches.Rectangle(*n_rec, linewidth=2, edgecolor='magenta', facecolor='none'))


class SignalToNoiseRMSAdaptive(BaseWindowMetric):
    """Signal to Noise RMS ratio computed in sliding windows along provided refractor velocity.
    RMS will be computed in two windows for every gather:
    1. Window shifted up from refractor velocity by `shift_up` ms. RMS in this window represents the noise value.
    2. WIndow shifted down from refractor velocity by `shift_down` ms`. RMS in this window represents the signal value.

    Only traces that contain noise and signal windows of the provided `window_size` are considered,
    the metric is 0 for other traces.


    Parameters
    ----------
    win_size : int
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

    def __init__(self, win_size, shift_up, shift_down, refractor_velocity, name=None):
        super().__init__(name=name)
        self.win_size = win_size
        self.shift_up = shift_up
        self.shift_down = shift_down
        self.refractor_velocity = refractor_velocity

    def __repr__(self):
        """String representation of the metric."""
        repr_str = f"(name='{self.name}', win_size='{self.win_size}', shift_up='{self.shift_up}', "\
                   f"shift_down='{self.shift_down}', refractor_velocity='{self.refractor_velocity}')"
        return f"{type(self).__name__}" + repr_str

    @property
    def header_cols(self):
        """Column names in survey.headers to srote metrics results."""
        return [self.name + postfix for postfix in ["_signal_sum", "_signal_n", "_noise_sum", "_noise_n"]]

    @njit(nogil=True, parallel=True)
    def _get_indices(self, win_size, shift_up, shift_down, samples, fbp_times, times_to_indices):
        """Convert times to use for noise and signal windows into indices"""

        signal_start_times = fbp_times + shift_down
        signal_end_times = np.clip(signal_start_times + win_size, None, samples[-1])

        noise_end_times = fbp_times - shift_up
        noise_start_times = np.clip(noise_end_times - win_size, 0, None)

        signal_mask = signal_start_times > samples[-1]
        noise_mask = noise_end_times < 0
        mask = signal_mask | noise_mask

        signal_start_ixs = times_to_indices(signal_start_times, samples).astype(np.int16)
        signal_end_ixs = times_to_indices(signal_end_times, samples).astype(np.int16)
        noise_start_ixs = times_to_indices(noise_start_times, samples).astype(np.int16)
        noise_end_ixs = times_to_indices(noise_end_times, samples).astype(np.int16)

        # Avoiding dividing signal rms by zero and optimize computations a little
        signal_start_ixs[mask] = -1
        signal_end_ixs[mask] = -1

        noise_start_ixs[mask] = -1
        noise_end_ixs[mask] = -1

        return signal_start_ixs, signal_end_ixs, noise_start_ixs, noise_end_ixs

    def get_mask(self, gather):
        """QC indicator implementation. See `plot` docstring for parameters descriptions."""
        fbp_times = self.refractor_velocity(gather.offsets)
        return self.numba_get_mask(gather.data, self._get_indices, self.compute_stats_by_ixs, win_size=self.win_size,
                                   shift_up=self.shift_up, shift_down=self.shift_down, samples=gather.samples,
                                   fbp_times=fbp_times, times_to_indices=times_to_indices)

    @staticmethod
    @njit(nogil=True)
    def numba_get_mask(traces, _get_indices, compute_stats_by_ixs, win_size, shift_up, shift_down, samples, fbp_times,
                       times_to_indices):
        ssi, sei, nsi, nei = _get_indices(traces, win_size, shift_up, shift_down, samples, fbp_times, times_to_indices)
        return compute_stats_by_ixs(traces, (ssi, nsi), (sei, nei))

    @staticmethod
    def _calculate_metric_from_stats(stats):
        return (stats[:, 0] / stats[:, 1] + 1e-10) / (stats[:, 2] / stats[:, 3] + 1e-10)

    def _plot(self, ax, gather):
        """Gather plot sorted by offset with tracewise indicator on a separate axis and signal and noise windows."""
        indices = self._get_indices(gather)
        indices = np.where(np.asarray(indices) == -1, np.nan, indices)

        ax.plot(np.arange(gather.n_traces), indices[0], color='lime')
        ax.plot(np.arange(gather.n_traces), indices[1], color='lime')
        ax.plot(np.arange(gather.n_traces), indices[2], color='magenta')
        ax.plot(np.arange(gather.n_traces), indices[3], color='magenta')

DEFAULT_TRACEWISE_METRICS = [TraceAbsMean, TraceMaxAbs, MaxClipsLen, MaxConstLen, DeadTrace, WindowRMS]
