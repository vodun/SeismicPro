""" Metrics for raw seismic data QC """

import warnings

import numpy as np
from scipy import signal

from mpl_toolkits.axes_grid1 import make_axes_locatable

from seismicpro.batchflow import V, B

from .pipeline_metric import PipelineMetric, pass_calc_args
from ..const import HDR_DEAD_TRACE, EPS
from .utils import fill_nulls, calc_spikes, get_val_subseq, get_const_subseq


class TracewiseMetric(PipelineMetric):
    """Base class for tracewise metrics with addidional plotters and aggregations

    Child classes should redefine `get_res` method."""

    args_to_unpack = "gather"


    min_value = None
    max_value = None
    is_lower_better = True

    views = ("plot_res", "plot_image_filter", "plot_worst_trace", "plot_wiggle")

    threshold = 10
    top_ax_y_scale='linear'


    @staticmethod
    def get_res(gather, **kwargs):
        """QC indicator implementation.
        Takes a gather as an argument and returns either a samplewise qc indicator with shape
        (`gather.n_traces`, `gather.n_samples - d`), where `d >= 0`,
        or a tracewize indicator with shape (`gather.n_traces`,).
        For a 2d output with 2-d axis smaller than `gather.n_samples`,
        it will be padded with zeros at the beggining in `filter_res`.
        """
        raise NotImplementedError

    @staticmethod
    def norm_data(gather):
        """"Gather data normalization"""
        traces = gather.data
        return (traces - np.nanmean(traces, axis=1, keepdims=True))/(np.nanstd(traces, axis=1, keepdims=True) + EPS)


    @classmethod
    def filter_res(cls, gather):
        """Return QC indicator with zero traces masked with `np.nan`
        and output shape eithet `gater.data.shape`, or (`gather.n_traces`,)."""
        res = cls.get_res(gather)

        if HDR_DEAD_TRACE in gather.headers:
            res[gather.headers[HDR_DEAD_TRACE]] = np.nan

        if res.ndim == 2 and res.shape[1] != 1:
            leading_zeros = np.zeros((gather.n_traces, gather.n_samples - res.shape[1]), dtype=res.dtype)
            res = np.concatenate((leading_zeros, res), axis=1)

        return res

    @classmethod
    def aggr(cls, gather, from_headers=None, to_headers=None, tracewise=False):
        """Return aggregated QC indicator depending on `cls.is_lower_better`

        Parameters
        ----------
        gather : SeismicGather
            input gather
        from_headers : str or None, optional
            if not None, the result is taken from the corresponding gather header, it it exists, by default None
        to_headers : str or None, optional
            if not None, the tracewise result is written into specified gather heaader, by default None
        tracewise : bool, optional
            whether to return tracewise values, or to aggregate result for the whole gather, by default False

        Returns
        -------
        np.array or float
            aggregated result for the whole gather, or an array of values for each trace
        """

        if from_headers and from_headers in gather.headers:
            return gather.headers[from_headers].values

        res = cls.filter_res(gather)

        fn = np.nanmax if cls.is_lower_better else np.nanmin

        tw_res = res if res.ndim == 1 else fn(res, axis=1)

        if to_headers:
            if isinstance(to_headers, str):
                twm_hdr = to_headers
            else:
                twm_hdr = (from_headers or '_'.join(['twm', cls.__name__, gather.survey.name]))
            gather.headers[twm_hdr] = tw_res

        if tracewise:
            return tw_res
        return fn(tw_res)

    @classmethod
    def calc(cls, gather, from_headers=None, to_headers=None): # pylint: disable=arguments-renamed
        """Return an already calculated metric."""
        return cls.aggr(gather, from_headers, to_headers, tracewise=False)

    @pass_calc_args
    def plot_res(cls, gather, ax, from_headers=None, to_headers=None, **kwargs):
        """Gather plot with tracewise indicator on a separate axis"""
        _ = to_headers
        gather.plot(ax=ax, **kwargs)
        divider = make_axes_locatable(ax)

        res = cls.aggr(gather, from_headers, tracewise=True)

        top_ax = divider.append_axes("top", sharex=ax, size="12%", pad=0.05)
        top_ax.plot(res, '.--')
        top_ax.axhline(cls.threshold, alpha=0.5)
        top_ax.xaxis.set_visible(False)
        top_ax.set_yscale(cls.top_ax_y_scale)

        set_title(top_ax, gather)

    @pass_calc_args
    def plot_wiggle(cls, gather, ax, **kwargs):
        """"Gather wiggle plot where samples with indicator above/blow `cls.threshold` are in red."""
        _ = kwargs
        fn = np.greater_equal if cls.is_lower_better else np.less_equal
        res = fn(cls.filter_res(gather), cls.threshold)
        wiggle_plot_with_filter(gather.data, res, ax, std=0.5)
        set_title(ax, gather)

    @pass_calc_args
    def plot_image_filter(cls, gather, ax, **kwargs):
        """Gather plot where samples with indicator above/blow `cls.threshold` are highlited."""
        _ = kwargs
        fn = np.greater_equal if cls.is_lower_better else np.less_equal
        res = fn(cls.filter_res(gather), cls.threshold)
        image_filter(gather.data, res, ax)
        set_title(ax, gather)

    @pass_calc_args
    def plot_worst_trace(cls, gather, ax, **kwargs):
        """Wiggle plot of the trace with the worst indicator value and 2 its neighboring traces."""
        _ = kwargs
        res = cls.aggr(gather, tracewise=True)
        plot_worst_trace(ax, gather.data, gather.headers.TraceNumber.values, res, cls.is_lower_better)
        set_title(ax, gather)



def set_title(ax, gather):
    """Set gather index as the axis title"""
    idx = np.unique(gather.headers.index.values)
    if len(idx) == 1:
        ttl = str(idx[0])
    else:
        ttl = f"[{idx[0]}...{idx[-1]}]"

    ax.set_title(ttl)

def wiggle_plot_with_filter(traces, mask, ax, std=0.1, **kwargs):
    """Wiggle plot with samples highlighted according to provided mask

    Parameters
    ----------
    traces : np.array
        array of traces
    mask : np.array of shape `arr.shape` or `(arr.shape[0], )`
        samples/traces with `mak > 0` will be highlited
    ax : matplotlib.axes.Axes
        An axis of the figure to plot on.
    std : float, optional
        scaling coefficient for traces amplitudes, by default 0.1
    """

    y_coords = np.arange(traces.shape[-1])

    if mask.ndim == 1 or mask.ndim == 2 and mask.shape[1] == 1:
        if mask.ndim == 2:
            mask = mask.squeeze(axis=1)
        mask = np.stack([mask]*traces.shape[1], axis=1)

    blurred = blure_mask(mask)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        traces = std * ((traces - np.nanmean(traces, axis=1, keepdims=True)) /
                        (np.nanstd(traces, axis=1, keepdims=True) + 1e-10))
    for i, trace in enumerate(traces):
        ax.plot(i + trace, y_coords, color='black', alpha=0.1, **kwargs)
        ax.fill_betweenx(y_coords, i, i + trace, where=(trace > 0), color='black', alpha=0.1, **kwargs)
        ax.fill_betweenx(y_coords, i - 1, i + 1, where=(blurred[i] > 0), color='red', alpha=0.1, **kwargs)
        ax.fill_betweenx(y_coords, i, i + trace, where=(mask[i] > 0), color='red', alpha=1, **kwargs)
    ax.invert_yaxis()

def blure_mask(flt, eps=EPS):
    """Blure filter values"""
    if np.any(flt == 1):
        win_size = np.floor(min((np.prod(flt.shape)/np.sum(flt == 1)), np.min(flt.shape)/10)).astype(int)

        if win_size > 4:
            kernel_1d = signal.windows.gaussian(win_size, win_size//2)
            kernel = np.outer(kernel_1d, kernel_1d)
            flt = signal.fftconvolve(flt, kernel, mode='same')
            flt[flt < eps] = 0

    return flt

def image_filter(traces, mask, ax, **kwargs):
    """Traves plot with samples highlighted according to provided mask.

    Parameters
    ----------
    traces : np.array
        array of traces
    mask : np.array of shape `traces.shape` or `(traces.shape[0], )`
        samples/traces with `mak > 0` will be highlited
    ax : matplotlib.axes.Axes
        An axis of the figure to plot on.
    kwargs : misc, optional
        additional `imshow` kwargs
    """

    if mask.ndim == 1 or mask.ndim == 2 and mask.shape[1] == 1:
        if mask.ndim == 2:
            mask = mask.squeeze(axis=1)
        mask = np.stack([mask]*traces.shape[1], axis=1)

    mask = blure_mask(mask)

    vmin, vmax = np.nanquantile(traces, q=[0.05, 0.95])
    kwargs = {"cmap": "gray", "aspect": "auto", "vmin": vmin, "vmax": vmax, **kwargs}
    ax.imshow(traces.T, **kwargs)
    ax.imshow(mask.T, alpha=0.5, cmap='Reds', aspect='auto')

def plot_worst_trace(ax, traces, trace_numbers, indicators, max_is_worse, std=0.5, **kwargs):
    """Wiggle plot of the trace with the worst indicator value
    and 2 its neighboring traces (with respect to TraceNumbers).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        An axis of the figure to plot on.
    traces : np.array
        Array of traces.
    trace_numbers : np.array
        TraceNumbers of provided traces.
    indicators : np.array of shape `traces.shape` or `(traces.shape[0], )`
        Indicators to select the worst trace.
    max_is_worse : bool
        Specifies what type of extemum corresponds to the worst trace.
    std : float, optional
        Scaling coefficient for traces amplitudes, by default 0.5
    """

    _, n_samples = traces.shape

    fn = np.argmax if max_is_worse else np.argmin
    worst_tr_idx = fn(indicators)

    if worst_tr_idx == 0:
        indices, colors = (0, 1, 2), ('red', 'black', 'black')
    elif worst_tr_idx == n_samples - 1:
        indices, colors = (n_samples - 3, n_samples - 2, n_samples - 1), ('black', 'black', 'red')
    else:
        indices, colors = (worst_tr_idx - 1, worst_tr_idx, worst_tr_idx + 1), ('black', 'red', 'black')

    traces = traces[indices,]
    tns = trace_numbers[indices,]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        traces = std * ((traces - np.nanmean(traces, axis=None, keepdims=True)) /
                        (np.nanstd(traces, axis=None, keepdims=True) + 1e-10))

    y_coords = np.arange(n_samples)
    for i, (trace, col) in enumerate(zip(traces, colors)):
        ax.plot(i + trace, y_coords, color=col, **kwargs)
        ax.fill_betweenx(y_coords, i, i + trace, where=(trace > 0), color=col, **kwargs)
        ax.text(i, 0, f"{tns[i]}", color=col)

    ax.invert_yaxis()


class SpikesMetric(TracewiseMetric):
    """Spikes detection."""
    name = "spikes"
    min_value = 0
    max_value = None
    is_lower_better = True

    threshold=2

    @staticmethod
    def get_res(gather):
        """QC indicator implementation."""
        traces = gather.data.copy()
        fill_nulls(traces)

        res = calc_spikes(traces)
        return np.pad(res, ((0,0), (1, 1)))


class AutocorrMetric(TracewiseMetric):
    """Autocorrelation with shift 1"""
    name = "autocorr"
    is_lower_better = False
    threshold = 0.9

    @staticmethod
    def get_res(gather):
        """QC indicator implementation."""
        return (np.nansum(gather.data[...,1:] * gather.data[..., :-1], axis=1) /
                (gather.n_samples - np.isnan(gather.data).sum(axis=1) + EPS))


class TraceMeanAbs(TracewiseMetric):
    """Absolute value of the traces mean."""
    name = "trace_meanabs"
    is_lower_better = True
    threshold = 0.1
    top_ax_y_scale = 'log'


    @staticmethod
    def get_res(gather):
        """QC indicator implementation."""
        return np.abs(gather.data.mean(axis=1) / (gather.data.std(axis=1) + EPS))

class MaxClipsLenMetric(TracewiseMetric):
    """Detecting minimum/maximun clips"""
    name = "max_clips_len"
    min_value = 1
    max_value = None
    is_lower_better = True

    threshold = 3

    @staticmethod
    def get_res(gather):
        """QC indicator implementation."""
        traces = gather.data

        maxes = traces.max(axis=-1, keepdims=True)
        mins = traces.min(axis=-1, keepdims=True)

        res_plus = get_val_subseq(traces, maxes)
        res_minus = get_val_subseq(traces, mins)

        return (res_plus + res_minus).astype(np.float32)

class ConstLenMetric(TracewiseMetric):
    """Detecting constant subsequences"""
    name = "const_len"
    is_lower_better = True
    threshold = 4

    @staticmethod
    def get_res(gather):
        """QC indicator implementation."""
        res = get_const_subseq(gather.data)
        return res.astype(np.float32)

class StdFraqMetricGlob(TracewiseMetric):
    """Traces std relative to survey's std, log10 scale"""
    name = "std_fraq_glob"
    min_value = None
    max_value = None
    is_lower_better = False
    threshold = -2

    @staticmethod
    def get_res(gather):
        """QC indicator implementation."""
        res = np.log10(gather.data.std(axis=1) / gather.survey.std)
        return res

def add_metric(ppl, metric_cls, src, **kwargs):
    """Add PipelineMetrics to a pipeline, and write corresponding tracewise aggregations to a pipeline variable."""
    acc_name = '_'.join(['mmap', metric_cls.__name__, src])
    twm_name =  '_'.join(['twm', metric_cls.__name__, src])
    ppl = (ppl
           .init_variable(acc_name)
           .calculate_metric(metric_cls, gather=src, save_to=V(acc_name, mode="a"), to_headers=twm_name, **kwargs)
           ########## tracewise metics to ppl variable #####################
           .init_variable(twm_name, [])
           .apply_parallel(metric_cls.aggr, src=src, dst=twm_name, from_headers=twm_name, tracewise=True)
           .update(V(twm_name, mode='e'), B(twm_name))
    )

    return ppl
