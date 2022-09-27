""" Metrics for raw seismic data QC """

import warnings

import numpy as np
from scipy import signal

from matplotlib import patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .metrics import Metric
from ..const import HDR_DEAD_TRACE, EPS, HDR_FIRST_BREAK
from .utils import fill_leading_nulls, get_val_subseq, get_const_subseq, rms_2_windows_ratio, mute_and_norm
from ..gather.utils import times_to_indices


class TracewiseMetric(Metric):
    """Base class for tracewise metrics with addidional plotters and aggregations
    Child classes should redefine `_get_res` method, and optionnaly `preprocess`.
    """

    min_value = None
    max_value = None
    is_lower_better = True

    views = ("plot_res", "plot_image_filter", "plot_worst_trace", "plot_wiggle")

    threshold = None
    top_ax_y_scale='linear'


    def __init__(self, survey, coords_cols, **kwargs):
        super().__init__(**kwargs)
        self.survey = survey.reindex(coords_cols)

    @staticmethod
    def _get_res(gather, **kwargs):
        """QC indicator implementation.
        Takes a gather as an argument and returns either a samplewise qc indicator with shape
        (`gather.n_traces`, `gather.n_samples - d`), where `d >= 0`,
        or a tracewize indicator with shape (`gather.n_traces`,).
        For a 2d output with 2-d axis smaller than `gather.n_samples`,
        it will be padded with zeros at the beggining in `get_res`.
        """
        raise NotImplementedError

    @classmethod
    def preprocess(cls, gather, **kwargs):
        """Preprocess gather for calculating metric. Identity by default."""
        _ = kwargs
        return gather

    @classmethod
    def get_res(cls, gather, from_headers=True, **kwargs):
        """Return QC indicator with zero traces masked with `np.nan`
        and output shape either `gater.data.shape`, or (`gather.n_traces`,)."""

        if from_headers and cls.__name__ in gather.headers:
            return np.stack(gather.headers[cls.__name__].values)

        gather = cls.preprocess(gather, **kwargs)

        res = cls._get_res(gather, **kwargs)

        if HDR_DEAD_TRACE in gather.headers:
            res[gather.headers[HDR_DEAD_TRACE]] = np.nan

        if res.ndim == 2 and res.shape[1] != 1:
            leading_zeros = np.zeros((gather.n_traces, gather.n_samples - res.shape[1]), dtype=res.dtype)
            res = np.concatenate((leading_zeros, res), axis=1)

        return res

    @classmethod
    def aggr(cls, res, tracewise=False):
        """Aggregte input depending on `cls.is_lower_better`

        Parameters
        ----------
        res : np.array
            input 1d or 2d array
        tracewise : bool, optional
            whether to return tracewise values, or to aggregate result for the whole gather, by default False

        Returns
        -------
        np.array or float
            aggregated result for the whole gather, or an array of values for each trace
        """
        fn = np.nanmax if cls.is_lower_better else np.nanmin

        tw_res = res if res.ndim == 1 else fn(res, axis=1)

        if tracewise:
            return tw_res
        return fn(tw_res)

    @classmethod
    def calc(cls, gather, from_headers=False, tracewise=False, **kwargs): # pylint: disable=arguments-renamed
        """Return an already calculated metric."""
        res = cls.get_res(gather, from_headers=from_headers, **kwargs)
        return cls.aggr(res, tracewise=tracewise)

    def plot_res(self, coords, ax, **kwargs):
        """Gather plot with tracewise indicator on a separate axis"""
        gather = self.survey.get_gather(coords)

        res = self.calc(gather, tracewise=True, **kwargs)

        gather = self.preprocess(gather, **kwargs)
        gather.plot(ax=ax)
        divider = make_axes_locatable(ax)

        top_ax = divider.append_axes("top", sharex=ax, size="12%", pad=0.05)
        top_ax.plot(res, '.--')
        if self.threshold is not None:
            top_ax.axhline(self.threshold, alpha=0.5)
        top_ax.xaxis.set_visible(False)
        top_ax.set_yscale(self.top_ax_y_scale)

        set_title(top_ax, gather)

    def _plot_filter(self, mode, coords, ax, **kwargs):
        """Gather plot with filter"""
        gather = self.survey.get_gather(coords)

        fn = np.greater_equal if self.is_lower_better else np.less_equal

        if self.threshold is not None:
            res = fn(self.get_res(gather, from_headers=False, **kwargs), self.threshold)
        else:
            res = np.zeros_like(gather.data)

        gather = self.preprocess(gather, **kwargs)

        if mode == 'wiggle':
            wiggle_plot_with_filter(gather.data, res, ax, std=0.5)
        else:
            image_filter(gather.data, res, ax)

        set_title(ax, gather)

    def plot_wiggle(self, coords, ax, **kwargs):
        """"Gather wiggle plot where samples with indicator above/blow `cls.threshold` are in red."""
        self._plot_filter('wiggle', coords, ax, **kwargs)

    def plot_image_filter(self, coords, ax, **kwargs):
        """Gather plot where samples with indicator above/blow `cls.threshold` are highlited."""
        self._plot_filter('image', coords, ax, **kwargs)

    def plot_worst_trace(self, coords, ax, **kwargs):
        """Wiggle plot of the trace with the worst indicator value and 2 its neighboring traces."""
        gather = self.survey.get_gather(coords)
        res = self.calc(gather, tracewise=True, **kwargs)

        gather = self.preprocess(gather, **kwargs)
        plot_worst_trace(ax, gather.data, gather.headers.TraceNumber.values, res, self.is_lower_better)
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

    fn = np.nanargmax if max_is_worse else np.nanargmin
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

    @classmethod
    def preprocess(cls, gather, muter_col=HDR_FIRST_BREAK, **kwargs):
        _ = kwargs
        return mute_and_norm(gather, muter_col)

    @staticmethod
    def _get_res(gather, **kwargs):
        """QC indicator implementation."""
        _ = kwargs
        traces = gather.data
        fill_leading_nulls(traces)

        running_mean = (traces[:, 1:-1] + traces[:, 2:] + traces[:, :-2])/3
        res = np.abs(traces[:, 1:-1] - running_mean)

        return np.pad(res, ((0,0), (1, 1)))

class AutocorrMetric(TracewiseMetric):
    """Autocorrelation with shift 1"""
    name = "autocorr"
    is_lower_better = False
    threshold = 0.9

    @classmethod
    def preprocess(cls, gather, muter_col=HDR_FIRST_BREAK, **kwargs):
        _ = kwargs
        return mute_and_norm(gather, muter_col)

    @staticmethod
    def _get_res(gather, **kwargs):
        """QC indicator implementation."""
        _ = kwargs
        return (np.nansum(gather.data[...,1:] * gather.data[..., :-1], axis=1) /
                (gather.n_samples - np.isnan(gather.data).sum(axis=1) + EPS))

class TraceAbsMean(TracewiseMetric):
    """Absolute value of the traces mean."""
    name = "trace_absmean"
    is_lower_better = True
    threshold = 0.1
    top_ax_y_scale = 'log'

    @staticmethod
    def _get_res(gather, **kwargs):
        """QC indicator implementation."""
        _ = kwargs
        return np.abs(gather.data.mean(axis=1) / (gather.data.std(axis=1) + EPS))

class TraceMaxAbs(TracewiseMetric):
    """Maximun absolute amplitude value."""
    name = "trace_maxabs"
    is_lower_better = True
    threshold = None
    top_ax_y_scale = 'log'

    @staticmethod
    def _get_res(gather, **kwargs):
        """QC indicator implementation."""
        _ = kwargs
        return np.max(np.abs(gather.data), axis=1) / (gather.data.std(axis=1) + EPS)

class MaxClipsLenMetric(TracewiseMetric):
    """Detecting minimum/maximun clips"""
    name = "max_clips_len"
    min_value = 1
    max_value = None
    is_lower_better = True

    threshold = 3

    @staticmethod
    def _get_res(gather, **kwargs):
        """QC indicator implementation."""
        _ = kwargs
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
    def _get_res(gather, **kwargs):
        """QC indicator implementation."""
        _ = kwargs
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
    def _get_res(gather, **kwargs):
        """QC indicator implementation."""
        _ = kwargs
        res = np.log10(gather.data.std(axis=1) / gather.survey.std)
        return res

class TraceSinalToNoiseRMSRatio(TracewiseMetric):
    """ Signal to Noise RMS ratio computed using provided windows """
    name = "trace_RMS_Ratio"
    is_lower_better = False
    threshold = None
    top_ax_y_scale = 'log'
    views = 'plot'

    @staticmethod
    def _get_indices(gather, win_size, n_start, s_start, mask, first_breaks_col):
        """Convert times to use for noise and signal windows into indices"""
        if first_breaks_col in gather.headers:
            fb_high = gather.headers[first_breaks_col][mask].min()
            fb_low = gather.headers[first_breaks_col][mask].max()
        else:
            fb_high = gather.samples[-1]
            fb_low = gather.samples[0]

        n_beg, n_end = times_to_indices(np.asarray([max(gather.samples[0], n_start), min(n_start + win_size, fb_high)]),
                                        gather.samples).astype(int)

        s_beg, s_end = times_to_indices(np.asarray([max(fb_low, s_start), min(s_start + win_size, gather.samples[-1])]),
                                        gather.samples).astype(int)

        n_begs = np.full(gather.n_traces, fill_value=n_beg, dtype=int)
        s_begs = np.full(gather.n_traces, fill_value=s_beg, dtype=int)
        win_size = min(n_end - n_beg, s_end - s_beg)

        return n_begs, s_begs, win_size

    @staticmethod
    def _get_res(gather, offsets, win_size, n_start, s_start, first_breaks_col=HDR_FIRST_BREAK, **kwargs):
        """QC indicator implementation."""
        _ = kwargs

        mask = ((gather.offsets >= offsets[0]) & (gather.offsets <= offsets[1]))

        n_begs, s_begs, win_size = TraceSinalToNoiseRMSRatio._get_indices(gather, win_size, n_start, s_start,
                                                                          mask, first_breaks_col)

        s_begs[~mask] = -1
        n_begs[~mask] = -1

        res = rms_2_windows_ratio(gather.data, n_begs, s_begs, win_size)

        return res

    def plot(self, coords, ax, offsets, win_size, n_start, s_start, first_breaks_col=HDR_FIRST_BREAK, **kwargs):
        """Gather plot sorted by offset with tracewise indicator on a separate axis and signal and noise windows"""
        gather = self.survey.get_gather(coords)

        res = self.calc(gather, from_headers=False, tracewise=True, offsets=offsets, win_size=win_size,
                        n_start=n_start, s_start=s_start, first_breaks_col=first_breaks_col, **kwargs)

        gather = self.preprocess(gather, **kwargs)
        order = np.argsort(gather.offsets.ravel(), kind='stable')

        gather.sort(by='offset').plot(ax=ax)

        divider = make_axes_locatable(ax)
        top_ax = divider.append_axes("top", sharex=ax, size="12%", pad=0.05)
        top_ax.xaxis.set_visible(False)
        top_ax.set_yscale(self.top_ax_y_scale)

        top_ax.plot(res[order], '.--')
        if self.threshold is not None:
            top_ax.axhline(self.threshold, alpha=0.5)

        mask = (gather.offsets >= offsets[0]) & (gather.offsets <= offsets[1])
        offs_ind = np.nonzero(mask)[0]

        n_begs, s_begs, win_size = self._get_indices(gather, win_size, n_start, s_start, mask, first_breaks_col)

        n_rec = (offs_ind[0], n_begs[0]), len(offs_ind), win_size
        ax.add_patch(patches.Rectangle(*n_rec, linewidth=1, edgecolor='magenta', facecolor='none'))
        s_rec = (offs_ind[0], s_begs[0]), len(offs_ind), win_size
        ax.add_patch(patches.Rectangle(*s_rec, linewidth=1, edgecolor='lime',facecolor='none'))

        set_title(top_ax, gather)

class TraceSinalToNoiseRMSRatioAdaptive(TracewiseMetric):
    """ Signal to Noise RMS ratio computed using provided windows """
    name = "trace_RMS_Ratio_Adaptive"
    is_lower_better = False
    threshold = None
    top_ax_y_scale = 'log'
    views = 'plot'

    @staticmethod
    def _get_indices(gather,  win_size, shift_up, shift_down, first_breaks_col):
        """Convert times to use for noise and signal windows into indices"""
        if first_breaks_col not in gather.headers:
            raise RuntimeError(f"{first_breaks_col} not in headers")

        fbp = gather.headers[first_breaks_col].values

        noise_beg = fbp - shift_up - win_size
        noise_beg[noise_beg < 0] = np.nan

        signal_beg = fbp + shift_down
        signal_beg[signal_beg > gather.samples[-1] - win_size] = np.nan

        s_begs = times_to_indices(signal_beg, gather.samples)
        n_begs = times_to_indices(noise_beg, gather.samples)

        return n_begs, s_begs

    @staticmethod
    def _get_res(gather, win_size, shift_up, shift_down, first_breaks_col=HDR_FIRST_BREAK, **kwargs):
        """QC indicator implementation."""
        _ = kwargs

        n_begs, s_begs = TraceSinalToNoiseRMSRatioAdaptive._get_indices(gather, win_size,
                                                                        shift_up, shift_down, first_breaks_col)

        s_begs[np.isnan(s_begs)] = -1
        n_begs[np.isnan(n_begs)] = -1

        win_size = np.rint(win_size/gather.sample_rate).astype(int)

        res = rms_2_windows_ratio(gather.data, n_begs.astype(int), s_begs.astype(int), win_size)

        return res

    def plot(self, coords, ax,  win_size, shift_up, shift_down, first_breaks_col=HDR_FIRST_BREAK, **kwargs):
        """Gather plot sorted by offset with tracewise indicator on a separate axis and signal and noise windows"""
        gather = self.survey.get_gather(coords)

        res = self.calc(gather, from_headers=False, tracewise=True, win_size=win_size,
                        shift_up=shift_up, shift_down=shift_down, first_breaks_col=first_breaks_col, **kwargs)

        gather = self.preprocess(gather, **kwargs)
        order = np.argsort(gather.offsets.ravel(), kind='stable')

        gather.sort(by='offset').plot(ax=ax)

        divider = make_axes_locatable(ax)
        top_ax = divider.append_axes("top", sharex=ax, size="12%", pad=0.05)
        top_ax.xaxis.set_visible(False)
        top_ax.set_yscale(self.top_ax_y_scale)

        top_ax.plot(res[order], '.--')
        if self.threshold is not None:
            top_ax.axhline(self.threshold, alpha=0.5)

        n_begs, s_begs = self._get_indices(gather, win_size, shift_up, shift_down, first_breaks_col)

        n_begs[np.isnan(s_begs)] = np.nan
        s_begs[np.isnan(n_begs)] = np.nan

        win_size = np.rint(win_size/gather.sample_rate)

        ax.plot(np.arange(gather.n_traces), n_begs, color='magenta')
        ax.plot(np.arange(gather.n_traces), n_begs+win_size, color='magenta')
        ax.plot(np.arange(gather.n_traces), s_begs, color='lime')
        ax.plot(np.arange(gather.n_traces), s_begs+win_size, color='lime')

        set_title(top_ax, gather)
