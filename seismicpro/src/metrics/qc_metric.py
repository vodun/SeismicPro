
import warnings

import numpy as np
from scipy import signal

from mpl_toolkits.axes_grid1 import make_axes_locatable


from .pipeline_metric import PipelineMetric, pass_calc_args
from ..const import HDR_DEAD_TRACE

EPS = 1e-10

class TracewiseMetric(PipelineMetric):

    views = ("plot_res", "plot_image_filter", "plot_worst_trace", "plot_wiggle")

    threshold = 10
    top_ax_y_scale='linear'


    @staticmethod
    def get_res(gather, **kwargs):
        raise NotImplementedError

    @classmethod
    def filter_res(cls, gather):
        res = cls.get_res(gather)

        if HDR_DEAD_TRACE in gather.headers:
            res[gather.headers[HDR_DEAD_TRACE]] = None

        if res.ndim == 2 and res.shape[1] != 1:
            leading_zeros = np.zeros((gather.n_traces, gather.n_samples - res.shape[1]), dtype=int)
            res = np.concatenate((leading_zeros, res.astype(int)), axis=1)

        return res

    @classmethod
    def aggr(cls, res, tracewise=False):
        fn = np.nanmax if cls.is_lower_better else np.nanmin
        if tracewise:
            return res if res.ndim == 1 else fn(res, axis=1)
        return fn(res)

    @classmethod
    def calc(cls, gather):
        return cls.aggr(cls.filter_res(gather))

    @pass_calc_args
    def plot_res(cls, gather, ax, **kwargs):
        gather.plot(ax=ax, **kwargs)
        divider = make_axes_locatable(ax)

        res = cls.aggr(cls.filter_res(gather), tracewise=True)

        top_ax = divider.append_axes("top", sharex=ax, size="12%", pad=0.05)
        top_ax.plot(res, '.--')
        top_ax.axhline(cls.threshold, alpha=0.5)
        top_ax.xaxis.set_visible(False)
        top_ax.set_yscale(cls.top_ax_y_scale)

        set_title(top_ax, gather)

    @pass_calc_args
    def plot_wiggle(cls, gather, ax, **kwargs):
        fn = np.greater_equal if cls.is_lower_better else np.less_equal
        res = fn(cls.filter_res(gather), cls.threshold)
        wiggle_plot_with_filter(gather.data, res, ax, std=0.5)
        set_title(ax, gather)

    @pass_calc_args
    def plot_image_filter(cls, gather, ax, **kwargs):
        fn = np.greater_equal if cls.is_lower_better else np.less_equal
        res = fn(cls.filter_res(gather), cls.threshold)
        image_filter(gather.data, res, ax)
        set_title(ax, gather)

    @pass_calc_args
    def plot_worst_trace(cls, gather, ax, **kwargs):
        res = cls.filter_res(gather)
        plot_worst_trace(ax, gather.data, gather.headers.TraceNumber.values, res, is_lower_better=cls.is_lower_better)
        set_title(ax, gather)



def set_title(ax, gather):
    idx = np.unique(gather.headers.index.values)
    if len(idx) == 1:
        ttl = str(idx[0])
    else:
        ttl = f"[{idx[0]}...{idx[-1]}]"

    ax.set_title(ttl)

def wiggle_plot_with_filter(arr, flt, ax, flt_color='red', std=0.1, **kwargs):

    n_traces, n_samples = arr.shape

    if not isinstance(flt_color, (tuple, list, np.ndarray)):
        flt_color = [flt_color]* n_traces
    elif len(flt_color) != n_traces:
        raise AttributeError("incorrect color length")

    y_coords = np.arange(n_samples)

    if flt.ndim == 1 or flt.ndim == 2 and flt.shape[1] == 1:
        if flt.ndim == 2:
            flt = flt.squeeze(axis=1)
        flt = np.stack([flt]*arr.shape[1], axis=1)

    blurred = make_mask(flt)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        traces = std * ((arr - np.nanmean(arr, axis=1, keepdims=True)) /
                        (np.nanstd(arr, axis=1, keepdims=True) + 1e-10))
    for i, (trace, col) in enumerate(zip(traces, flt_color)):
        ax.plot(i + trace, y_coords, color='black', alpha=0.1, **kwargs)
        ax.fill_betweenx(y_coords, i, i + trace, where=(trace > 0), color='black', alpha=0.1, **kwargs)
        ax.fill_betweenx(y_coords, i - 1, i + 1, where=(blurred[i] > 0), color=col, alpha=0.1, **kwargs)
        ax.fill_betweenx(y_coords, i, i + trace, where=(flt[i] > 0), color=col, alpha=1, **kwargs)
    ax.invert_yaxis()

def make_mask(flt, eps=EPS):
    if np.any(flt == 1):
        win_size = np.floor(min((np.prod(flt.shape)/np.sum(flt == 1)), np.min(flt.shape)/10)).astype(int)

        if win_size > 4:
            kernel = np.outer(signal.windows.gaussian(win_size, win_size//2), signal.windows.gaussian(win_size, win_size//2))
            flt = signal.fftconvolve(flt, kernel, mode='same')
            flt[flt < eps] = 0

    return flt

def image_filter(arr, flt, ax, **kwargs):


    if flt.ndim == 1 or flt.ndim == 2 and flt.shape[1] == 1:
        if flt.ndim == 2:
            flt = flt.squeeze(axis=1)
        flt = np.stack([flt]*arr.shape[1], axis=1)

    flt = make_mask(flt)

    vmin, vmax = np.quantile(arr, q=[0.05, 0.95])
    kwargs = {"cmap": "gray", "aspect": "auto", "vmin": vmin, "vmax": vmax, **kwargs}
    ax.imshow(arr.T, **kwargs)
    ax.imshow(flt.T, alpha=0.5, cmap='Reds', aspect='auto')

def plot_worst_trace(ax, arr, tns, flt, std=0.5, is_lower_better=True, **kwargs):

    _, n_samples = arr.shape

    func, idx_func = (np.max, np.argmax) if is_lower_better else (np.min, np.argmin)

    if flt.ndim == 2:
        flt = func(flt, axis=1)

    worst_tr_idx = idx_func(flt, axis=0)
    if worst_tr_idx == 0:
        indices, colors = (0, 1, 2), ('red', 'black', 'black')
    elif worst_tr_idx == n_samples - 1:
        indices, colors = (n_samples - 3, n_samples - 2, n_samples - 1), ('black', 'black', 'red')
    else:
        indices, colors = (worst_tr_idx - 1, worst_tr_idx, worst_tr_idx + 1), ('black', 'red', 'black')

    traces = arr[indices,]
    tns_ = tns[indices,]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        traces = std * ((traces - np.nanmean(traces, axis=None, keepdims=True)) /
                        (np.nanstd(traces, axis=None, keepdims=True) + 1e-10))

    y_coords = np.arange(n_samples)
    for i, (trace, col) in enumerate(zip(traces, colors)):
        ax.plot(i + trace, y_coords, color=col, **kwargs)
        ax.fill_betweenx(y_coords, i, i + trace, where=(trace > 0), color=col, **kwargs)
        ax.text(i, 0, f"{tns_[i]}", color=col)

    ax.invert_yaxis()

