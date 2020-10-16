"""File contains metircs for seismic processing."""
import numpy as np
from numba import njit, prange

from .plot_utils import plot_metrics_map

from seismicpro.batchflow.models.metrics import Metrics


class MetricsMap(Metrics):
    """seismic metrics class"""

    def __init__(self, metrics, coords, *args, **kwargs):
        _ = args, kwargs
        super().__init__()

        self.metrics_values = list(metrics) if isinstance(metrics, (list, tuple, set, np.ndarray)) else [metrics]
        self.coords = list(coords)

        if len(self.metrics_values) != len(self.coords):
            raise ValueError('length of given metrics do not match with length of coords.')

        self._agg_fn_dict = {'mean': np.nanmean,
                             'max': np.nanmax,
                             'min': np.nanmin}

    def extend(self, metrics):
        """Extend coordinates and metrics to global container."""
        self.metrics_values.extend(metrics.metrics_values)
        self.coords.extend(metrics.coords)

    def construct_map(self, bin_size=500, cm=None, title=None, figsize=None, save_to=None, dpi=None, #pylint: disable=too-many-arguments
                      pad=False, plot=True, agg_bins_fn='mean', agg_bins_kwargs=None, **plot_kwargs):
        """ Each value in resulted map represent aggregated value of metrics for coordinates belongs to current bin.
        """

        if isinstance(bin_size, int):
            bin_size = (bin_size, bin_size)

        coords = np.array(self.coords)
        coords_x = np.array(coords[:, 0], dtype=np.float32)
        coords_y = np.array(coords[:, 1], dtype=np.float32)

        metrics = np.array(list(self.metrics_values))

        if isinstance(metrics[0], (tuple, list, set, np.ndarray)):
            if len(np.array(metrics[0]).shape) > 1:
                raise ValueError('Construct map does not work with 3d metrics yet.')
            metrics_shapes = np.array([metric.shape for metric in metrics])
            metrics_storage = np.empty((len(metrics), np.max(metrics_shapes)))
            metrics_storage.fill(np.nan)
            mask = np.zeros_like(metrics_storage) + np.arange(np.max(metrics_shapes))
            mask = mask < metrics_shapes
            metrics_storage[mask] = np.concatenate(metrics)
            metrics = metrics_storage.copy()
        metrics = np.array(metrics, dtype=np.float32)

        if isinstance(agg_bins_fn, str):
            call_agg_bins = getattr(NumbaNumpy, agg_bins_fn)
            call_agg_bins = call_agg_bins(None, **agg_bins_kwargs) if agg_bins_kwargs is not None else call_agg_bins
        elif callable(agg_bins_fn):
            call_agg_bins = agg_bins_fn
        else:
            raise ValueError('agg_bins_fn should be whether str or callable, not {}'.format(type(agg_bins_fn)))

        metric_map = self.construct_metrics_map(coords_x=coords_x, coords_y=coords_y,
                                                metrics=metrics, bin_size=bin_size,
                                                agg_bins_fn=call_agg_bins)

        if plot:
            extent = [coords_x.min(), coords_x.max(), coords_y.min(), coords_y.max()]
            plot_metrics_map(metrics_map=metric_map, extent=extent, cm=cm, title=title,
                             figsize=figsize, save_to=save_to, dpi=dpi, pad=pad,
                             **plot_kwargs)
        return metric_map

    @staticmethod
    @njit(parallel=True)
    def construct_metrics_map(coords_x, coords_y, metrics, bin_size, agg_bins_fn):
        """njit map"""
        bin_size_x, bin_size_y = bin_size
        range_x = np.arange(coords_x.min(), coords_x.max() + 1, bin_size_x)
        range_y = np.arange(coords_y.min(), coords_y.max() + 1, bin_size_y)
        metrics_map = np.full((len(range_y), len(range_x)), np.nan)
        for i in prange(len(range_x)): #pylint: disable=not-an-iterable
            for j in prange(len(range_y)): #pylint: disable=not-an-iterable
                mask = ((coords_x - range_x[i] >= 0) & (coords_x - range_x[i] < bin_size_x) &
                        (coords_y - range_y[j] >= 0) & (coords_y - range_y[j] < bin_size_y))
                if mask.sum() > 0:
                    metrics_map[j, i] = agg_bins_fn(np.ravel(metrics[mask]))
        return metrics_map


def quantile(array, q):
    """ numba quantile. """
    _ = array
    return  njit(lambda array: np.quantile(array, q=q))

def absquantile(array, q):
    """ numba absquantile. """
    _ = array
    return njit(lambda array: _absquantile(array, q=q))

@njit
def _absquantile(array, q):
    shifted_array = array - np.mean(array)
    val = np.quantile(np.abs(shifted_array), q)
    ind = np.argmin(np.abs(np.abs(shifted_array) - val))
    return array[ind]

class NumbaNumpy:
    """ Holder for jit-accelerated functions. """
    #pylint: disable = unnecessary-lambda, undefined-variable
    nanmin = njit(lambda array: np.nanmin(array))
    nanmax = njit(lambda array: np.nanmax(array))
    nanmean = njit(lambda array: np.nanmean(array))
    nanstd = njit(lambda array: np.nanstd(array))

    min = nanmin
    max = nanmax
    mean = nanmean
    std = nanstd
    quantile = quantile
    absquantile = absquantile
