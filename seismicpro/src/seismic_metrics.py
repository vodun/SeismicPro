"""File contains metircs for seismic processing."""
import numpy as np
from numba import njit, prange

from .plot_utils import plot_metrics_map

from ..batchflow.models.metrics import Metrics


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

class MetricsMap(Metrics):
    """seismic metrics class"""

    def __init__(self, metrics, coords, *args, **kwargs):
        _ = args, kwargs
        super().__init__()

        self.metrics = metrics
        self.coords = coords

        if len(self.metrics) != len(self.coords):
            raise ValueError('length of given metrics do not match with length of coords.')

        self.maps_list = [[*coord, metric] for coord, metric in zip(self.coords, self.metrics)]

        self._agg_fn_dict = {'mean': np.nanmean,
                             'max': np.nanmax,
                             'min': np.nanmin}

    def extend(self, metrics):
        """Extend coordinates and metrics to global container."""
        self.maps_list.extend(metrics.maps_list)

    def construct_map(self, bin_size=500, vmin=None, vmax=None, cm=None, title=None, figsize=None, #pylint: disable=too-many-arguments
                      save_to=None, pad=False, plot=True, agg_bins_fn='mean', agg_bins_kwargs=None):
        """ Each value in resulted map represent aggregated value of metrics for coordinates belongs to current bin.
        """

        if isinstance(bin_size, int):
            bin_size = (bin_size, bin_size)

        maps_list_transposed = np.array(self.maps_list).T

        coords_x = np.array(maps_list_transposed[0], dtype=np.float32)
        coords_y = np.array(maps_list_transposed[1], dtype=np.float32)
        metrics = np.array(list(maps_list_transposed[2]))
        if isinstance(metrics[0], (tuple, list, set, np.ndarray)):
            if len(metrics[0].shape) > 1:
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
        elif isinstance(agg_bins_fn, callable):
            raise ValueError('agg_bins_fn should be whether str or callable, not {}'.format(type(call_agg_bins)))

        metric_map,shapes = self.construct_metrics_map(coords_x=coords_x, coords_y=coords_y,
                                                metrics=metrics, bin_size=bin_size,
                                                agg_bins_fn=call_agg_bins)
        extent_coords = [coords_x.min(), coords_x.max(), coords_y.min(), coords_y.max()]
        if plot:
            plot_metrics_map(metrics_map=metric_map, vmin=vmin, vmax=vmax, extent_coords=extent_coords, cm=cm,
                             title=title, figsize=figsize, save_to=save_to, pad=pad)
        return metric_map, shapes

    @staticmethod
    @njit(parallel=True)
    def construct_metrics_map(coords_x, coords_y, metrics, bin_size, agg_bins_fn):
        """njit map"""
        shapes = []
        bin_size_x, bin_size_y = bin_size
        range_x = np.arange(coords_x.min(), coords_x.max() + 1, bin_size_x)
        range_y = np.arange(coords_y.min(), coords_y.max() + 1, bin_size_y)
        metrics_map = np.full((len(range_y), len(range_x)), np.nan)
        for i in prange(len(range_x)): #pylint: disable=not-an-iterable
            for j in prange(len(range_y)): #pylint: disable=not-an-iterable
                mask = ((coords_x - range_x[i] >= 0) & (coords_x - range_x[i] < bin_size_x) &
                        (coords_y - range_y[j] >= 0) & (coords_y - range_y[j] < bin_size_y))
                if mask.sum() > 0:
                    notnan = np.ravel(metrics[mask])
                    notnan = np.sum(~np.isnan(notnan))
                    shapes.append([i, j, notnan])
                    metrics_map[j, i] = agg_bins_fn(np.ravel(metrics[mask]))
        return metrics_map, shapes
