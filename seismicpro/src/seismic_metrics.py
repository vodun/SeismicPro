"""File contains metircs for seismic processing."""
import numpy as np
from numba import njit, prange

from .plot_utils import plot_metrics_map

from ..batchflow import inbatch_parallel
from ..batchflow.models.metrics import Metrics


def quantile(array, q):
    return  njit(lambda array: np.quantile(array, q=q))

def absquantile(array, q):
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

    AVALIBALE_METRICS = [
        'construct_map'
    ]

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

    def _split_result(self):
        """split_result"""
        coords_x, coords_y, metrics = np.array(self.maps_list).T
        metrics = np.array(list(metrics), dtype=np.float32)
        return np.array(coords_x, dtype=np.float32), np.array(coords_y, dtype=np.float32), metrics

    def construct_map(self, bin_size=500, vmin=None, vmax=None, cm=None, title=None, figsize=None,
                      save_dir=None, pad=False, plot=True, aggr_bins='mean', aggr_bins_kwargs=None):
        """ Each value in resulted map represent aggregated value of metrics for coordinates belongs to current bin.
        """

        if isinstance(bin_size, int):
            bin_size = (bin_size, bin_size)
        coords_x, coords_y, metrics = self._split_result()
        call_aggr_bins = getattr(NumbaNumpy, aggr_bins)
        call_aggr_bins = call_aggr_bins(None, **aggr_bins_kwargs) if aggr_bins_kwargs is not None else call_aggr_bins

        metric_map = self.construct_metrics_map(coords_x=coords_x, coords_y=coords_y,
                                                metrics=metrics, bin_size=bin_size,
                                                aggr_bins=call_aggr_bins)
        extent_coords = [coords_x.min(), coords_x.max(), coords_y.min(), coords_y.max()]
        if plot:
            plot_metrics_map(metrics_map=metric_map, vmin=vmin, vmax=vmax, extent_coords=extent_coords, cm=cm,
                            title=title, figsize=figsize, save_dir=save_dir, pad=pad)
        return metric_map

    @staticmethod
    @njit(parallel=True)
    def construct_metrics_map(coords_x, coords_y, metrics, bin_size, aggr_bins):
        """njit map"""
        bin_size_x, bin_size_y = bin_size
        range_x = np.arange(coords_x.min(), coords_x.max() + bin_size_x, bin_size_x)
        range_y = np.arange(coords_y.min(), coords_y.max() + bin_size_y, bin_size_y)
        metrics_map = np.full((len(range_y), len(range_x)), np.nan)

        for i in prange(len(range_x)):
            for j in prange(len(range_y)):
                mask = ((coords_x - range_x[i] >= 0) & (coords_x - range_x[i] < bin_size_x) &
                        (coords_y - range_y[j] >= 0) & (coords_y - range_y[j] < bin_size_y))
                if mask.sum() > 0:
                    metrics_map[j, i] = aggr_bins(metrics[mask])
        return metrics_map
