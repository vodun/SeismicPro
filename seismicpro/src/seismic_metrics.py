"""File contains metircs for seismic processing."""
import numpy as np
from numba import njit, prange
from scipy import stats

from .plot_utils import plot_metrics_map

from ..batchflow import action, inbatch_parallel
from ..batchflow.models.metrics import Metrics

METRICS_ALIASES = {
    'map': 'construct_map'
}

class SemblanceMetrics:
    """"Semblance metrics class"""

    @staticmethod
    @inbatch_parallel(init="_init_component", target="threads")
    def calculate_minmax(batch, index, src, dst):
        """some docs"""
        pos = batch.get_pos(None, src, index)
        semblance = getattr(batch, src)[pos]
        getattr(batch, dst)[pos] = np.max(np.max(semblance, axis=1) - np.min(semblance, axis=1))
        return batch

    @staticmethod
    @inbatch_parallel(init="_init_component", target="threads")
    def calculate_std(batch, index, src, dst):
        """some docs"""
        pos = batch.get_pos(None, src, index)
        semblance = getattr(batch, src)[pos]
        getattr(batch, dst)[pos] = np.max(np.std(semblance, axis=1))
        return batch

class PickingMetrics:
    """Docs """

    @staticmethod
    @inbatch_parallel(init="_init_component", target="threads")
    def velocity(batch, index, dst, src_picking='picking', src_offset='offset'):
        """some docs"""
        pos = batch.get_pos(None, src_picking, index)
        time = getattr(batch, src_picking)[pos]
        offset = getattr(batch, src_offset)[pos]
        mask = [time > 1]
        time = time[mask]
        offset = offset[mask]
        getattr(batch, dst)[pos] = np.mean(offset / time)
        return batch

class MetricsMap(Metrics):
    """seismic metrics class"""
    def __init__(self, metrics, coords, *args, **kwargs):
        _ = args, kwargs
        super().__init__()

        self.metrics = metrics
        self.coords = coords

        if len(self.metrics) != len(self.coords):
            raise ValueError('length of given metrics do not match with length of coords.')

        self._maps_list = [[*coord, metric] for coord, metric in zip(self.coords, self.metrics)]

        self._agg_fn_dict = {'mean': np.nanmean,
                             'max': np.nanmax,
                             'min': np.nanmin}

    @property
    def maps_list(self):
        """get map list"""
        return self._maps_list

    def append(self, metrics):
        """append"""
        self._maps_list.extend(metrics._maps_list)

    def __getattr__(self, name):
        if name == "METRICS_ALIASES":
            raise AttributeError # See https://nedbatchelder.com/blog/201010/surprising_getattr_recursion.html
        name = METRICS_ALIASES.get(name, name)
        return object.__getattribute__(self, name)

    def __split_result(self):
        """split_result"""
        coords_x, coords_y, metrics = np.array(self._maps_list).T
        metrics = np.array(list(metrics), dtype=np.float32)
        return np.array(coords_x, dtype=np.float32), np.array(coords_y, dtype=np.float32), metrics

    def construct_map(self, bin_size=500, vmin=None, vmax=None, cm=None, title=None, figsize=None, save_dir=None, pad=False, plot=True):
        """Each value in resulted map represent average value of metrics for coordinates belongs to current bin."""

        if isinstance(bin_size, int):
            bin_size = (bin_size, bin_size)
        coords_x, coords_y, metrics = self.__split_result()
        metric_map = self.construct_metrics_map(coords_x=coords_x, coords_y=coords_y, metrics=metrics, bin_size=bin_size)
        extent_coords = [coords_x.min(), coords_x.max(), coords_y.min(), coords_y.max()]
        if plot:
            plot_metrics_map(metrics_map=metric_map, vmin=vmin, vmax=vmax, extent_coords=extent_coords, cm=cm,
                            title=title, figsize=figsize, save_dir=save_dir, pad=pad)
        return metric_map

    @staticmethod
    @njit(parallel=True)
    def construct_metrics_map(coords_x, coords_y, metrics, bin_size):
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
                    metrics_map[j, i] = metrics[mask].mean()
        return metrics_map
