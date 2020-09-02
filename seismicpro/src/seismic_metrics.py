"""File contains metircs for seismic processing."""
import numpy as np

from .utils import construct_metrics_map
from .plot_utils import plot_metrics_map

from .seismic_batch import SeismicBatch
from ..batchflow import action, inbatch_parallel
from ..batchflow.models.metrics import Metrics

METRICS_ALIASES = {
    'map': 'construct_map'
}

class SemblanceMetrics():
    @action
    @inbatch_parallel(init="_init_component", target="threads")
    def calculate_minmax(self, index, src, dst):
        """some docs"""
        pos = self.get_pos(None, src, index)
        semblance = getattr(self, src)[pos]
        getattr(self, dst)[pos] = np.max(np.max(semblance, axis=1) - np.min(semblance, axis=1))
        return self

    @action
    @inbatch_parallel(init="_init_component", target="threads")
    def calculate_std(self, index, src, dst):
        """some docs"""
        pos = self.get_pos(None, src, index)
        semblance = getattr(self, src)[pos]
        getattr(self, dst)[pos] =  np.max(np.std(semblance, axis=1))
        return self

class MetricsMap(Metrics):
    """seismic metrics class"""
    def __init__(self, index, metrics, *args, map_type='sources', **kwargs):
        _ = args, kwargs
        super().__init__()
        self.metrics = metrics

        self.coords = None
        if map_type == 'sources':
            self.coords = index.get_df()[["SourceX", "SourceY"]].values[0]
        elif map_type == 'receivers':
            self.coords = index.get_df()[["GroupX", "GroupY"]].values[0]

        self._maps_list = [[*self.coords, self.metrics]]
        self._agg_fn_dict = {'mean': np.nanmean,
                             'max': np.nanmax,
                             'min': np.nanmin}

    @property
    def maps_list(self):
        """get map list"""
        return self._maps_list

    def append(self, metrics):
        """append"""
        self._maps_list.append(metrics._maps_list[0])

    def __getattr__(self, name):
        if name == "METRICS_ALIASES":
            raise AttributeError # See https://nedbatchelder.com/blog/201010/surprising_getattr_recursion.html
        name = METRICS_ALIASES.get(name, name)
        return object.__getattribute__(self, name)

    def _split_result(self):
        """split_result"""
        coords_x, coords_y, metrics = np.array(self._maps_list).T
        metrics = np.array(list(metrics), dtype=np.float32)
        return np.array(coords_x, dtype=np.float32), np.array(coords_y, dtype=np.float32), metrics

    def construct_map(self, bin_size=500, max_value=None, title=None, figsize=None, save_dir=None):
        coords_x, coords_y, metrics = self._split_result()

        metric_map = construct_metrics_map(coords_x=coords_x, coords_y=coords_y, metrics=metrics, bin_size=bin_size)
        extent_coords = [coords_x.min(), coords_x.max(), coords_y.min(), coords_y.max()]
        plot_metrics_map(metrics_map=metric_map, max_value=max_value, extent_coords=extent_coords,
                         title=title, figsize=figsize, save_dir=save_dir)
        return metric_map

