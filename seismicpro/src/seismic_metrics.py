import numpy as np

from .utils import construct_metrics_map
from .plot_utils import plot_metrics_map
from ..batchflow.models.metrics import Metrics

METRICS_ALIASES = {
    'minmax': 'minmax_map',
    'std': 'std_map'
}

def _aggr_func(metrics):
    return np.nanmean(metrics)

class SeismicMetrics(Metrics):
    """seismic metrics class"""
    def __init__(self, index, diff_semblance, *args, map_type='receivers', _calc=True, **kwargs):
        super().__init__()
        self.diff = diff_semblance
        self.coords = None
        if map_type == 'receivers':
            self.coords = np.array(index.get_df()[["SourceX", "SourceY"]])[0]
        elif map_type == 'shots':
            # TODO find out where are shots stored.
            pass
        self.resulted_list = [[*self.coords, self.diff]]
        self._agg_fn_dict = {'mean': np.nanmean,
                             'max': np.nanmax,
                             'min': np.nanmin}

    def append(self, metrics):
        """append"""
        self.resulted_list.append(metrics.resulted_list[0])

    def __getattr__(self, name):
        if name == "METRICS_ALIASES":
            raise AttributeError # See https://nedbatchelder.com/blog/201010/surprising_getattr_recursion.html
        name = METRICS_ALIASES.get(name, name)
        return object.__getattribute__(self, name)

    def _split_result(self):
        """split_result"""
        coords_x, coords_y, diff = np.array(self.resulted_list).T
        diff = np.array(list(diff), dtype=np.float32)
        return np.array(coords_x, dtype=np.float32), np.array(coords_y, dtype=np.float32), diff

    def minmax_map(self, bin_size=500, max_value=None, title=None, figsize=None, save_dir=None):
        """map"""
        coords_x, coords_y, diff = self._split_result()
        metrics = np.max(np.max(diff, axis=2) - np.min(diff, axis=2), axis=1)

        metrics_map = construct_metrics_map(coords_x=coords_x, coords_y=coords_y, metrics=metrics, bin_size=bin_size)
        extent_coords = [coords_x.min(), coords_x.max(), coords_y.min(), coords_y.max()]
        plot_metrics_map(metrics_map=metrics_map, max_value=max_value, extent_coords=extent_coords,
                         title=title, figsize=figsize, save_dir=save_dir)

        return metrics_map

    def std_map(self, bin_size=500, max_value=None, title=None, figsize=None, save_dir=None):
        """map"""
        coords_x, coords_y, diff = self._split_result()
        metrics = np.max(np.std(diff, axis=2), axis=1)

        metrics_map = construct_metrics_map(coords_x=coords_x, coords_y=coords_y, metrics=metrics, bin_size=bin_size)
        extent_coords = [coords_x.min(), coords_x.max(), coords_y.min(), coords_y.max()]
        plot_metrics_map(metrics_map=metrics_map, max_value=max_value, extent_coords=extent_coords,
                         title=title, figsize=figsize, save_dir=save_dir)
        return metrics_map
