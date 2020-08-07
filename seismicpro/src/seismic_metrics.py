import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

from ..batchflow.models.metrics import Metrics

METRICS_ALIASES = {
    'minmax': 'minmax_map',
    'std': 'std_map'
}
class SeismicMetrics(Metrics):
    """"""
    def __init__(self, index, diff_semblance, *args, map_type='receivers', _calc=True, **kwargs):
        super().__init__()
        self.diff = diff_semblance
        self.coords = None
        self.coords = np.array([*index])
        # if map_type == 'receivers':
        #     self.coords = np.array(index.get_df()[["SourceX", "SourceY"]])[0]
        # elif map_type == 'shots':
            # TODO find out where are shots stored.
            # pass

        self.resulted_list = [[*self.coords, self.diff]]

    def append(self, metrics):
        self.resulted_list.append(metrics.resulted_list[0])

    def __getattr__(self, name):
        if name == "METRICS_ALIASES":
            raise AttributeError # See https://nedbatchelder.com/blog/201010/surprising_getattr_recursion.html
        name = METRICS_ALIASES.get(name, name)
        return object.__getattribute__(self, name)

    def _split_result(self):
        coords_x = []
        coords_y = []
        diff = []
        for x, y, d in self.resulted_list:
            coords_x.append(x)
            coords_y.append(y)
            diff.append(d)
        # resulted_list = np.array(self.resulted_list)
        return np.array(coords_x), np.array(coords_y), np.array(diff)

    # @njit(parallel=True)
    def calc_metric(self, rps_x, rps_y, rps_val, bin_size):
        x_min = rps_x.min()
        x_max = rps_x.max()
        x = np.arange(x_min, x_max, bin_size)

        y_min = rps_y.min()
        y_max = rps_y.max()
        y = np.arange(y_min, y_max, bin_size)

        metric_map = np.full((len(y), len(x)), np.nan)
        print(metric_map)
        for i in prange(len(x)):
            for j in prange(len(y)):
                mask = ((rps_x - x[i] > 0) & (rps_x - x[i] < bin_size) &
                        (rps_y - y[j] > 0) & (rps_y - y[j] < bin_size))
                if mask.sum() > 0:
                    metric_map[j, i] = rps_val[mask].mean()
        return metric_map

    def minmax_map(self, bin_size=500):
        coords_x, coords_y, diff = self._split_result()

        metrics = np.max(np.max(diff, axis=2) - np.min(diff, axis=2), axis=1)

        print(type(coords_x), type(coords_y), type(metrics))
        metric_map = self.calc_metric(rps_x=coords_x, rps_y=coords_y, rps_val=metrics, bin_size=bin_size)
        print(metric_map)
        plt.imshow(metric_map, origin='lower')#, extent=[x_min, x_max, y_min, y_max])
        return metric_map

    def std_map(self, bin_size=500):
        coords_x, coords_y, diff = self._split_result()

        metrics = np.max(np.std(diff, axis=2), axis=1)

        metric_map = self.calc_metric(coords_x, coords_y, metrics, bin_size)
        plt.imshow(metric_map, origin='lower')#, extent=[x_min, x_max, y_min, y_max])

