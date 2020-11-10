"""File contains metircs for seismic processing."""
# pylint: disable=no-name-in-module, import-error
import numpy as np
from numba import njit, prange

from ..batchflow.models.metrics import Metrics

from .plot_utils import plot_metrics_map


class MetricsMap(Metrics):
    """seismic metrics class"""

    def __init__(self, coords, *args, **kwargs):
        _ = args
        super().__init__()
        if not kwargs:
            raise ValueError('Add at least one metrics.')

        # Create attributes with metrics.
        for name, metrics in kwargs.items():
            if isinstance(metrics, (int, float, bool, np.number)):
                setattr(self, name, [metrics])
            elif isinstance(metrics, (list, tuple, set, np.ndarray)):
                setattr(self, name, [*metrics])
            else:
                raise ValueError("Wrong type of metrics have been given. "\
                                 "Should be number or array but {} received.".format(type(metrics)))

        self.metrics_names = list(kwargs.keys())
        self.coords = list(coords)
        self.call_agg_bins = None
        self.call_name = None

        self._agg_fn_dict = {'mean': np.nanmean,
                             'max': np.nanmax,
                             'min': np.nanmin}

    def extend(self, metrics):
        """Extend coordinates and metrics to global container."""
        # Extend all attributes with given metrics values.
        for name in self.metrics_names:
            metrics_attr = getattr(self, name)
            metrics_attr.extend(getattr(metrics, name))
            setattr(self, name, metrics_attr)

        self.coords.extend(metrics.coords)

    def construct_map(self, metrics_name, bin_size=500, cm=None, title=None, figsize=None, save_to=None, dpi=None, #pylint: disable=too-many-arguments
                      pad=False, plot=True, agg_bins_fn='mean', agg_bins_kwargs=None, **plot_kwargs):
        """ Each value in resulted map represent aggregated value of metrics for coordinates belongs to current bin.
        """
        metrics = np.array(list(getattr(self, metrics_name)))

        if not len(metrics):
            raise ValueError('Given metrics is empty.')

        # if metrics has an array for one coordinate, we repeat the coordinate value
        # and expand the metric values into a one-dimensional array.
        if isinstance(metrics[0], (list, tuple, set, np.ndarray)):
            len_of_copy = [len(metrics_array) for metrics_array in metrics]
            coords = np.repeat(self.coords, len_of_copy, axis=0)
            metrics = np.concatenate(metrics)
        else:
            coords = np.array(self.coords)

        if len(metrics) != len(coords):
            raise ValueError("The length of given metrics is not corresponds with length of the coordinates.\
                              Check the metrics array, it is souldn't have a nested structure.")

        coords_x = np.array(coords[:, 0], dtype=np.int32)
        coords_y = np.array(coords[:, 1], dtype=np.int32)
        metrics = np.array(metrics, dtype=np.float32)

        if isinstance(bin_size, int):
            bin_size = (bin_size, bin_size)

        if isinstance(agg_bins_fn, str):
            call_name = agg_bins_fn
            call_agg_bins = getattr(NumbaNumpy, agg_bins_fn)
            call_agg_bins = call_agg_bins(**agg_bins_kwargs) if agg_bins_kwargs else call_agg_bins
        elif callable(agg_bins_fn):
            call_name = agg_bins_fn.__name__
            call_agg_bins = agg_bins_fn
        else:
            raise ValueError('agg_bins_fn should be whether str or callable, not {}'.format(type(agg_bins_fn)))

        # We need to avoid recompiling the numba function if aggregation function hasn't changed.
        if self.call_name is None or self.call_name != call_name:
            self.call_name = call_name
            self.call_agg_bins = call_agg_bins

        metric_map = self.construct_metrics_map(coords_x=coords_x, coords_y=coords_y,
                                                metrics=metrics, bin_size=bin_size,
                                                agg_bins_fn=self.call_agg_bins)
        if plot:
            extent = [coords_x.min(), coords_x.max(), coords_y.min(), coords_y.max()]
            # Avoid the situation when we have only one coordinate for x or y dimension.
            extent[1] += 1 if extent[0] - extent[1] == 0 else 0
            extent[3] += 1 if extent[2] - extent[3] == 0 else 0
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


class NumbaNumpy:
    """ Holder for jit-accelerated functions. """
    #pylint: disable = unnecessary-lambda, undefined-variable
    min = njit(lambda array: np.nanmin(array))
    max = njit(lambda array: np.nanmax(array))
    mean = njit(lambda array: np.nanmean(array))
    std = njit(lambda array: np.nanstd(array))

    @staticmethod
    def quantile(q):
        """ numba quantile. """
        return  njit(lambda array: np.quantile(array, q=q))

    @staticmethod
    def absquantile(q):
        """ numba absquantile. """
        return njit(lambda array: _absquantile(array, q=q))

    @staticmethod
    @njit
    def _absquantile(array, q):
        shifted_array = array - np.mean(array)
        return np.quantile(np.abs(shifted_array), q)
