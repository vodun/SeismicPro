"""File contains metircs for seismic processing."""
# pylint: disable=no-name-in-module, import-error
import numpy as np
from numba import njit, prange

from ..batchflow.models.metrics import Metrics

from .plot_utils import plot_metrics_map


class MetricsMap(Metrics):
    """ Class for metrics aggregation and plotting. The aim on this class is accumulate coordinates
    and quality values for current coordinates. Therefore, all calculation of quality values or coordinates
    must be performed outside of the class.

    Parameters
    ----------
    coords : array-like with length 2
        Array with coordinates for X and Y axes.
    kwargs : dict
        All given kwargs are considered quality values. Key value - any name, while value must be
        whether number or 1-dimensional array.

    Attributes
    ----------
    metrics_names : array-like
        names of existing metrics.
    coords : array-like
        Array with shape (N, 2) that contains the X and Y coordinates.
        Where N is a number of given coordinates.
    aggr_func : str or callable
        Function to aggregate metrics values in one bin.
        If str, should be attribue's name from :class:`.NumbaNumpy`.
        if callable, it will be applied for every bin independently.
    call_name : str
        Contains the name of used aggregation function.
    Note
    ----
    1. All keys from kwargs become class attributes.
    """

    def __init__(self, coords, **kwargs):
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
        self.aggr_func = None
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

    def construct_map(self, metrics_name, bin_size=500, aggr_func='mean',
                      agg_func_kwargs=None, plot=True, **plot_kwargs):
        """ All optained coordinates, are splitted into bins of the specified `bin_size`. Resulted map
        is an array in which every value reflects the metric's value in one current bin. If there are no values
        are included in the bin, it's value is None. Otherwise, the value of this bin is calculated based
        on the aggregation function `agg_func`.

        Each value in resulted map represent aggregated value of metrics for coordinates belongs to current bin.

        Parameters
        ----------
        metrics_name : str
            The name of metric to draw.
        bin_size : int or array-like with length 2, optional, default 500
            The size of bin by X and Y axes. Based on the received coordinates, the entire map
            will be divided into bins with size `bin_size`.
            If int, the bin size will be same for X and Y dimensions.
        aggr_func : str or callable, optional, default 'mean'
            Function to aggregate metrics values in one bin.
            If str, should be attribue's name from meth:`NumbaNumpy`.
            if callable, it will be applied for every bin independently.
        agg_func_kwargs : dict, optional
            Kwargs that will be applied to aggr_func before evaluating.
        plot : bool, optional, default True
            If True, metrics will be plotted
        **plot_kwargs : dict
            Kwargs that are passed directly to plotter, see :func:`.plot_utils.plot_metrics_map`.
            (allowed arguments: cm, title, figsize, save_to, dpi, pad, font_size, x_ticks, y_ticks
                               and kwargs to :func:`matplotlib.pyplot.imshow`.)
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

        if isinstance(aggr_func, str):
            call_name = aggr_func
            aggr_func = getattr(NumbaNumpy, aggr_func)
            aggr_func = aggr_func(**agg_func_kwargs) if agg_func_kwargs else aggr_func
        elif callable(aggr_func):
            call_name = aggr_func.__name__
            aggr_func = aggr_func
        else:
            raise ValueError('aggr_func should be whether str or callable, not {}'.format(type(aggr_func)))

        # We need to avoid recompiling the numba function if aggregation function hasn't changed.
        if self.call_name is None or self.call_name != call_name:
            self.call_name = call_name
            self.aggr_func = aggr_func

        metric_map = self.construct_metrics_map(coords_x=coords_x, coords_y=coords_y,
                                                metrics=metrics, bin_size=bin_size,
                                                aggr_func=self.aggr_func)

        if plot:
            extent = [coords_x.min(), coords_x.max(), coords_y.min(), coords_y.max()]
            # Avoid the situation when we have only one coordinate for x or y dimension.
            extent[1] += 1 if extent[0] - extent[1] == 0 else 0
            extent[3] += 1 if extent[2] - extent[3] == 0 else 0
            plot_metrics_map(metrics_map=metric_map, extent=extent, **plot_kwargs)
        return metric_map

    @staticmethod
    @njit(parallel=True)
    def construct_metrics_map(coords_x, coords_y, metrics, bin_size, aggr_func):
        """Calculation of metrics map.

        Parameters
        ----------
        coords_x : array-like
            Coordinates for X axis.
        coords_x : array-like
            Coordinates for Y axis.
        metrics : array-like
            Quality values.
        bin_size : tuple with length 2
            The size of bin by X and Y axes.
        arrg_func : numba callable
            Function to aggregate metrics values in one bin.
        """
        bin_size_x, bin_size_y = bin_size
        range_x = np.arange(coords_x.min(), coords_x.max() + 1, bin_size_x)
        range_y = np.arange(coords_y.min(), coords_y.max() + 1, bin_size_y)
        metrics_map = np.full((len(range_y), len(range_x)), np.nan)
        for i in prange(len(range_x)): #pylint: disable=not-an-iterable
            for j in prange(len(range_y)): #pylint: disable=not-an-iterable
                mask = ((coords_x - range_x[i] >= 0) & (coords_x - range_x[i] < bin_size_x) &
                        (coords_y - range_y[j] >= 0) & (coords_y - range_y[j] < bin_size_y))
                if mask.sum() > 0:
                    metrics_map[j, i] = aggr_func(metrics[mask])
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
