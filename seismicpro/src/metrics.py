"""Implements MetricsMap class for metric visualization over a field map"""

# pylint: disable=no-name-in-module, import-error
import inspect

import numpy as np
from numba import njit, prange

from .utils import plot_metrics_map
from ..batchflow.models.metrics import Metrics


class MetricsMap(Metrics):
    """Accumulate metric values and their coordinates to further aggregate them into a metrics map.

    Parameters
    ----------
    coords : array-like
        Array of arrays or 2d array with coordinates for X and Y axes.
    kwargs : misc
        Metrics and their values to aggregate and plot on the map. The `kwargs` dict has the following structure:
        `{metric_name_1: metric_values_1,
          ...,
          metric_name_N: metric_values_N
         }`

        Here, `metric_name` is any `str` while `metric_values` should have one of the following formats:
        * If 1d array, each value corresponds to a pair of coordinates with the same index.
        * If an array of arrays, all values from each inner array correspond to a pair of coordinates with the same
          index as in the outer metrics array.
        In both cases the length of `metric_values` must match the length of coordinates array.

    Attributes
    ----------
    attribute_names : tuple
        Names of passed metrics and coords.
    coords : 2d np.ndarray
        An array with shape (N, 2) which contains X and Y coordinates for each corresponding metric value.
    DEFAULT_METRICS : dict
        A dictionary of aggregation functions within a bin. Available functions include:
            - std
            - min
            - max
            - mean
            - quantile
            - absquantile
    kwargs keys : np.ndarray
        All keys from `kwargs` become instance attributes and contain the corresponding metric values.

    Raises
    ------
    ValueError
        If `kwargs` were not passed.
        If `ndim` for given coordinate is not equal to 2.
        If shape of the first dim of the coordinate array is not equal to 2.
        If the length of the metric array does not match the length of the array with coordinates.
    TypeError
        If given coordinates are not array-like.
        If given metrics are not array-like.
    """
    DEFAULT_METRICS = {
        'std' : njit(lambda array: np.nanstd(array)),
        'max' : njit(lambda array: np.nanmax(array)),
        'min' : njit(lambda array: np.nanmin(array)),
        'mean' : njit(lambda array: np.nanmean(array)),
        'median' : njit(lambda array: np.nanmedian(array)),
        'quantile' : njit(lambda array, q: np.nanquantile(array, q=q)),
        'absquantile' : njit(lambda array, q: np.nanquantile(np.abs(array - np.nanmean(array)), q))
    }

    def __init__(self, coords, **kwargs):
        super().__init__()

        if not kwargs:
            raise ValueError("At least one metric should be passed.")

        if not isinstance(coords, (list, tuple, np.ndarray)):
            raise TypeError(f"coords must be array-like but {type(coords)} was given.")
        coords = np.asarray(coords)

        # If coords is an array of arrays, convert it to an array with numeric dtype and check its shape
        coords = np.array(coords.tolist()) if coords.ndim == 1 else coords
        if coords.ndim != 2:
            raise ValueError("Coordinates array must be 2-dimensional.")
        if coords.shape[1] != 2:
            raise ValueError("Coordinates  array must have shape (N, 2), where N is the number of elements"\
                             " but an array with shape {} was given".format(coords.shape))
        self.coords = coords

        # Create attributes with metric values
        for name, metrics in kwargs.items():
            if not isinstance(metrics, (list, tuple, np.ndarray)):
                raise TypeError(f"'{name}' metric must be array-like but {type(metrics)} received.")
            metrics = np.asarray(metrics)

            # If metrics is a 1d array with numeric dtype, convert it to a 2d array to unify further computation logic
            # with the case when metrics is an array of arrays
            if not isinstance(metrics[0], (list, tuple, np.ndarray)):
                metrics = metrics.reshape(-1, 1)
            if len(self.coords) != len(metrics):
                raise ValueError("Length of coordinates array ({0}) doesn't match the length of '{1}' "\
                                 "attribute ({2}).".format(len(self.coords), name, len(metrics)))
            setattr(self, name, metrics)

        self.attribute_names = ('coords',) + tuple(kwargs.keys())

        # The dictionary stores functions to aggregate the resulting metrics map
        self._agg_fn_dict = {'mean': np.nanmean,
                             'max': np.nanmax,
                             'min': np.nanmin}

    def append(self, metrics):
        """Append coordinates and metrics to the global container."""
        for name in self.attribute_names:
            updated_metrics = np.concatenate([getattr(self, name), getattr(metrics, name)])
            setattr(self, name, updated_metrics)

    def construct_map(self, metric_name, bin_size=500, agg_func='mean', agg_func_kwargs=None, plot=True,
                      **plot_kwargs):
        """Calculate and optionally plot a metrics map.

        The map is constructed in the following way:
        1. All stored coordinates are divided into bins of the specified `bin_size`.
        2. All metric values are grouped by their bin.
        3. An aggregation is performed by calling `agg_func` for values in each bin. If no metric values were assigned
           to a bin, `np.nan` is returned.
        As a result, each value of the constructed map represents an aggregated metric for a particular bin.

        Parameters
        ----------
        metric_name : str
            The name of a metric to construct a map for.
        bin_size : int, float or array-like with length 2, optional, defaults to 500
            Bin size for X and Y axes. If single `int` or `float`, the same bin size will be used for both axes.
        agg_func : str or callable, optional, defaults to 'mean'
            Function to aggregate metric values in a bin.
            If `str`, the function from `DEFAULT_METRICS` will be used.
            If `callable`, it will be used directly. Note, that the function must be wrapped with `njit` decorator.
            Its first argument is a 1d np.ndarray containing metric values in a bin, all other arguments can take any
            numeric values and must be passed using the `agg_func_kwargs`.
        agg_func_kwargs : dict, optional
            Additional keyword arguments to be passed to `agg_func`.
        plot : bool, optional, defaults to True
            Whether to plot the constructed map.
        plot_kwargs : misc, optional
            Additional keyword arguments to be passed to :func:`.plot_utils.plot_metrics_map`.

        Returns
        -------
        metrics_map : 2d np.ndarray
            A map with aggregated metric values.

        Raises
        ------
        TypeError
            If `agg_func` is not `str` or `callable`.
        ValueError
            If `agg_func` is `str` and is not in DEFAULT_METRICS.
            If `agg_func` is not wrapped with `njit` decorator.
        """
        metrics = getattr(self, metric_name)

        # Handle the case when metric is an array of arrays by flattening the metrics array and duplicating the coords
        coords_repeats = [len(metrics_array) for metrics_array in metrics]
        coords = np.repeat(self.coords, coords_repeats, axis=0)
        metrics = np.concatenate(metrics)

        coords_x = np.array(coords[:, 0], dtype=np.int32)
        coords_y = np.array(coords[:, 1], dtype=np.int32)
        metrics = np.array(metrics, dtype=np.float32)

        if isinstance(bin_size, (int, float, np.number)):
            bin_size = (bin_size, bin_size)

        # Parse passed agg_func and check whether it is njitted.
        if isinstance(agg_func, str):
            agg_func = self.DEFAULT_METRICS.get(agg_func, agg_func)
            if not callable(agg_func):
                err_msg = "'{}' is not a valid aggregation function. Available options are: '{}'"
                raise ValueError(err_msg.format(agg_func, "', '".join(self.DEFAULT_METRICS.keys())))
        elif not callable(agg_func):
            raise TypeError(f"'agg_func' should be either str or callable, not {type(agg_func)}")

        if not hasattr(agg_func, 'py_func'):
            raise ValueError("It seems that the aggregation function is not njitted. "\
                             "Please wrap the function with @njit decorator.")

        # Convert passed agg_func kwargs to args since numba does not support kwargs unpacking and construct the map
        agg_func_kwargs = {} if agg_func_kwargs is None else agg_func_kwargs
        agg_func_args = self._kwargs_to_args(agg_func.py_func, **agg_func_kwargs)
        metrics_map = self.construct_metrics_map(coords_x=coords_x, coords_y=coords_y, metrics=metrics,
                                                 bin_size=bin_size, agg_func=agg_func, agg_func_args=agg_func_args)

        if plot:
            ticks_range_x = [coords_x.min(), coords_x.max()]
            ticks_range_y = [coords_y.min(), coords_y.max()]
            plot_metrics_map(metrics_map=metrics_map, ticks_range_x=ticks_range_x, ticks_range_y=ticks_range_y,
                             **plot_kwargs)
        return metrics_map

    @staticmethod
    def _kwargs_to_args(func, **kwargs):
        """Convert function kwargs to args.

        Currently, `numba` does not support kwargs unpacking but allows for unpacking args. That's why `kwargs` are
        transformed into `args` before the call with the first argument omitted even if it was set by `kwargs` since
        it will be passed automatically during the metrics map calculation.

        Parameters
        ----------
        func : callable
            Function to create positional arguments for.
        kwargs : misc, optional
            Keyword arguments to `func`.

        Returns
        -------
        args : tuple
            Positional arguments to `func` except for the first argument.
        """
        params = inspect.signature(func).parameters
        args = [kwargs.get(name, param.default) for name, param in params.items()][1:]
        params_names = list(params.keys())[1:]
        empty_params = [name for name, arg in zip(params_names, args) if arg == inspect.Parameter.empty]
        if empty_params:
            raise ValueError("Missed value to '{}' argument(s).".format("', '".join(empty_params)))
        return tuple(args)

    @staticmethod
    @njit(parallel=True)
    def construct_metrics_map(coords_x, coords_y, metrics, bin_size, agg_func, agg_func_args):
        """Calculate metrics map.

        Parameters
        ----------
        coords_x : 1d np.ndarray
            Metrics coordinates for X axis.
        coords_y : 1d np.ndarray
            Metrics coordinates for Y axis.
        metrics : 1d np.ndarray
            Metric values for corresponding coordinates.
        bin_size : tuple with length 2
            Bin size for X and Y axes.
        agg_func : njitted callable
            Aggregation function, whose first argument is a 1d np.ndarray containing metric values in a bin.
        agg_func_args : tuple
            Additional positional arguments to `agg_func`.

        Returns
        -------
        metrics_map : 2d np.ndarray
            A map with aggregated metric values.
        """
        bin_size_x, bin_size_y = bin_size
        range_x = np.arange(coords_x.min(), coords_x.max() + 1, bin_size_x)
        range_y = np.arange(coords_y.min(), coords_y.max() + 1, bin_size_y)
        metrics_map = np.full((len(range_y), len(range_x)), np.nan)
        for i in prange(len(range_x)):  #pylint: disable=not-an-iterable
            for j in prange(len(range_y)):  #pylint: disable=not-an-iterable
                mask = ((coords_x - range_x[i] >= 0) & (coords_x - range_x[i] < bin_size_x) &
                        (coords_y - range_y[j] >= 0) & (coords_y - range_y[j] < bin_size_y))
                if np.any(mask):
                    metrics_map[j, i] = agg_func(metrics[mask], *agg_func_args)
        return metrics_map
