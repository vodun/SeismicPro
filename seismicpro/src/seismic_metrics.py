"""File contains metircs for seismic processing."""
# pylint: disable=no-name-in-module, import-error
import inspect

import numpy as np
from numba import njit, prange

from ..batchflow.models.metrics import Metrics
from .plot_utils import plot_metrics_map


class MetricsMap(Metrics):
    """ Class for metrics aggregation and plotting. This class aims to accumulate
    coordinates and metrics values for current coordinates. Therefore, all calculations
    must be performed outside of the class.

    Parameters
    ----------
    coords : array-like
        Array of arrays or 2d array with coordinates for X and Y axes.
    kwargs : dict
        All of the given kwargs are considered as metrics. The kwargs dict has the following structure:

        ``{metircs_name_1 : metrics_value_1,
           ...
           metrics_name_N : metrics_value_N}``

        Here, the ``metric_name`` is any string while ``metrics_value`` should be represented by
        one of the following formats: a one-dimensional array, or an array of one-dimensional arrays.

            * If one-dimensional array, each value from the array will correspond to a pair of coordinates
              with the same index.
            * If an array of arrays, each array of metrics values will match to a pair of coordinates with
              the same index.

    Attributes
    ----------
    attribute_names : array-like
        Names of given metrics and coords.
    coords : array-like
        Array with shape (N, 2) that contains the X and Y coordinates.
        Where N is a number of given coordinates.
    DEFAULT_METRICS : dict
        Dict with functions for aggregation within a bin.
        Available functions:

            - std
            - min
            - max
            - mean
            - quantile
            - absquantile

    kwargs keys : array-like
        All keys from kwargs become instance attributes that contain the corresponding metric values.

    Raises
    ------
    ValueError : If kwargs are empty.
    ValueError : If ndim for given coordinate is not equal to 2.
    ValueError : If shape of first dim is not equal to 2.
    TypeError : If given coordinates are not array-like.
    TypeError : If given metrics is not array-like.
    ValueError : If the length of the metric array does not match the length of the array with coordinates.

    Note
    ----
    1. The length of the metric array and the coordinate array must match.
    """

    DEFAULT_METRICS = {
        'std' : njit(lambda array: np.nanstd(array)),
        'max' : njit(lambda array: np.nanmax(array)),
        'min' : njit(lambda array: np.nanmin(array)),
        'mean' : njit(lambda array: np.nanmin(array)),
        'median' : njit(lambda array: np.nanmedian(array)),
        'quantile' : njit(lambda array, q: np.nanquantile(array, q=q)),
        'absquantile' : njit(lambda array, q: np.nanquantile(np.abs(array - np.nanmean(array)), q))
    }

    def __init__(self, coords, **kwargs):
        super().__init__()
        if not kwargs:
            raise ValueError("At least one metric should be passed.")

        if not isinstance(coords, (list, tuple, np.ndarray)):
            raise TypeError("Wrong type of coords have been given. "\
                            "Should be array-like but {} received.".format(type(coords)))

        coords = np.asarray(coords)
        # If received array with dtype object, cast it to dtype int or float. As far as all coordinates must have
        # length 2, resulted array will have 2 dims.
        coords = np.array(coords.tolist()) if coords.ndim == 1 else coords
        if coords.ndim != 2:
            raise ValueError("Received coordinates have wrong number of dims.")
        if coords.shape[1] != 2:
            raise ValueError("An array with coordinates must have shape (N, 2), where N is a number of elements"\
                             " but given array has shape {}".format(coords.shape))
        self.coords = coords

        # Create attributes with metrics.
        for name, metrics in kwargs.items():
            if not isinstance(metrics, (list, tuple, np.ndarray)):
                raise TypeError("Received wrong type of '{}' metrics. "\
                                "Must be array-like but received {}".format(name, type(metrics)))
            metrics = np.asarray(metrics)
            # Check whether metrics contains numeric or iterable. If numeric, reshape it to 2-d array.
            if not isinstance(metrics[0], (list, tuple, np.ndarray)):
                metrics = metrics.reshape(-1, 1)
            setattr(self, name, metrics)

            if len(self.coords) != len(metrics):
                raise ValueError("Length of coordinates array doesn't match with '{0}' attribute. "\
                                 "Given length of coordinates is {1} while "\
                                 "length of '{0}' is {2}.". format(name, len(self.coords), len(metrics)))

        self.attribute_names = ('coords', ) + tuple(kwargs.keys())

        # The dictionary contains functions for aggregating the resulting map.
        self._agg_fn_dict = {'mean': np.nanmean,
                             'max': np.nanmax,
                             'min': np.nanmin}

    def append(self, metrics):
        """ Append coordinates and metrics to global container."""
        # Append all attributes with given metrics values.
        for name in self.attribute_names:
            updated_metrics = np.concatenate([getattr(self, name), getattr(metrics, name)])
            setattr(self, name, updated_metrics)

    def construct_map(self, metrics_name, bin_size=500, agg_func='mean',
                      agg_func_kwargs=None, plot=True, **plot_kwargs):
        """ All obtained coordinates are split into bins of the specified `bin_size`. Each value in the
        resulted map represents the aggregated value of metrics for coordinates that belong to the current
        bin. If there are no values included in the bin, its values is `np.nan`. Otherwise, the value of this
        bin is calculated by calling `agg_func`.

        Parameters
        ----------
        metrics_name : str
            The name of metric to draw.
        bin_size : int, float or array-like with length 2, optional, default 500
            The size of the bin by X and Y axes. Based on the received coordinates, the entire map
            will be divided into bins with the size `bin_size`.
            If int or float, the bin size will be the same for X and Y dimensions.
        agg_func : str or callable, optional, default 'mean'
            Function to aggregate metrics values in one bin.
            If str, the function from `DEFAULT_METRICS` attribute will be used for aggregation.
            If callable, it will be used for aggregation within a bin. The function used must
            be wrapped in the `njit` decorator. The first argument is a 1-d numpy.ndarray,
            containing metric values in a bin, the other arguments can take any numeric values
            and must be passed using the `agg_func_kwargs`.
        agg_func_kwargs : dict, optional
            Kwargs that will be passed to `agg_func` during evaluating.
        plot : bool, optional, default True
            If True, metrics will be plotted.
            Otherwise, the map will be returned without drawing.
        **plot_kwargs : dict
            Kwargs that are passed directly to plotter, see :func:`.plot_utils.plot_metrics_map`.

        Returns
        -------
        metrics_map : two-dimensional np.ndarray
            A matrix, where each value is an aggregated metrics value.

        Raises
        ------
        TypeError : If agg_func is not str or callable.
        ValueError : If agg_func is str and is not one of the keys of DEFAULT_METRICS.
        ValueError : If agg_func is not wrapped with @njit decorator.
        """
        metrics = getattr(self, metrics_name)

        coords_repeats = [len(metrics_array) for metrics_array in metrics]
        coords = np.repeat(self.coords, coords_repeats, axis=0)
        metrics = np.concatenate(metrics)

        coords_x = np.array(coords[:, 0], dtype=np.int32)
        coords_y = np.array(coords[:, 1], dtype=np.int32)
        metrics = np.array(metrics, dtype=np.float32)

        if isinstance(bin_size, (int, float, np.number)):
            bin_size = (bin_size, bin_size)

        if isinstance(agg_func, str):
            agg_func = self.DEFAULT_METRICS.get(agg_func, agg_func)
            if not callable(agg_func):
                raise ValueError("'{}' is not valid value for aggregation." \
                                 " Supported values are: '{}'".format(agg_func,
                                                                      "', '".join(self.DEFAULT_METRICS.keys())))
        elif not callable(agg_func):
            raise TypeError("'agg_func' should be either str or callable, not {}".format(type(agg_func)))

        # Check whether the function is njitted.
        if not hasattr(agg_func, 'py_func'):
            raise ValueError("It seems that the aggregation function is not njitted. "\
                             "Please wrap the function with @njit decorator.")
        agg_func_kwargs = dict() if agg_func_kwargs is None else agg_func_kwargs
        args = self._create_args(agg_func.py_func, **agg_func_kwargs)

        metrics_map = self.construct_metrics_map(coords_x=coords_x, coords_y=coords_y,
                                                 metrics=metrics, bin_size=bin_size,
                                                 agg_func=agg_func, args=args)

        if plot:
            ticks_range_x = [coords_x.min(), coords_x.max()]
            ticks_range_y = [coords_y.min(), coords_y.max()]
            plot_metrics_map(metrics_map=metrics_map, ticks_range_x=ticks_range_x,
                             ticks_range_y=ticks_range_y, **plot_kwargs)
        return metrics_map

    def _create_args(self, call, **kwargs):
        """ Constructing tuple with positional arguments to callable `call` based on
        `kwargs` and `call`'s defaults. The function omits the first argument even if
        it was set by `kwargs` because the first argument will be passed during the
        metrics map's calculation.

        Parameters
        ----------
        call : callable
            Function to create positional arguments for.
        kwargs : dict
            Keyword arguments to function `call`.

        Returns
        -------
        args : tuple
            Positional arguments to `call` without first argument.
        """
        params = inspect.signature(call).parameters
        args = [kwargs.get(name, param.default) for name, param in params.items()][1:]
        params_names = list(params.keys())[1:]
        empty_params = [name for name, arg in zip(params_names, args) if arg == inspect.Parameter.empty]
        if empty_params:
            raise ValueError("Missed value to '{}' argument(s).".format("', '".join(empty_params)))
        return tuple(args)

    @staticmethod
    @njit(parallel=True)
    def construct_metrics_map(coords_x, coords_y, metrics, bin_size, agg_func, args):
        """ Calculate of metrics map.

        Parameters
        ----------
        coords_x : array-like
            Coordinates for X axis.
        coords_x : array-like
            Coordinates for Y axis.
        metrics : array-like
            1d array with metrics values.
        bin_size : tuple with length 2
            The size of bin by X and Y axes.
        agg_func : numba callable in nopython mode
            Function to aggregate metrics values in one bin.

        Returns
        -------
        metrics_map : two-dimensional array
            The resulting map with aggregated metric values by bins.
        """
        bin_size_x, bin_size_y = bin_size
        range_x = np.arange(coords_x.min(), coords_x.max() + 1, bin_size_x)
        range_y = np.arange(coords_y.min(), coords_y.max() + 1, bin_size_y)
        metrics_map = np.full((len(range_y), len(range_x)), np.nan)
        for i in prange(len(range_x)): #pylint: disable=not-an-iterable
            for j in prange(len(range_y)): #pylint: disable=not-an-iterable
                mask = ((coords_x - range_x[i] >= 0) & (coords_x - range_x[i] < bin_size_x) &
                        (coords_y - range_y[j] >= 0) & (coords_y - range_y[j] < bin_size_y))
                if np.any(mask):
                    metrics_map[j, i] = agg_func(metrics[mask], *args)
        return metrics_map
