"""File contains metircs for seismic processing."""
# pylint: disable=no-name-in-module, import-error
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
    coords : array-like with length 2
        Array with coordinates for X and Y axes.
    kwargs : dict
        All given kwargs are considered quality values. The key value is any name,
        and the value of the metric can be represented by a number, a one-dimensional vector,
        or an array of one-dimensional arrays.
        A metric can only be a number if one coordinate was passed, then the metric value
        will correspond to that coordinate.
        If the metric is a one-dimensional vector, each value from the array will correspond
        to its own pair of coordinates.
        If the metric is an array of arrays, each coordinate will have an array suit to it.

    Attributes
    ----------
    metrics_names : array-like
        Names of given metrics and coords.
    coords : array-like
        Array with shape (N, 2) that contains the X and Y coordinates.
        Where N is a number of given coordinates.
    agg_func : str or callable
        See :meth:`.construct_map`
    call_name : str
        Contains the name of the used aggregation function.

    Raises
    ------
    ValueError : If kwargs are empty.
    ValueError : If given coordinates have more than two values for one metrics value.
    TypeError : If given coordinates are not array-like.
    TypeError : If the metric is not a number or an array.
    ValueError : If the length of the metric array does not match the length of the array with coordinates.

    Note
    ----
    1. All keys from kwargs become class attributes.
    2. The length of the metric array and the coordinate array must match.
    """

    def __init__(self, coords, **kwargs):
        super().__init__()
        if not kwargs:
            raise ValueError("At least one metric should be passed.")

        if isinstance(coords, (list, tuple, set, np.ndarray)):
            self.coords = np.asarray(coords)
            for coord_ix, coord in enumerate(coords):
                if len(coord) != 2:
                    raise ValueError("An array with coordinates must contain only two values "\
                                     "(X and Y) for each metrics value, but the coordinate with index "\
                                     "{} has length = {}.".format(coord_ix, len(coord)))
        else:
            raise TypeError("Wrong type of coords have been given. "\
                            "Should be array-like but {} received.".format(type(coords)))

        # Create attributes with metrics.
        for name, metrics in kwargs.items():
            if isinstance(metrics, (int, float, bool, np.number)):
                setattr(self, name, np.asarray([metrics]))
            elif isinstance(metrics, (list, tuple, set)):
                setattr(self, name, np.array(metrics, dtype=np.object))
            elif isinstance(metrics, np.ndarray):
                setattr(self, name, metrics)
            else:
                raise TypeError("Wrong type of metrics have been given. "\
                                "Should be number or array but {} received.".format(type(metrics)))

            if len(self.coords) != len(getattr(self, name)):
                raise ValueError("Length of coordinates array doesn't match with '{0}' attribute. "\
                                 "Given length of coordinates is {1} while "\
                                 "length of '{0}' is {2}.". format(name, len(self.coords), len(getattr(self, name))))

        self.metrics_names = ('coords', ) + tuple(kwargs.keys())
        self.agg_func = None
        self.call_name = None

        # The dictionary contains functions for aggregating the resulting map.
        self._agg_fn_dict = {'mean': np.nanmean,
                             'max': np.nanmax,
                             'min': np.nanmin}

    def extend(self, metrics):
        """Extend coordinates and metrics to global container."""
        # Extend all attributes with given metrics values.
        for name in self.metrics_names:
            updated_metrics = np.append(getattr(self, name), getattr(metrics, name), axis=0)
            setattr(self, name, updated_metrics)

    def construct_map(self, metrics_name, bin_size=500, agg_func='mean',
                      agg_func_kwargs=None, plot=True, **plot_kwargs):
        """ All optained coordinates are split into bins of the specified `bin_size`. The resulted map
        is an array in which every value reflects the metric's value in one current bin. If there are
        no values are included in the bin, it values is np.nan. Otherwise, the value of this bin is
        calculated based on the aggregation function `agg_func`.

        Each value in the resulted map represents the aggregated value of metrics for coordinates
        belongs to the current bin.

        Parameters
        ----------
        metrics_name : str
            The name of metric to draw.
        bin_size : int, float or array-like with length 2, optional, default 500
            The size of the bin by X and Y axes. Based on the received coordinates, the entire map
            will be divided into bins with the size `bin_size`.
            If int, the bin size will be the same for X and Y dimensions.
        agg_func : str or callable, optional, default 'mean'
            Function to aggregate metrics values in one bin.
            If str, the function from :class:`.NumbaNumpy` will be used for aggregation.
            If callable, it will be used for aggregation as this. The function used must
            be wrapped in the `njit` decorator. The first argument is data for aggregation
            within the bin, the other arguments can take any numeric values and must be
            passed using the `agg_func_kwargs`.
        agg_func_kwargs : dict, optional
            Kwargs that will be applied to agg_func before evaluating.
        plot : bool, optional, default True
            If True, metrics will be plotted.
            Otherwise, the map will be returned without drawing.
        **plot_kwargs : dict
            Kwargs that are passed directly to plotter, see :func:`.plot_utils.plot_metrics_map`.

        Returns
        -------
            : two-dimensional np.ndarray
            A matrix, where each value corresponds to the aggregated value
            of all metrics included in a specific bin.

        Raises
        ------
        ValueError : If agg_func is not str or callable.
        """
        metrics = getattr(self, metrics_name)

        # If metric has an array for one coordinate, we repeat the coordinate value
        # and expand the metric values into a one-dimensional array.
        if isinstance(metrics[0], (list, tuple, set, np.ndarray)):
            len_of_copy = [len(metrics_array) for metrics_array in metrics]
            coords = np.repeat(self.coords, len_of_copy, axis=0)
            metrics = np.concatenate(metrics)
        else:
            coords = self.coords

        coords_x = np.array(coords[:, 0], dtype=np.int32)
        coords_y = np.array(coords[:, 1], dtype=np.int32)
        metrics = np.array(metrics, dtype=np.float32)

        if isinstance(bin_size, (int, float, np.number)):
            bin_size = (bin_size, bin_size)

        if isinstance(agg_func, str):
            call_name = agg_func
            agg_func = getattr(NumbaNumpy, agg_func)
        elif callable(agg_func):
            call_name = agg_func.__name__
        else:
            raise ValueError('agg_func should be whether str or callable, not {}'.format(type(agg_func)))

        agg_func = agg_func(**agg_func_kwargs) if agg_func_kwargs else agg_func

        # We need to avoid recompiling the numba function if aggregation function hasn't changed.
        if self.call_name is None or self.call_name != call_name:
            self.call_name = call_name
            self.agg_func = agg_func

        metrics_map = self.construct_metrics_map(coords_x=coords_x, coords_y=coords_y,
                                                metrics=metrics, bin_size=bin_size,
                                                agg_func=self.agg_func)

        if plot:
            extent = [coords_x.min(), coords_x.max(), coords_y.min(), coords_y.max()]
            # Avoid the situation when we have only one unique coordinate for x or y dimension.
            # Because in this case our maximum and minimum values for this axis will be
            # the same, and imshow will draw an empty image.
            extent[1] += 1 if extent[0] - extent[1] == 0 else 0
            extent[3] += 1 if extent[2] - extent[3] == 0 else 0
            plot_metrics_map(metrics_map=metrics_map, extent=extent, **plot_kwargs)
        return metrics_map

    @staticmethod
    @njit(parallel=True)
    def construct_metrics_map(coords_x, coords_y, metrics, bin_size, agg_func):
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

        Returns
        -------
            : two-dimensional array
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
                if mask.sum() > 0:
                    metrics_map[j, i] = agg_func(metrics[mask])
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
