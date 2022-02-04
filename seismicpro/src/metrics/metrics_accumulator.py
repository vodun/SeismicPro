"""Implements MetricsAccumulator class for collecting metrics calculated for individual batches and MetricMap class for
a particular metric visualization over a field map"""

# pylint: disable=no-name-in-module, import-error
import numpy as np
import pandas as pd

from .metric import Metric
from .metric_map import MetricMap
from .utils import parse_accumulator_inputs
from ..utils import to_list
from ...batchflow.models.metrics import Metrics


class MetricsAccumulator(Metrics):
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
    metrics_list : list of pd.DataFrame
        An array with shape (N, 2) which contains X and Y coordinates for each corresponding metric value.
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
    def __init__(self, coords, *, coords_cols=None, metrics_types=None, **metrics):
        super().__init__()
        metrics, self.coords_cols, self.metrics_names = parse_accumulator_inputs(coords, metrics, coords_cols)
        self.metrics_list = [metrics]
        self.metrics_types = metrics_types if metrics_types is not None else {}

    @property
    def metrics(self):
        if len(self.metrics_list) > 1:
            self.metrics_list = [pd.concat(self.metrics_list, ignore_index=True)]
        return self.metrics_list[0]

    def append(self, other):
        """Append coordinates and metric values to the global container."""
        if self.coords_cols != other.coords_cols:
            raise ValueError("Only MetricsAccumulator with the same coordinates columns can be appended")
        self.metrics_names = sorted(set(self.metrics_names + other.metrics_names))
        self.metrics_list += other.metrics_list
        self.metrics_types.update(other.metrics_types)

    def evaluate(self, metrics=None, agg="mean"):
        is_single_metric = isinstance(metrics, str)
        metrics = to_list(metrics) if metrics is not None else self.metrics_names

        agg = to_list(agg)
        if len(agg) == 1:
            agg *= len(metrics)
        if len(agg) != len(metrics):
            raise ValueError("The number of aggregation functions must match the length of metrics to calculate")

        metrics_vals = [self.metrics[metric].dropna().explode().agg(agg_func)
                        for metric, agg_func in zip(metrics, agg)]
        if is_single_metric:
            return metrics_vals[0]
        return metrics_vals

    def construct_maps(self, metrics=None, map_params=None, agg=None, bin_size=None, metric_type=None):
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
        is_single_metric = isinstance(metrics, str)
        metrics = to_list(metrics) if metrics is not None else self.metrics_names
        if map_params is None:
            map_params = {}

        metrics_maps = []
        for metric in metrics:
            metric_map_params = {"metric_type": self.metrics_types.get(metric, metric_type), "agg": agg,
                                 "bin_size": bin_size, **map_params.get(metric, {}), metric: self.metrics[metric],
                                 "coords": self.metrics[self.coords_cols]}
            metrics_maps.append(MetricMap(**metric_map_params))

        if is_single_metric:
            return metrics_maps[0]
        return metrics_maps
