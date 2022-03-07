"""Implements MetricsAccumulator class that accumulates metrics calculated for individual batches of data and
aggregates them into maps"""

# pylint: disable=no-name-in-module, import-error
import pandas as pd

from .metrics import Metric, PartialMetric
from .metric_map import MetricMap
from .utils import parse_coords
from ..utils import to_list, align_args
from ...batchflow.models.metrics import Metrics


class MetricsAccumulator(Metrics):
    """Accumulate metric values and their coordinates to further aggregate them into metric maps.

    Parameters
    ----------
    coords : array-like
        Array of arrays or 2d array with coordinates for X and Y axes.
    coords_cols : ...
        ...
    indices : ...
        ...
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
    """
    def __init__(self, coords, *, coords_cols=None, indices=None, **kwargs):
        super().__init__()
        coords, coords_cols = parse_coords(coords, coords_cols)
        metrics_df = pd.DataFrame(coords, columns=coords_cols, index=indices)

        metrics_types = {}
        metrics_names = []
        for metric_name, metric_val in kwargs.items():
            if not isinstance(metric_val, dict):
                metric_val = {"values": metric_val}
            metric_val = {"metric_type": Metric, **metric_val}
            metric_values = metric_val.pop("values")
            metric_type = metric_val.pop("metric_type")
            metrics_df[metric_name] = metric_values
            metrics_types[metric_name] = PartialMetric(metric_type, **metric_val)
            metrics_names.append(metric_name)

        self.coords_cols = coords_cols
        self.metrics_list = [metrics_df]
        self.metrics_names = metrics_names
        self.metrics_types = metrics_types
        self.stores_indices = indices is not None

    @property
    def metrics(self):
        if len(self.metrics_list) > 1:
            self.metrics_list = [pd.concat(self.metrics_list)]
        return self.metrics_list[0]

    def append(self, other):
        """Append coordinates and metric values to the global container."""
        # TODO: allow for accumulation of different metrics
        if (set(self.coords_cols) != set(other.coords_cols)) or (set(self.metrics_names) != set(other.metrics_names)):
            raise ValueError("Only MetricsAccumulator with the same coordinates columns and metrics can be appended")
        if ((self.stores_indices != self.stores_indices) or
            (self.metrics_list[0].index.names != other.metrics_list[0].index.names)):
            raise ValueError("Both accumulators must store the same types of indices")
        self.metrics_list += other.metrics_list
        self.metrics_types = other.metrics_types

    def _parse_requested_metrics(self, metrics):
        is_single_metric = isinstance(metrics, str) or metrics is None and len(self.metrics_names) == 1
        metrics = to_list(metrics) if metrics is not None else self.metrics_names
        return metrics, is_single_metric

    def evaluate(self, metrics=None, agg="mean"):
        metrics, is_single_metric = self._parse_requested_metrics(metrics)
        metrics, agg = align_args(metrics, agg)
        metrics_vals = [self.metrics[metric].dropna().explode().agg(metric_agg)
                        for metric, metric_agg in zip(metrics, agg)]
        if is_single_metric:
            return metrics_vals[0]
        return metrics_vals

    def construct_map(self, metrics=None, agg=None, bin_size=None):
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
        metrics, is_single_metric = self._parse_requested_metrics(metrics)
        metrics, agg, bin_size = align_args(metrics, agg, bin_size)

        coords_to_indices = None
        if self.stores_indices:
            coords_to_indices = self.metrics.groupby(by=self.coords_cols).groups
            coords_to_indices = {coords: indices.unique() for coords, indices in coords_to_indices.items()}

        metrics_maps = []
        for metric, metric_agg, metric_bin_size in zip(metrics, agg, bin_size):
            metric_type = PartialMetric(self.metrics_types[metric], coords_to_indices=coords_to_indices)
            metric_map = metric_type.map_class(self.metrics[self.coords_cols], self.metrics[metric],
                                               metric=metric_type, agg=metric_agg, bin_size=metric_bin_size)
            metrics_maps.append(metric_map)

        if is_single_metric:
            return metrics_maps[0]
        return metrics_maps
