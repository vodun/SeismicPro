"""Implements MetricsAccumulator class that collects metric values calculated for individual subsets of data and
aggregates them into maps"""

# pylint: disable=no-name-in-module, import-error
import pandas as pd

from .metrics import Metric, PartialMetric
from .utils import parse_coords
from ..utils import to_list, align_args
from ...batchflow.models.metrics import Metrics


class MetricsAccumulator(Metrics):
    """Accumulate metric values and their coordinates to further aggregate them into metric maps.

    Parameters
    ----------
    coords : 2d array-like with 2 columns
        Metrics coordinates for X and Y axes.
    coords_cols : array-like with 2 elements, optional
        Names of X and Y coordinates. Usually names of survey headers used to extract coordinates from. Defaults to
        ("X", "Y") if not given and cannot be inferred from `coords`.
    indices : pandas.Index, optional
        Dataset indices, that produced corresponding `coords` and metric values. May be used by `PipelineMetric` to
        speed up batch generation on click on an interactive metric map.
    kwargs : misc
        Metrics and their values to accumulate. Each `kwargs` item define metric name and its values in one of the
        following formats:
        * A 1d array-like: defines a single metric value for the corresponding pair of `coords`,
        * An array of 1d arrays: defines several metric values for the corresponding pair of `coords`,
        * A `dict` with the following keys:
            * "values" - metric values as a 1d array-like or an array of 1d arrays as explained above,
            * "metric_type" - the class of the metric (optional, defaults to `Metric`),
            * Any other key-value pairs will be used to further instantiate the metric class.

    Attributes
    ----------
    coords_cols : array-like with 2 elements
        Names of X and Y coordinates.
    metrics_list : list of pandas.DataFrame
        Accumulated metrics values. Should not be used directly but via `metrics` property.
    metrics_names : list of str
        Names of accumulated metrics.
    metrics_types : list of subclasses of Metric
        Types of accumulated metrics.
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
        """pd.DataFrame: collected coordinates, metrics and indices."""
        if len(self.metrics_list) > 1:
            self.metrics_list = [pd.concat(self.metrics_list)]
        return self.metrics_list[0]

    def append(self, other):
        """Append data from `other` accumulator to `self`."""
        # TODO: allow for accumulation of different metrics
        if (set(self.coords_cols) != set(other.coords_cols)) or (set(self.metrics_names) != set(other.metrics_names)):
            raise ValueError("Only MetricsAccumulator with the same coordinates columns and metrics can be appended")
        if ((self.stores_indices != other.stores_indices) or
            (self.metrics_list[0].index.names != other.metrics_list[0].index.names)):
            raise ValueError("Both accumulators must store the same types of indices")
        self.metrics_list += other.metrics_list
        self.metrics_types = other.metrics_types

    def _parse_requested_metrics(self, metrics):
        is_single_metric = isinstance(metrics, str) or metrics is None and len(self.metrics_names) == 1
        metrics = to_list(metrics) if metrics is not None else self.metrics_names
        return metrics, is_single_metric

    def evaluate(self, metrics=None, agg="mean"):
        """Aggregate metrics values.

        Parameters
        ----------
        metrics : str or list of str or None, optional
            Metrics to evaluate. If not given, evaluate all the accumulated metrics in the order they appear in
            `metrics_names`.
        agg : str or callable or list of str or callable, optional, defaults to "mean"
            A function used for aggregating values of each metric from `metrics`. If a single `agg` is given it will be
            used to evaluate all the `metrics`. Passed directly to `pandas.core.groupby.DataFrameGroupBy.agg`.

        Returns
        -------
        metrics_vals : float or list of float
            Evaluated metrics values. Has the same shape as `metrics`.
        """
        metrics, is_single_metric = self._parse_requested_metrics(metrics)
        metrics, agg = align_args(metrics, agg)
        metrics_vals = [self.metrics[metric].dropna().explode().agg(metric_agg)
                        for metric, metric_agg in zip(metrics, agg)]
        if is_single_metric:
            return metrics_vals[0]
        return metrics_vals

    def construct_map(self, metrics=None, agg=None, bin_size=None):
        """Aggregate metrics values into field maps.

        Parameters
        ----------
        metrics : str or list of str or None, optional
            Metrics to construct field maps for. If not given, construct maps for all the accumulated metrics in the
            order they appear in `metrics_names`.
        agg : str or callable or list of str or callable, optional
            A function used for aggregating each metric from `metrics` by coordinates. If a single `agg` is given it
            will be used to aggregate all the `metrics`. If not given, will be determined by the value of
            `is_lower_better` attribute of the corresponding metric class in order to highlight outliers. Passed
            directly to `pandas.core.groupby.DataFrameGroupBy.agg`.
        bin_size : int, float or array-like with length 2, optional
            Bin size for X and Y axes. If single `int` or `float`, the same bin size will be used for both axes.

        Returns
        -------
        metrics_maps : BaseMetricMap or list of BaseMetricMap
            Constructed maps. Has the same shape as `metrics`.
        """
        metrics, is_single_metric = self._parse_requested_metrics(metrics)
        metrics, agg, bin_size = align_args(metrics, agg, bin_size)

        coords_to_indices = None
        if self.stores_indices:
            # Rename metrics coordinates columns to avoid possible collision with index names which breaks groupby
            renamed_metrics = self.metrics.rename(columns=dict(zip(self.coords_cols, ["X", "Y"])))
            coords_to_indices = renamed_metrics.groupby(by=["X", "Y"]).groups
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
