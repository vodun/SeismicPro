"""Implements MetricsAccumulator class for collecting metrics calculated for individual batches and MetricMap class for
a particular metric visualization over a field map"""

# pylint: disable=no-name-in-module, import-error
import pandas as pd

from .metric import Metric, PartialMetric
from .metric_map import MetricMap
from .utils import parse_coords
from ..utils import to_list
from ...batchflow.models.metrics import Metrics


class MetricAccumulator(Metrics):
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
            raise ValueError("Only MetricAccumulator with the same coordinates columns and metrics can be appended")
        if ((self.stores_indices != self.stores_indices) or
            (self.metrics_list[0].index.names != other.metrics_list[0].index.names)):
            raise ValueError("Both accumulators must store the same types of indices")
        self.metrics_list += other.metrics_list
        self.metrics_types = other.metrics_types

    def _process_aggregation_args(self, metrics, *args):
        is_single_metric = isinstance(metrics, str) or metrics is None and len(self.metrics_names) == 1
        metrics = to_list(metrics) if metrics is not None else self.metrics_names
        processed_args = []
        for arg in args:
            arg = to_list(arg)
            if len(arg) == 1:
                arg *= len(metrics)
            if len(arg) != len(metrics):
                raise ValueError("Lengths of all passed arguments must match the length of metrics to calculate")
            processed_args.append(arg)
        return metrics, *processed_args, is_single_metric

    def evaluate(self, metrics=None, agg="mean"):
        metrics, agg, is_single_metric = self._process_aggregation_args(metrics, agg)
        metrics_vals = [self.metrics[metric].dropna().explode().agg(metric_agg)
                        for metric, metric_agg in zip(metrics, agg)]
        if is_single_metric:
            return metrics_vals[0]
        return metrics_vals

    def construct_map(self, metrics=None, agg=None, bin_size=None):
        metrics, agg, bin_size, is_single_metric = self._process_aggregation_args(metrics, agg, bin_size)

        coords_to_indices = None
        if self.stores_indices:
            coords_to_indices = self.metrics.groupby(by=self.coords_cols).groups
            coords_to_indices = {coords: indices.unique() for coords, indices in coords_to_indices.items()}

        metrics_maps = []
        for metric, metric_agg, metric_bin_size in zip(metrics, agg, bin_size):
            metric_type = PartialMetric(self.metrics_types[metric], coords_to_indices=coords_to_indices)
            metric_map = MetricMap(self.metrics[self.coords_cols], self.metrics[metric], metric=metric_type,
                                   agg=metric_agg, bin_size=metric_bin_size)
            metrics_maps.append(metric_map)

        if is_single_metric:
            return metrics_maps[0]
        return metrics_maps
