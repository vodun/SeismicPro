"""Implements MetricsAccumulator class for collecting metrics calculated for individual batches and MetricMap class for
a particular metric visualization over a field map"""

# pylint: disable=no-name-in-module, import-error
import pandas as pd

from .metric_map import MetricMap
from .utils import parse_coords
from ...batchflow.models.metrics import Metrics


class MetricAccumulator(Metrics):
    def __init__(self, coords, metric, *, coords_cols=None, metric_type=None, metric_name=None, indices=None):
        super().__init__()
        if metric_name is None and metric_type is not None:
            metric_name = metric_type.name
        else:
            metric_name = "metric"
        coords, coords_cols = parse_coords(coords, coords_cols)
        metric_df = pd.DataFrame(coords, columns=coords_cols)
        metric_df[metric_name] = metric

        coords_to_indices = {}
        if indices is not None:
            coords_to_indices = {tuple(coords): ix for coords, ix in zip(coords, indices)}
            if len(coords_to_indices) != len(coords):  # TODO: fix by grouping them into a list
                raise ValueError("The same coordinates were returned for different indices")

        self.coords_cols = coords_cols
        self.metric_name = metric_name
        self.metric_type = metric_type
        self.metric_list = [metric_df]
        self.coords_to_indices_list = [coords_to_indices]

    @property
    def metric(self):
        if len(self.metric_list) > 1:
            self.metric_list = [pd.concat(self.metric_list, ignore_index=True)]
        return self.metric_list[0]

    @property
    def coords_to_indices(self):
        if len(self.coords_to_indices_list) > 1:
            mereged_dict = {key: val for dct in self.coords_to_indices_list for key, val in dct.items()}
            if len(mereged_dict) != sum(len(dct) for dct in self.coords_to_indices_list):
                raise ValueError("The same coordinates were returned for different indices")
            self.coords_to_indices_list = [mereged_dict]
        return self.coords_to_indices_list[0]

    def append(self, other):
        """Append coordinates and metric values to the global container."""
        if self.coords_cols != other.coords_cols:
            raise ValueError("Only MetricAccumulator with the same coordinates columns can be appended")
        if self.metric_name != other.metric_name:
            raise ValueError("Only MetricAccumulator with the same metric can be appended")
        self.metric_type = other.metric_type
        self.metric_list += other.metric_list
        self.coords_to_indices_list += other.coords_to_indices_list

    def evaluate(self, agg="mean"):
        return self.metric[self.metric_name].agg(agg)

    def construct_map(self, agg=None, bin_size=None):
        self.metric_type.finalize(self.coords_to_indices)
        return MetricMap(self.metric[self.coords_cols], self.metric[self.metric_name], coords_cols=self.coords_cols,
                         metric_type=self.metric_type, metric_name=self.metric_name, agg=agg, bin_size=bin_size)
