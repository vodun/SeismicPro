"""Implements MetricsAccumulator class that collects metric values calculated for individual subsets of data and
aggregates them into maps"""

import pandas as pd

from .metrics import Metric, PartialMetric
from .utils import parse_coords
from ..utils import to_list, align_args


class MetricsAccumulator:
    """Accumulate metric values and their coordinates to further aggregate them into metric maps.

    Examples
    --------
    Accumulate minimum and maximum amplitudes for each common source gather in a survey:
    >>> survey = Survey(path, header_index="FieldRecord", header_cols=["SourceY", "SourceX", "offset"], name="raw")
    >>> dataset = SeismicDataset(survey)
    >>> pipeline = (dataset
    ...     .pipeline()
    ...     .load(src="raw")
    ...     .gather_metrics(MetricsAccumulator, coords=L("raw").coords, min_amplitude=L("raw").data.min(),
    ...                     max_amplitude=L("raw").data.max(), save_to=V("accumulator", mode="a"))
    ... )
    >>> pipeline.run(batch_size=16, n_epochs=1)
    >>> accumulator = pipeline.v("accumulator")

    The calculated metric values can be aggregated over the whole field. Global minimum and maximum amplitudes can be
    obtained as follows:
    >>> min_amplitude = accumulator.evaluate("min_amplitude", agg="min")
    >>> max_amplitude = accumulator.evaluate("max_amplitude", agg="max")

    Accumulator object can construct a map for each metric to display its values over the field map:
    >>> min_map, max_map = accumulator.construct_map(["min_amplitude", "max_amplitude"])
    >>> min_map.plot()
    >>> max_map.plot()

    Parameters
    ----------
    coords : 2d array-like with 2 columns
        Metrics coordinates for X and Y axes.
    coords_cols : array-like with 2 elements, optional
        Names of X and Y coordinates. Usually names of survey headers used to extract coordinates from. Defaults to
        ("X", "Y") if not given and cannot be inferred from `coords`.
    kwargs : misc
        Metrics and their values to accumulate. Each `kwargs` item define metric name and its values in one of the
        following formats:
        * A 1d array-like: defines a single metric value for the corresponding pair of `coords`,
        * An array of 1d arrays: defines several metric values for the corresponding pair of `coords`,
        * A `dict` with the following keys:
            * "values" - metric values as a 1d array-like or an array of 1d arrays as explained above,
            * "metric_type" - the class of the metric (optional, defaults to `Metric`),

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
    def __init__(self, coords, *, coords_cols=None, **kwargs):
        super().__init__()
        coords, coords_cols = parse_coords(coords, coords_cols)
        metrics_df = pd.DataFrame(coords, columns=coords_cols)

        metrics_types = {}
        metrics_names = []
        for metric_name, metric_val in kwargs.items():
            if not isinstance(metric_val, dict):
                metric_val = {"values": metric_val}
            metric_val = {"metric_type": Metric, **metric_val}

            expected_keys = {"values", "metric_type"}
            if metric_val.keys() != expected_keys:
                raise ValueError(f"If metric value is dict, its keys must be only {', '.join(expected_keys)}")

            # Cast metric type to degenerate PartialMetric to simplify append logic
            metrics_types[metric_name] = PartialMetric(metric_val["metric_type"])
            metrics_df[metric_name] = metric_val["values"]
            metrics_names.append(metric_name)

        self.coords_cols = coords_cols
        self.metrics_list = [metrics_df]
        self.metrics_names = metrics_names
        self.metrics_types = metrics_types

    @property
    def metrics(self):
        """pd.DataFrame: collected metrics and their coordinates."""
        if len(self.metrics_list) > 1:
            self.metrics_list = [pd.concat(self.metrics_list)]
        return self.metrics_list[0]

    def append(self, other):
        """Append data from `other` accumulator to `self`."""
        # TODO: allow for accumulation of different metrics
        if (set(self.coords_cols) != set(other.coords_cols)) or (set(self.metrics_names) != set(other.metrics_names)):
            raise ValueError("Only MetricsAccumulator with the same coordinates columns and metrics can be appended")
        self.metrics_list += other.metrics_list
        for name in self.metrics_names:
            self.metrics_types[name].update(other.metrics_types[name])

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

        metrics_maps = []
        for metric, metric_agg, metric_bin_size in zip(metrics, agg, bin_size):
            metric_type = self.metrics_types[metric]
            metric_map = metric_type.map_class(self.metrics[self.coords_cols], self.metrics[metric],
                                               metric=metric_type, agg=metric_agg, bin_size=metric_bin_size)
            metrics_maps.append(metric_map)

        if is_single_metric:
            return metrics_maps[0]
        return metrics_maps
