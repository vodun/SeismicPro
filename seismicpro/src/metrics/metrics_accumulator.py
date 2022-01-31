"""Implements MetricsAccumulator class for collecting metrics calculated for individual batches and MetricMap class for
a particular metric visualization over a field map"""

# pylint: disable=no-name-in-module, import-error
import numpy as np
import pandas as pd

from seismicpro.src.metrics.metric import Metric

from .metric_map import MetricMap
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
    def __init__(self, coords, *, metrics_types=None, **kwargs):
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
            raise ValueError("Coordinates array must have shape (N, 2), where N is the number of elements"
                             f" but an array with shape {coords.shape} was given")

        # Create a dict with coordinates and passed metrics values
        metrics_dict = {"x": coords[:, 0], "y": coords[:, 1]}
        for metric_name, metric_values in kwargs.items():
            if not isinstance(metric_values, (list, tuple, np.ndarray)):
                raise TypeError(f"'{metric_name}' metric value must be array-like but {type(metric_values)} received")
            metric_values = np.asarray(metric_values)

            if len(metric_values) != len(coords):
                raise ValueError(f"The length of {metric_name} metric array must match the length of coordinates "
                                 f"array ({len(coords)}) but equals {len(metric_values)}")
            metrics_dict[metric_name] = metric_values

        self.metrics_names = sorted(kwargs.keys())
        self.metrics_list = [pd.DataFrame(metrics_dict)]
        self.metrics_types = metrics_types if metrics_types is not None else {}

    @property
    def metrics(self):
        if len(self.metrics_list) > 1:
            self.metrics_list = [pd.concat(self.metrics_list, ignore_index=True)]
        return self.metrics_list[0]

    def append(self, other):
        """Append coordinates and metric values to the global container."""
        self.metrics_names = sorted(set(self.metrics_names + other.metrics_names))
        self.metrics_list += other.metrics_list
        self.metrics_types.update(other.metrics_types)

    def _process_metrics_agg(self, metrics, agg):
        is_single_metric = isinstance(metrics, str)
        metrics = to_list(metrics) if metrics is not None else self.metrics_names

        agg = to_list(agg)
        if len(agg) == 1:
            agg *= len(metrics)
        if len(agg) != len(metrics):
            raise ValueError("The number of aggregation functions must match the length of metrics to calculate")

        return metrics, agg, is_single_metric

    def evaluate(self, metrics=None, agg="mean"):
        metrics, agg, is_single_metric = self._process_metrics_agg(metrics, agg)
        metrics_vals = [self.metrics[metric].dropna().explode().agg(agg_func)
                        for metric, agg_func in zip(metrics, agg)]
        if is_single_metric:
            return metrics_vals[0]
        return metrics_vals

    def construct_map(self, metrics=None, agg="mean", bin_size=500, metrics_types=None):
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
        metrics, agg, is_single_metric = self._process_metrics_agg(metrics, agg)
        if isinstance(bin_size, (int, float, np.number)):
            bin_size = (bin_size, bin_size)
        if metrics_types is None:
            metrics_types = self.metrics_types

        # Binarize metrics for further aggregation into maps
        metrics_df = self.metrics.copy(deep=False)
        metrics_df["x_bin"] = ((metrics_df["x"] - metrics_df["x"].min()) // bin_size[0]).astype(np.int32)
        metrics_df["y_bin"] = ((metrics_df["y"] - metrics_df["y"].min()) // bin_size[1]).astype(np.int32)
        x_bin_coords = bin_size[0] * np.arange(metrics_df["x_bin"].max() + 1) + bin_size[0] // 2
        y_bin_coords = bin_size[1] * np.arange(metrics_df["y_bin"].max() + 1) + bin_size[1] // 2
        metrics_df = metrics_df.set_index(["x_bin", "y_bin", "x", "y"]).sort_index()

        # Group metrics by generated bins and create maps
        metrics_maps = []
        for metric, agg_func in zip(metrics, agg):
            metric_map = np.full((len(x_bin_coords), len(y_bin_coords)), fill_value=np.nan)
            metric_df = metrics_df[metric].dropna().explode()

            metric_agg = metric_df.groupby(["x_bin", "y_bin"]).agg(agg_func)
            x = metric_agg.index.get_level_values(0)
            y = metric_agg.index.get_level_values(1)
            metric_map[x, y] = metric_agg

            bin_to_coords = metric_df.groupby(["x_bin", "y_bin", "x", "y"]).agg(agg_func)
            bin_to_coords = bin_to_coords.to_frame().reset_index(level=["x", "y"]).groupby(["x_bin", "y_bin"])

            agg_func = agg_func.__name__ if callable(agg_func) else agg_func
            metric_type = metrics_types.get(metric, Metric)
            metric_map = MetricMap(metric_map, x_bin_coords, y_bin_coords, metric, metric_type, bin_size,
                                   bin_to_coords, agg_func)
            metrics_maps.append(metric_map)

        if is_single_metric:
            return metrics_maps[0]
        return metrics_maps
