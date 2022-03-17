"""Implements MetricMap class for metric visualization over a field map"""

import numpy as np
import pandas as pd
from matplotlib import colors as mcolors

from .interactive_map import ScatterMapPlot, BinarizedMapPlot
from .utils import parse_coords, parse_metric_values
from ..decorators import plotter
from ..utils import to_list, add_colorbar, set_ticks, set_text_formatting


class BaseMetricMap:
    def __init__(self, coords, metric_values, *, coords_cols=None, metric=None, metric_name=None, agg=None):
        from .metrics import Metric, PartialMetric
        if metric is None:
            metric = Metric
        if not (isinstance(metric, (Metric, PartialMetric)) or
                isinstance(metric, type) and issubclass(metric, Metric)):
            raise ValueError("metric must be either of a Metric type or a subclass of Metric")

        coords, coords_cols = parse_coords(coords, coords_cols)
        metric_values, metric_name = parse_metric_values(metric_values, metric_name, metric)
        metric_data = pd.DataFrame(coords, columns=coords_cols)
        metric_data[metric_name] = metric_values

        self.metric_data = metric_data.dropna()
        self.coords_cols = coords_cols
        self.metric_name = metric_name

        if isinstance(metric, Metric):
            self.metric = metric
            self.metric.name = metric_name
            self.metric.coords_cols = coords_cols
        else:
            self.metric = PartialMetric(metric, name=metric_name, coords_cols=coords_cols)

        if agg is None:
            default_agg = {True: "max", False: "min", None: "mean"}
            agg = default_agg[self.metric.is_lower_better]
        self.agg = agg

    def __getattr__(self, name):
        return getattr(self.metric, name)

    @property
    def plot_title(self):
        agg_name = self.agg.__name__ if callable(self.agg) else self.agg
        return f"{agg_name}({self.metric_name})"

    @property
    def x_tick_labels(self):
        return None

    @property
    def y_tick_labels(self):
        return None

    @plotter(figsize=(10, 7))
    def _plot(self, *, title=None, x_ticker=None, y_ticker=None, is_lower_better=None, vmin=None, vmax=None, cmap=None,
              colorbar=True, center_colorbar=True, clip_threshold_quantile=0.95, keep_aspect=False, ax=None, **kwargs):
        is_lower_better = self.is_lower_better if is_lower_better is None else is_lower_better
        # Handle plain Metric case
        vmin = vmin or self.vmin or self.min_value
        vmax = vmax or self.vmax or self.max_value

        if is_lower_better is None and center_colorbar:
            global_mean = self.metric_data[self.metric_name].agg("mean")
            clip_threshold = (self.metric_data[self.metric_name] - global_mean).abs().quantile(clip_threshold_quantile)
            norm = mcolors.CenteredNorm(global_mean, clip_threshold)
        else:
            norm = mcolors.Normalize(vmin, vmax)

        if cmap is None:
            if is_lower_better is None:
                cmap = "coolwarm"
            else:
                colors = ((0.0, 0.6, 0.0), (.66, 1, 0), (0.9, 0.0, 0.0))
                if not is_lower_better:
                    colors = colors[::-1]
                cmap = mcolors.LinearSegmentedColormap.from_list("cmap", colors)

        (title, x_ticker, y_ticker), kwargs = set_text_formatting(title, x_ticker, y_ticker, **kwargs)
        map_obj = self._plot_map(ax, is_lower_better=is_lower_better, cmap=cmap, norm=norm, **kwargs)
        ax.set_title(**{"label": self.plot_title, **title})
        ax.ticklabel_format(style="plain", useOffset=False)
        if keep_aspect:
            ax.set_aspect("equal", adjustable="box")
        add_colorbar(ax, map_obj, colorbar, y_ticker=y_ticker)
        set_ticks(ax, "x", self.coords_cols[0], self.x_tick_labels, **x_ticker)
        set_ticks(ax, "y", self.coords_cols[1], self.y_tick_labels, **y_ticker)

    def plot(self, *, interactive=False, plot_on_click=None, **kwargs):
        if not interactive:
            return self._plot(**kwargs)

        if plot_on_click is not None:
            plot_on_click_list = to_list(plot_on_click)
        else:
            # Instantiate the metric if it hasn't been done yet
            from .metrics import Metric
            if not isinstance(self.metric, Metric):
                self.metric = self.metric()
            plot_on_click_list, kwargs = self.metric.get_views(**kwargs)
        if len(plot_on_click_list) == 0:
            raise ValueError("At least one click view must be specified")
        return self.interactive_map_class(self, plot_on_click=plot_on_click_list, **kwargs).plot()

    def aggregate(self, agg=None, bin_size=None):
        return self.map_class(self.metric_data[self.coords_cols], self.metric_data[self.metric_name],
                              metric=self.metric, agg=agg, bin_size=bin_size)


class ScatterMap(BaseMetricMap):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        exploded = self.metric_data.explode(self.metric_name)
        self.map_data = exploded.groupby(self.coords_cols).agg(self.agg)[self.metric_name]

    def _plot_map(self, ax, is_lower_better, **kwargs):
        sort_key = None
        if is_lower_better is None:
            is_lower_better = True
            global_agg = self.map_data.agg(self.agg)
            sort_key = lambda col: (col - global_agg).abs()
        # Guarantee that extreme values are always displayed on top of the others
        map_data = self.map_data.sort_values(ascending=is_lower_better, key=sort_key)
        coords_x, coords_y = map_data.index.to_frame().values.T
        x_margin = 0.05 * coords_x.ptp()
        y_margin = 0.05 * coords_y.ptp()
        ax.set_xlim(coords_x.min() - x_margin, coords_x.max() + x_margin)
        ax.set_ylim(coords_y.min() - y_margin, coords_y.max() + y_margin)
        return ax.scatter(coords_x, coords_y, c=map_data, **kwargs)

    def get_worst_coords(self, is_lower_better=None):
        is_lower_better = self.is_lower_better if is_lower_better is None else is_lower_better
        if is_lower_better is None:
            data = (self.map_data - self.metric_data[self.metric_name].agg(self.agg)).abs()
        elif is_lower_better:
            data = self.map_data
        else:
            data = -self.map_data
        return self.map_data.index[data.argmax()]


class BinarizedMap(BaseMetricMap):
    """
    The map is constructed in the following way:
        1. All stored coordinates are divided into bins of the specified `bin_size`.
        2. All metric values are grouped by their bin.
        3. An aggregation is performed by calling `agg_func` for values in each bin. If no metric values were assigned
           to a bin, `np.nan` is returned.
        As a result, each value of the constructed map represents an aggregated metric for a particular bin.
    """
    def __init__(self, *args, bin_size, **kwargs):
        super().__init__(*args, **kwargs)

        if bin_size is not None:
            if isinstance(bin_size, (int, float, np.number)):
                bin_size = (bin_size, bin_size)
            bin_size = np.array(bin_size)
        self.bin_size = bin_size

        # Perform a shallow copy of the metric data since new columns are going to be appended
        metric_data = self.metric_data.copy(deep=False)

        # Binarize map coordinates
        bin_cols = ["BIN_X", "BIN_Y"]
        min_coords = metric_data[self.coords_cols].min(axis=0).values
        metric_data[bin_cols] = (metric_data[self.coords_cols] - min_coords) // self.bin_size
        x_bin_range = np.arange(metric_data["BIN_X"].max() + 1)
        y_bin_range = np.arange(metric_data["BIN_Y"].max() + 1)
        self.x_bin_coords = min_coords[0] + self.bin_size[0] * x_bin_range + self.bin_size[0] // 2
        self.y_bin_coords = min_coords[1] + self.bin_size[1] * y_bin_range + self.bin_size[1] // 2
        metric_data = metric_data.set_index(bin_cols + self.coords_cols)[self.metric_name].explode().sort_index()

        # Construct a binarized map
        binarized_metric = metric_data.groupby(bin_cols).agg(self.agg)
        x = binarized_metric.index.get_level_values(0)
        y = binarized_metric.index.get_level_values(1)
        self.map_data = np.full((len(x_bin_range), len(y_bin_range)), fill_value=np.nan)
        self.map_data[x, y] = binarized_metric

        # Construct a mapping from a bin to its contents
        bin_to_coords = metric_data.groupby(bin_cols + self.coords_cols).agg(self.agg)
        self.bin_to_coords = bin_to_coords.to_frame().reset_index(level=self.coords_cols).groupby(bin_cols)

    @property
    def plot_title(self):
        return super().plot_title + f" in {self.bin_size[0]}x{self.bin_size[1]} bins"

    @property
    def x_tick_labels(self):
        return self.x_bin_coords

    @property
    def y_tick_labels(self):
        return self.y_bin_coords

    def _plot_map(self, ax, is_lower_better, **kwargs):
        _ = is_lower_better
        kwargs = {"interpolation": "none", "origin": "lower", "aspect": "auto", **kwargs}
        return ax.imshow(self.map_data.T, **kwargs)

    def get_worst_coords(self, is_lower_better=None):
        is_lower_better = self.is_lower_better if is_lower_better is None else is_lower_better
        if is_lower_better is None:
            data = np.abs(self.map_data - self.metric_data[self.metric_name].agg(self.agg))
        elif is_lower_better:
            data = self.map_data
        else:
            data = -self.map_data
        return np.unravel_index(np.nanargmax(data), self.map_data.shape)

    def get_bin_contents(self, coords):
        if coords not in self.bin_to_coords.groups:
            return None
        contents = self.bin_to_coords.get_group(coords).set_index(self.coords_cols)[self.metric_name]
        return contents.sort_values(ascending=not self.is_lower_better)


class MetricMapMeta(type):
    def __call__(cls, *args, bin_size=None, **kwargs):
        if bin_size is None:
            map_class = cls.scatter_map_class
            interactive_map_class = cls.interactive_scatter_map_class
        else:
            map_class = cls.binarized_map_class
            interactive_map_class = cls.interactive_binarized_map_class
            kwargs["bin_size"] = bin_size

        instance = object.__new__(map_class)
        instance.__init__(*args, **kwargs)
        instance.map_class = cls
        instance.interactive_map_class = interactive_map_class
        return instance


class MetricMap(metaclass=MetricMapMeta):
    scatter_map_class = ScatterMap
    binarized_map_class = BinarizedMap
    interactive_scatter_map_class = ScatterMapPlot
    interactive_binarized_map_class = BinarizedMapPlot
