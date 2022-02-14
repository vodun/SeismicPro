import numpy as np
import pandas as pd
from matplotlib import colors as mcolors

from .metric import is_metric, Metric, PlottableMetric, PartialMetric
from .interactive_plot import ScatterMapPlot, BinarizedMapPlot
from .utils import parse_coords, parse_metric_values
from ..decorators import plotter
from ..utils import add_colorbar, set_ticks, set_text_formatting


class MetricMap:
    def __new__(self, coords, metric_values, *, coords_cols=None, metric=None, metric_name=None, agg=None,
                bin_size=None):
        _ = coords, metric_values, coords_cols, metric, agg
        metric_cls = ScatterMap if bin_size is None else BinarizedMap
        return super().__new__(metric_cls)

    def __init__(self, coords, metric_values, *, coords_cols=None, metric=None, metric_name=None, agg=None,
                 bin_size=None):
        if metric is None:
            metric = Metric
        if not is_metric(metric):
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

        if bin_size is not None:
            if isinstance(bin_size, (int, float, np.number)):
                bin_size = (bin_size, bin_size)
            bin_size = np.array(bin_size)
        self.bin_size = bin_size

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
    def _plot(self, title=None, x_ticker=None, y_ticker=None, is_lower_better=None, vmin=None, vmax=None, cmap=None,
              colorbar=True, center_colorbar=True, threshold_quantile=0.95, ax=None, **kwargs):
        is_lower_better = self.is_lower_better if is_lower_better is None else is_lower_better
        vmin = vmin or self.vmin or self.min_value
        vmax = vmax or self.vmax or self.max_value

        if is_lower_better is None and center_colorbar:
            global_agg = self.metric_data[self.metric_name].agg(self.agg)
            threshold = (self.metric_data[self.metric_name] - global_agg).abs().quantile(threshold_quantile)
            norm = mcolors.CenteredNorm(self.metric_data[self.metric_name].agg(self.agg), threshold)
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
        res = self._plot_map(ax, is_lower_better=is_lower_better, cmap=cmap, norm=norm, **kwargs)
        ax.set_title(**{"label": self.plot_title, **title})
        ax.ticklabel_format(style="plain", useOffset=False)
        add_colorbar(ax, res, colorbar, y_ticker=y_ticker)
        set_ticks(ax, "x", self.coords_cols[0], self.x_tick_labels, **x_ticker)
        set_ticks(ax, "y", self.coords_cols[1], self.y_tick_labels, **y_ticker)

    def plot(self, *args, interactive=False, plot_on_click=None, **kwargs):
        if not interactive:
            return self._plot(*args, **kwargs)

        # Instantiate the metric if it hasn't been done yet
        if isinstance(self.metric, (type, PartialMetric)):
            self.metric = self.metric()
        if plot_on_click is None and not isinstance(self.metric, PlottableMetric):
            raise ValueError("plot_on_click must be passed if not defined in the metric class")
        if plot_on_click is None:
            plot_on_click = self.plot_on_click
        return self.interactive_map_class(self, plot_on_click, *args, **kwargs).plot()

    def aggregate(self, agg=None, bin_size=None):
        return MetricMap(self.metric_data[self.coords_cols], self.metric_data[self.metric_name],
                         coords_cols=self.coords_cols, metric=self.metric, agg=agg, bin_size=bin_size)


class ScatterMap(MetricMap):
    def __init__(self, coords, metric_values, *, coords_cols=None, metric=None, agg=None, bin_size=None):
        super().__init__(coords, metric_values, coords_cols=coords_cols, metric=metric, agg=agg, bin_size=bin_size)
        exploded = self.metric_data.explode(self.metric_name)
        self.map_data = exploded.groupby(self.coords_cols).agg(self.agg)[self.metric_name]

    @property
    def interactive_map_class(self):
        return getattr(self, "interactive_scatter_map_class", ScatterMapPlot)

    def _plot_map(self, ax, is_lower_better, **kwargs):
        key = None
        if is_lower_better is None:
            is_lower_better = True
            global_agg = self.map_data.agg(self.agg)
            key = lambda col: (col - global_agg).abs()
        map_data = self.map_data.sort_values(ascending=is_lower_better, key=key)
        return ax.scatter(*map_data.index.to_frame().values.T, c=map_data, **kwargs)

    def get_worst_coords(self, is_lower_better=None):
        is_lower_better = self.is_lower_better if is_lower_better is None else is_lower_better
        if is_lower_better is None:
            data = (self.map_data - self.metric_data[self.metric_name].agg(self.agg)).abs()
        elif is_lower_better:
            data = self.map_data
        else:
            data = -self.map_data
        return self.map_data.index[data.argmax()]


class BinarizedMap(MetricMap):
    def __init__(self, coords, metric_values, *, coords_cols=None, metric=None, agg=None, bin_size=None):
        super().__init__(coords, metric_values, coords_cols=coords_cols, metric=metric, agg=agg, bin_size=bin_size)
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
    def interactive_map_class(self):
        return getattr(self, "interactive_binarized_map_class", BinarizedMapPlot)

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

    def get_worst_coords(self, is_lower_better):
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
            return
        contents = self.bin_to_coords.get_group(coords).set_index(self.coords_cols)[self.metric_name]
        return contents.sort_values(ascending=not self.is_lower_better)
