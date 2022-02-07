import numpy as np
from matplotlib import colors as mcolors

from .metric import Metric
from .interactive_plot import ScatterMapPlot, BinarizedMapPlot
from .utils import parse_accumulator_inputs
from ..decorators import plotter
from ..utils import add_colorbar, set_ticks, set_text_formatting


class MetricMap:
    def __new__(self, coords, *, agg=None, bin_size=None, coords_cols=None, metric_type=None, **metric):
        _ = coords, agg, coords_cols, metric_type, metric
        metric_cls = ScatterMap if bin_size is None else BinarizedMap
        return super().__new__(metric_cls)

    def __init__(self, coords, *, agg=None, bin_size=None, coords_cols=None, metric_type=None, **metric):
        if len(metric) != 1:
            raise ValueError("Exactly one metric must be passed to construct a map")
        metric, self.coords_cols, (self.metric_name,) = parse_accumulator_inputs(coords, metric, coords_cols)
        self.metric = metric.dropna()

        if metric_type is None:
            metric_type = Metric
        self.metric_type = metric_type

        if agg is None:
            default_agg = {True: "max", False: "min", None: "mean"}
            agg = default_agg[metric_type.is_lower_better]
        self.agg = agg

        if bin_size is not None:
            if isinstance(bin_size, (int, float, np.number)):
                bin_size = (bin_size, bin_size)
            bin_size = np.array(bin_size)
        self.bin_size = bin_size

    def __getattr__(self, name):
        return getattr(self.metric_type, name)

    @property
    def is_binarized(self):
        return self.bin_size is not None

    @property
    def plot_title(self):
        agg_name = self.agg.__name__ if callable(self.agg) else self.agg
        title = f"{agg_name}({self.metric_name})"
        if self.is_binarized:
            title += f" in {self.bin_size[0]}x{self.bin_size[1]} bins"
        return title

    def _get_tick_labels(self):
        return None, None

    @plotter(figsize=(10, 7))
    def _plot(self, title=None, cmap=None, x_ticker=None, y_ticker=None, colorbar=True, vmin=None, vmax=None,
              is_lower_better=None, ax=None, **kwargs):
        is_lower_better = self.is_lower_better if is_lower_better is None else is_lower_better
        vmin = self.vmin if vmin is None else vmin
        vmax = self.vmax if vmax is None else vmax

        if cmap is None:
            if is_lower_better is None:
                cmap = "coolwarm"
            else:
                colors = ((0.0, 0.6, 0.0), (.66, 1, 0), (0.9, 0.0, 0.0))
                if not is_lower_better:
                    colors = colors[::-1]
                cmap = mcolors.LinearSegmentedColormap.from_list("cmap", colors)
        (title, x_ticker, y_ticker), kwargs = set_text_formatting(title, x_ticker, y_ticker, **kwargs)
        ax.set_title(**{"label": self.plot_title, **title})

        res = self._plot_map(ax, is_lower_better=is_lower_better, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        add_colorbar(ax, res, colorbar, y_ticker=y_ticker)
        ax.ticklabel_format(style="plain", useOffset=False)

        x_tick_labels, y_tick_labels = self._get_tick_labels()
        set_ticks(ax, "x", self.coords_cols[0], x_tick_labels, **x_ticker)
        set_ticks(ax, "y", self.coords_cols[1], y_tick_labels, **y_ticker)

    def plot(self, interactive=False, plot_on_click=None, **kwargs):
        if not interactive:
            return self._plot(**kwargs)
        return self.interactive_plot_class(self, plot_on_click=plot_on_click, **kwargs).plot()

    def aggregate(self, agg=None, bin_size=None):
        map_params = {"coords": self.metric[self.coords_cols], self.metric_name: self.metric[self.metric_name],
                      "metric_type": self.metric_type, "agg": agg, "bin_size": bin_size}
        return MetricMap(**map_params)


class ScatterMap(MetricMap):
    def __init__(self, coords, *, agg=None, bin_size=None, coords_cols=None, metric_type=None, **metric):
        super().__init__(coords, agg=agg, bin_size=bin_size, coords_cols=coords_cols, metric_type=metric_type,
                         **metric)
        self.map_data = self.metric.explode(self.metric_name).groupby(self.coords_cols).agg(self.agg)[self.metric_name]
        self.interactive_plot_class = ScatterMapPlot

    def _plot_map(self, ax, is_lower_better, **kwargs):
        map_data = self.map_data
        if is_lower_better is None:
            global_agg = map_data.agg(self.agg)
            map_data = (map_data - global_agg).abs().sort_values(ascending=True)
        else:
            map_data = map_data.sort_values(ascending=is_lower_better)
        return ax.scatter(*np.stack(map_data.index.values).T, c=map_data.values, **kwargs)


class BinarizedMap(MetricMap):
    def __init__(self, coords, *, agg=None, bin_size=None, coords_cols=None, metric_type=None, **metric):
        super().__init__(coords, agg=agg, bin_size=bin_size, coords_cols=coords_cols, metric_type=metric_type,
                         **metric)
        metric = self.metric.copy(deep=False)

        # Binarize map coordinates
        bin_cols = ["BIN_X", "BIN_Y"]
        min_coords = metric[self.coords_cols].min(axis=0).values
        metric[bin_cols] = (metric[self.coords_cols] - min_coords) // self.bin_size
        x_bin_range = np.arange(metric["BIN_X"].max() + 1)
        y_bin_range = np.arange(metric["BIN_Y"].max() + 1)
        self.x_bin_coords = min_coords[0] + self.bin_size[0] * x_bin_range + self.bin_size[0] // 2
        self.y_bin_coords = min_coords[1] + self.bin_size[1] * y_bin_range + self.bin_size[1] // 2
        metric = metric.set_index(bin_cols + self.coords_cols)[self.metric_name].explode().sort_index()

        # Construct a binarized map
        binarized_metric = metric.groupby(bin_cols).agg(self.agg)
        x = binarized_metric.index.get_level_values(0)
        y = binarized_metric.index.get_level_values(1)
        self.map_data = np.full((len(x_bin_range), len(y_bin_range)), fill_value=np.nan)
        self.map_data[x, y] = binarized_metric

        # Construct a mapping from a bin to its contents
        bin_to_coords = metric.groupby(bin_cols + self.coords_cols).agg(self.agg)
        self.bin_to_coords = bin_to_coords.to_frame().reset_index(level=self.coords_cols).groupby(bin_cols)

        self.interactive_plot_class = BinarizedMapPlot

    def _get_tick_labels(self):
        return self.x_bin_coords, self.y_bin_coords

    def _plot_map(self, ax, is_lower_better, **kwargs):
        _ = is_lower_better
        kwargs = {"interpolation": "none", "origin": "lower", "aspect": "auto", **kwargs}
        return ax.imshow(self.map_data.T, **kwargs)

    def get_bin_contents(self, coords):
        if coords not in self.bin_to_coords.groups:
            return
        contents = self.bin_to_coords.get_group(coords).set_index(self.coords_cols)[self.metric_name]
        return contents.sort_values(ascending=not self.is_lower_better)
