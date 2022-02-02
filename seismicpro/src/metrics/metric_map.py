from matplotlib import colors as mcolors

from .interactive_plot import MetricMapPlot
from ..decorators import plotter
from ..utils import as_dict, add_colorbar, set_ticks


class MetricMap:
    def __init__(self, metric_map, x_bin_coords, y_bin_coords, coords_cols, metric_name, metric_type, bin_size,
                 bin_to_coords, agg_func):
        self.metric_map = metric_map
        self.x_bin_coords = x_bin_coords
        self.y_bin_coords = y_bin_coords
        self.coords_cols = coords_cols
        self.metric_name = metric_name
        self.metric_type = metric_type
        self.bin_size = bin_size
        self.bin_to_coords = bin_to_coords
        self.agg_func = agg_func

    def __getattr__(self, name):
        return getattr(self.metric_type, name)

    @property
    def plot_title(self):
        return f"{self.agg_func}({self.metric_name}) in {self.bin_size[0]}x{self.bin_size[1]} bins"

    def get_bin_contents(self, coords):
        if coords not in self.bin_to_coords.groups:
            return
        contents = self.bin_to_coords.get_group(coords).set_index(self.coords_cols)[self.metric_name]
        return contents.sort_values(ascending=not self.is_lower_better)

    @plotter(figsize=(10, 7))
    def plot(self, title=None, interpolation="none", origin="lower", aspect="auto", cmap=None, x_ticker=None,
             y_ticker=None, colorbar=True, vmin=None, vmax=None, is_lower_better=True, ax=None, **kwargs):
        if cmap is None:
            colors = ((0.0, 0.6, 0.0), (.66, 1, 0), (0.9, 0.0, 0.0))
            if not is_lower_better:
                colors = colors[::-1]
            cmap = mcolors.LinearSegmentedColormap.from_list("cmap", colors)
        vmin = self.vmin if vmin is None else vmin
        vmax = self.vmax if vmax is None else vmax
        img = ax.imshow(self.metric_map.T, origin=origin, aspect=aspect, cmap=cmap, interpolation=interpolation,
                        vmin=vmin, vmax=vmax, **kwargs)
        add_colorbar(ax, img, colorbar)

        title = {} if title is None else as_dict(title, key="label")
        title = {"label": self.plot_title, **title}
        ax.set_title(**title)

        x_ticker = {} if x_ticker is None else x_ticker
        y_ticker = {} if y_ticker is None else y_ticker
        set_ticks(ax, "x", self.coords_cols[0], self.x_bin_coords, **x_ticker)
        set_ticks(ax, "y", self.coords_cols[1], self.y_bin_coords, **y_ticker)

    def plot_interactive(self):
        MetricMapPlot(self).plot()
