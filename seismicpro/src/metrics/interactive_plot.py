from functools import partial

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ..metrics import PlottableMetric
from ..utils import set_text_formatting, MissingModule
from ..utils.interactive_plot_utils import InteractivePlot, ClickablePlot, OptionPlot

# Safe import of modules for interactive plotting
try:
    from ipywidgets import widgets
except ImportError:
    widgets = MissingModule("ipywidgets")

try:
    from IPython.display import display
except ImportError:
    display = MissingModule("IPython.display")


class ScatterMapPlot:
    def __init__(self, metric_map, plot_on_click=None, title=None, x_ticker=None, y_ticker=None, figsize=(4.5, 4.5),
                 fontsize=8, **kwargs):
        self.metric_map = metric_map
        (x_ticker, y_ticker), kwargs = set_text_formatting(x_ticker, y_ticker, fontsize=fontsize, **kwargs)
        if plot_on_click is None:
            if not isinstance(metric_map.metric_type, PlottableMetric):
                raise ValueError("Either plot_on_click should be passed explicitly or it should be defined "
                                 "in the metric class")
            plot_on_click = metric_map.metric_type.plot_on_click
        self.plot_on_click = partial(plot_on_click, x_ticker=x_ticker, y_ticker=y_ticker)
        plot_map = partial(metric_map.plot, title="", x_ticker=x_ticker, y_ticker=y_ticker, **kwargs)
        if title is None:
            title = metric_map.plot_title

        self.coords = metric_map.map_data.index.to_frame().values
        self.coords_neighbors = NearestNeighbors(n_neighbors=1).fit(self.coords)
        worst_ix = metric_map.map_data.argmax() if metric_map.is_lower_better else metric_map.map_data.argmin()
        init_click_coords = metric_map.map_data.index[worst_ix]

        self.left = ClickablePlot(figsize=figsize, plot_fn=plot_map, click_fn=self.click, allow_unclick=False,
                                  title=title, init_click_coords=init_click_coords)
        self.right = InteractivePlot(toolbar_position="right")
        self.box = widgets.HBox([self.left.box, self.right.box])

    def click(self, coords):
        coords_ix = self.coords_neighbors.kneighbors([coords], return_distance=False).item()
        coords = self.coords[coords_ix]
        self.right.set_title(coords)
        self.right.ax.clear()
        self.plot_on_click(coords, ax=self.right.ax)
        return coords

    def plot(self):
        display(self.box)
        self.left.plot(display_box=False)
        self.right.plot(display_box=False)


class BinarizedMapPlot:
    def __init__(self, metric_map, plot_on_click=None, title=None, x_ticker=None, y_ticker=None, figsize=(4.5, 4.5),
                 fontsize=8, **kwargs):
        self.metric_map = metric_map
        (x_ticker, y_ticker), kwargs = set_text_formatting(x_ticker, y_ticker, fontsize=fontsize, **kwargs)
        if plot_on_click is None:
            if not isinstance(metric_map.metric_type, PlottableMetric):
                raise ValueError("Either plot_on_click should be passed explicitly or it should be defined "
                                 "in the metric class")
            plot_on_click = metric_map.metric_type.plot_on_click
        plot_on_click = partial(plot_on_click, x_ticker=x_ticker, y_ticker=y_ticker)
        plot_map = partial(metric_map.plot, title="", x_ticker=x_ticker, y_ticker=y_ticker, **kwargs)
        if title is None:
            title = metric_map.plot_title

        find_worst = np.nanargmax if metric_map.is_lower_better else np.nanargmin
        init_click_coords = np.unravel_index(find_worst(metric_map.map_data), metric_map.map_data.shape)

        self.left = ClickablePlot(figsize=figsize, plot_fn=plot_map, click_fn=self.click, allow_unclick=False,
                                  title=title, init_click_coords=init_click_coords)
        self.right = OptionPlot(plot_fn=plot_on_click, toolbar_position="right")
        self.box = widgets.HBox([self.left.box, self.right.box])

    def click(self, coords):
        bin_coords = (int(coords[0] + 0.5), int(coords[1] + 0.5))
        contents = self.metric_map.get_bin_contents(bin_coords)
        if contents is None:  # Handle clicks outside bins
            return None
        self.right.update_state(0, contents)
        return bin_coords

    def plot(self):
        display(self.box)
        self.left.plot(display_box=False)
        self.right.plot(display_box=False)
