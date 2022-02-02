from functools import partial

import numpy as np
from ipywidgets import widgets
from IPython.display import display

from ..utils import set_text_formatting
from ..utils.interactive_plot_utils import ClickablePlot, OptionPlot


class MetricMapPlot:
    def __init__(self, metric_map, plot_on_click=None, title=None, x_ticker=None, y_ticker=None, figsize=(4.5, 4.5),
                 fontsize=8, **kwargs):
        self.metric_map = metric_map
        find_worst = np.nanargmax if self.metric_map.is_lower_better else np.nanargmin
        init_click_coords = np.unravel_index(find_worst(metric_map.metric_map), metric_map.metric_map.shape)
        init_options = metric_map.get_bin_contents(init_click_coords)

        (x_ticker, y_ticker), kwargs = set_text_formatting(x_ticker, y_ticker, fontsize=fontsize, **kwargs)
        if title is None:
            title = self.metric_map.plot_title

        plot_map = partial(self.metric_map.plot, title=None, x_ticker=x_ticker, y_ticker=y_ticker, **kwargs)
        if plot_on_click is None:
            plot_on_click = self.metric_map.metric_type.plot_on_click
        plot_on_click = partial(plot_on_click, x_ticker=x_ticker, y_ticker=y_ticker)

        self.left = ClickablePlot(figsize=figsize, plot_fn=plot_map, click_fn=self.click, allow_unclick=False,
                                  title=title)
        self.right = OptionPlot(plot_fn=plot_on_click, options=init_options, toolbar_position="right")
        self.box = widgets.HBox([self.left.box, self.right.box])
        self.click_list = []

    def click(self, coords):
        x, y = coords
        x = int(x + 0.5)
        y = int(y + 0.5)
        contents = self.metric_map.get_bin_contents((x, y))
        if contents is None:
            return
        self.right.update_state(0, contents)
        return x, y

    def plot(self):
        display(self.box)
        self.left.plot(display_box=False)
        self.right.plot(display_box=False)
        # TODO: Fix init click
        # self.left._click(self.init_click_coords)
