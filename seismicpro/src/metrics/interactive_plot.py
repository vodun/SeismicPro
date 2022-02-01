import numpy as np
from ipywidgets import widgets
from IPython.display import display

from ..utils.interactive_plot_utils import ClickablePlot, OptionPlot


class MetricMapPlot:
    def __init__(self, metric_map, figsize=(4.5, 4.5)):
        self.metric_map = metric_map
        init_func = np.nanargmax if self.metric_map.is_lower_better else np.nanargmin
        init_y, init_x = np.unravel_index(init_func(self.metric_map.metric_map), self.metric_map.metric_map.shape)
        title = self.metric_map.plot_title
        self.left = ClickablePlot(figsize=figsize, plot_fn=lambda ax: self.metric_map.plot(ax=ax, title=""),
                                  click_fn=self.click, allow_unclick=False, title=title)#, init_x=init_x, init_y=init_y)
        self.right = OptionPlot(plot_fn=self.metric_map.metric_type.plot_on_click,
                                options=self.metric_map.get_bin_contents((init_x, init_y)))
        self.box = widgets.HBox([self.left.box, self.right.box])
        self.click_list = []

    def click(self, x, y):
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
