from functools import partial

import numpy as np
from ipywidgets import widgets
from IPython.display import display

from ..utils import set_text_formatting
from ..utils.interactive_plot_utils import InteractivePlot, ClickablePlot


class SemblancePlot:
    def __init__(self, semblance, stacking_velocity=None, title="Semblance", x_ticker=None, y_ticker=None,
                 figsize=(4.5, 4.5), **kwargs):
        self.semblance = semblance
        self.gather = self.semblance.gather.copy(ignore="data")

        (x_ticker, y_ticker), kwargs = set_text_formatting(x_ticker, y_ticker, **kwargs)
        plot_semblance = partial(self.semblance.plot, stacking_velocity=stacking_velocity, title=None,
                                 x_ticker=x_ticker, y_ticker=y_ticker, **kwargs)
        plot_gather = partial(self.gather.plot, title=None, x_ticker=x_ticker, y_ticker=y_ticker)

        self.left = ClickablePlot(figsize=figsize, plot_fn=plot_semblance, click_fn=self.click,
                                  unclick_fn=self.unclick, title=title)
        self.right = InteractivePlot(figsize=figsize, plot_fn=plot_gather, title="Gather", toolbar_position="right")
        self.box = widgets.HBox([self.left.box, self.right.box])

    def click(self, x, y):
        click_vel = np.interp(x, np.arange(len(self.semblance.velocities)), self.semblance.velocities / 1000)
        click_time = self.semblance.times[int(y + 0.5)]
        self.gather["Hodograph"] = np.sqrt(click_time**2 + self.gather.offsets**2/click_vel**2)

        # TODO: redraw only hodograph
        self.right.ax.clear()
        self.gather.plot(ax=self.right.ax, event_headers={"headers": "Hodograph", "alpha": 0.25,
                                                          "process_outliers": "discard"})
        self.right.set_title(f"Hodograph from {click_time:.0f} ms with {click_vel:.2f} km/s velocity")
        self.right.ax.get_legend().remove()
        return x, y

    def unclick(self):
        # TODO: remove only hodograph
        self.right.ax.clear()
        self.gather.plot(ax=self.right.ax)
        self.right.set_title("Gather")

    def plot(self):
        display(self.box)
        self.left.plot(display_box=False)
        self.right.plot(display_box=False)
