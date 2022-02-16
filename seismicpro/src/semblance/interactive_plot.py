from functools import partial

import numpy as np

from ..utils import set_text_formatting, times_to_indices, MissingModule
from ..utils.interactive_plot_utils import InteractivePlot, ClickablePlot, PairedPlot

# Safe import of modules for interactive plotting
try:
    from ipywidgets import widgets
except ImportError:
    widgets = MissingModule("ipywidgets")

try:
    from IPython.display import display
except ImportError:
    display = MissingModule("IPython.display")


class SemblancePlot(PairedPlot):
    def __init__(self, semblance, *args, sharey=True, title="Semblance", x_ticker=None, y_ticker=None,
                 figsize=(4.5, 4.5), fontsize=8, gather_plot_kwargs=None, **kwargs):
        (x_ticker, y_ticker), kwargs = set_text_formatting(x_ticker, y_ticker, fontsize=fontsize, **kwargs)
        if gather_plot_kwargs is None:
            gather_plot_kwargs = {}
        gather_plot_kwargs = {"title": None, **gather_plot_kwargs}
        gather_plot_kwargs["x_ticker"] = {**x_ticker, **gather_plot_kwargs.get("x_ticker", {})}
        gather_plot_kwargs["y_ticker"] = {**y_ticker, **gather_plot_kwargs.get("y_ticker", {})}

        self.figsize = figsize
        self.title = title
        self.hodograph = None

        self.semblance = semblance
        self.gather = self.semblance.gather.copy(ignore="data")
        self.plot_semblance = partial(self.semblance.plot, *args, title=None, x_ticker=x_ticker, y_ticker=y_ticker,
                                      **kwargs)
        self.plot_gather = partial(self.gather.plot, **gather_plot_kwargs)

        super().__init__()
        if sharey:
            self.right.ax.sharey(self.left.ax)

    def construct_left_plot(self):
        return ClickablePlot(figsize=self.figsize, title=self.title, plot_fn=self.plot_semblance, click_fn=self.click,
                             unclick_fn=self.unclick)

    def construct_right_plot(self):
        return InteractivePlot(figsize=self.figsize, title="Gather", plot_fn=self.plot_gather,
                               toolbar_position="right")

    def click(self, coords):
        # Correction for pixel center
        click_x = coords[0] + 0.5
        click_y = coords[1] + 0.5
        click_time, click_vel = self.semblance.get_time_velocity(click_y, click_x)
        if (click_time is None) or (click_vel is None):
            return None  # Ignore click

        # Redraw hodograph
        if self.hodograph is not None:
            self.hodograph.remove()
        hodograph_times = np.sqrt(click_time**2 + self.gather.offsets**2/click_vel**2)
        hodograph_y = times_to_indices(hodograph_times, self.gather.times) - 0.5  # Correction for pixel center
        hodograph_low = np.clip(hodograph_y - self.semblance.win_size, 0, len(self.gather.times) - 1)
        hodograph_high = np.clip(hodograph_y + self.semblance.win_size, 0, len(self.gather.times) - 1)
        self.hodograph = self.right.ax.fill_between(np.arange(len(hodograph_times)), hodograph_low, hodograph_high,
                                                    color="tab:blue", alpha=0.5)
        return coords

    def unclick(self):
        if self.hodograph is not None:
            self.hodograph.remove()
            self.hodograph = None
        self.right.set_title("Gather")
