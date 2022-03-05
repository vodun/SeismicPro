from functools import partial

import numpy as np

from ..stacking_velocity import StackingVelocity
from ..utils import set_text_formatting, times_to_indices, MissingModule
from ..utils.interactive_plot_utils import InteractivePlot, PairedPlot

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
        self.gather_plot_kwargs = gather_plot_kwargs

        self.figsize = figsize
        self.title = title
        self.click_time = None
        self.click_vel = None

        self.semblance = semblance
        self.gather = self.semblance.gather.copy(ignore="data")
        self.plot_semblance = partial(self.semblance.plot, *args, title=None, x_ticker=x_ticker, y_ticker=y_ticker,
                                      **kwargs)

        super().__init__()
        if sharey:
            self.aux.ax.sharey(self.main.ax)

    def construct_main_plot(self):
        return InteractivePlot(plot_fn=self.plot_semblance, click_fn=self.click, unclick_fn=self.unclick,
                               title=self.title, figsize=self.figsize)

    def construct_aux_plot(self):
        plotter = InteractivePlot(plot_fn=[self.plot_gather, partial(self.plot_gather, corrected=True)],
                                  title=self.get_gather_title, figsize=self.figsize, toolbar_position="right")
        plotter.view_button.disabled = True
        return plotter

    def get_gather_title(self):
        if (self.click_time is None) or (self.click_vel is None):
            return "Gather"
        return f"Hodograph from {self.click_time:.0f} ms with {self.click_vel:.2f} km/s velocity"

    def get_gather(self, corrected):
        if not corrected:
            return self.gather
        velocity = StackingVelocity.from_constant_velocity(self.click_vel * 1000)
        return self.gather.copy(ignore=["headers", "data", "samples"]).apply_nmo(velocity)

    def get_hodograph(self, corrected):
        if (self.click_time is None) or (self.click_vel is None):
            return None
        if not corrected:
            return np.sqrt(self.click_time**2 + self.gather.offsets**2/self.click_vel**2)
        return np.full_like(self.gather.offsets, self.click_time)

    def plot_gather(self, ax, corrected=False):
        gather = self.get_gather(corrected=corrected)
        gather.plot(ax=ax, **self.gather_plot_kwargs)

        hodograph = self.get_hodograph(corrected=corrected)
        if hodograph is None:
            return
        hodograph_y = times_to_indices(hodograph, self.gather.times) - 0.5  # Correction for pixel center
        hodograph_low = np.clip(hodograph_y - self.semblance.win_size, 0, len(self.gather.times) - 1)
        hodograph_high = np.clip(hodograph_y + self.semblance.win_size, 0, len(self.gather.times) - 1)
        ax.fill_between(np.arange(len(hodograph)), hodograph_low, hodograph_high, color="tab:blue", alpha=0.5)

    def click(self, coords):
        click_time, click_vel = self.semblance.get_time_velocity(coords[1] + 0.5, coords[0] + 0.5)
        if (click_time is None) or (click_vel is None):
            return None  # Ignore click
        self.aux.view_button.disabled = False
        self.click_time = click_time
        self.click_vel = click_vel
        self.aux.redraw()
        return coords

    def unclick(self):
        self.click_time = None
        self.click_vel = None
        self.aux.set_view(0)
        self.aux.view_button.disabled = True
