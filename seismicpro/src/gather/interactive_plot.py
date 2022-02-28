from functools import partial

from ..stacking_velocity import StackingVelocity
from ..utils.interactive_plot_utils import InteractivePlot
from ..utils import MissingModule

# Safe import of modules for interactive plotting
try:
    from ipywidgets import widgets
except ImportError:
    widgets = MissingModule("ipywidgets")


class SlidingVelocityPlot(InteractivePlot):
    def __init__(self, *, slider_min,  slider_max, slide_fn=None, **kwargs):
        self.slider = widgets.FloatSlider(min=slider_min, max=slider_max, step=1, readout=False,
                                          layout=widgets.Layout(width="80%"))
        self.slider.observe(slide_fn, "value")
        slider_box = [widgets.HTML(value=str(slider_min)), self.slider, widgets.HTML(value=str(slider_max))]
        self.slider_box = widgets.HBox(slider_box, layout=widgets.Layout(justify_content="center"))
        super().__init__(**kwargs)

    def construct_header(self):
        header = super().construct_header()
        return widgets.VBox([header, self.slider_box])

    def on_view_toggle(self, event):
        super().on_view_toggle(event)
        if self.current_view == self.n_views - 1:
            self.slider_box.layout.visibility = "hidden"
        else:
            self.slider_box.layout.visibility = "visible"


class InteractiveCorrection:
    def __init__(self, gather, min_vel, max_vel, figsize, **kwargs):
        kwargs = {"fontsize": 8, **kwargs}
        self.gather = gather
        self.plotter = SlidingVelocityPlot(plot_fn=[partial(self.plot_corrected_gather, **kwargs),
                                                    partial(self.gather.plot, **kwargs)],
                                           slide_fn=self.on_velocity_change, slider_min=min_vel, slider_max=max_vel,
                                           title=[self.get_title, "Source gather"], figsize=figsize)

    def get_title(self):
        raise NotImplementedError

    @property
    def corrected_gather(self):
        raise NotImplementedError

    def plot_corrected_gather(self, ax, **kwargs):
        self.corrected_gather.plot(ax=ax, **kwargs)

    def on_velocity_change(self, change):
        _ = change
        self.plotter.redraw()

    def plot(self):
        self.plotter.plot()


class InteractiveNMOCorrection(InteractiveCorrection):
    def get_title(self):
        return f"Normal moveout correction with {self.plotter.slider.value:.0f} m/s velocity"

    @property
    def corrected_gather(self):
        new_vel = StackingVelocity.from_constant_velocity(self.plotter.slider.value)
        return self.gather.copy(ignore=["headers", "data", "samples"]).apply_nmo(new_vel)
