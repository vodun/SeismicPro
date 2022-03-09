"""Building blocks for interactive plots"""

from time import time

import matplotlib.pyplot as plt

from .general_utils import align_args, MissingModule

# Safe import of modules for interactive plotting
try:
    from ipywidgets import widgets
except ImportError:
    widgets = MissingModule("ipywidgets")

try:
    from IPython.display import display
except ImportError:
    display = MissingModule("IPython.display")


MAX_CLICK_TIME = 0.2


TEXT_LAYOUT = {
    "height": "28px",
    "display": "flex",
    "width": "100%",
    "justify_content": "center",
    "align_items": "center",
}


BUTTON_LAYOUT = {
    "height": "28px",
    "width": "35px",
    "min_width": "35px",
}


TITLE_STYLE = "<style>p{word-wrap:normal; text-align:center; font-size:14px}</style>"
TITLE_TEMPLATE = "{style} <b><p>{title}</p></b>"


class InteractivePlot:
    def __init__(self, *, plot_fn=None, click_fn=None, unclick_fn=None, marker_params=None, title="",
                 init_click_coords=None, toolbar_position="left", figsize=(4.5, 4.5)):
        list_args = align_args(plot_fn, click_fn, unclick_fn, marker_params, title)
        self.plot_fn_list, self.click_fn_list, self.unclick_fn_list, marker_params_list, self.title_list = list_args
        self.marker_params_list = []
        for params in marker_params_list:
            if params is None:
                params = {}
            params = {"color": "black", "marker": "+", **params}
            self.marker_params_list.append(params)

        self.n_views = len(self.plot_fn_list)
        self.current_view = 0

        self.click_time = None
        self.click_marker = None
        self.init_click_coords = init_click_coords

        # Construct a figure
        self.toolbar_position = toolbar_position
        if toolbar_position is None:
            toolbar_position = "left"
        with plt.ioff():
            # Add tight_layout to always correctly show colorbar ticks
            self.fig, self.ax = plt.subplots(figsize=figsize, tight_layout=True)  # pylint: disable=invalid-name
        self.fig.canvas.header_visible = False
        self.fig.canvas.toolbar_visible = False
        self.fig.canvas.toolbar_position = toolbar_position

        # Setup event handlers
        self.fig.interactive_plotter = self  # Always keep reference to self for all plots to remain interactive
        self.fig.canvas.mpl_connect("resize_event", self.on_resize)
        if self.is_clickable:
            self.fig.canvas.mpl_connect("button_press_event", self.on_click)
            self.fig.canvas.mpl_connect("button_release_event", self.on_release)
            self.fig.canvas.mpl_connect("key_press_event", self.on_press)

        # Build plot box
        self.title_widget = widgets.HTML(value="", layout=widgets.Layout(**TEXT_LAYOUT))
        self.view_button = widgets.Button(icon="exchange", tooltip="Switch to the next view",
                                          layout=widgets.Layout(**BUTTON_LAYOUT))
        self.view_button.on_click(self.on_view_toggle)
        self.header = self.construct_header()
        self.toolbar = self.construct_toolbar()
        self.box = self.construct_box()

    def __del__(self):
        del self.fig.interactive_plotter
        plt.close(self.fig)

    @property
    def plot_fn(self):
        return self.plot_fn_list[self.current_view]

    @property
    def click_fn(self):
        return self.click_fn_list[self.current_view]

    @property
    def is_clickable(self):
        return self.click_fn is not None

    @property
    def unclick_fn(self):
        return self.unclick_fn_list[self.current_view]

    @property
    def is_unclickable(self):
        return self.unclick_fn is not None

    @property
    def marker_params(self):
        return self.marker_params_list[self.current_view]

    @property
    def title(self):
        title = self.title_list[self.current_view]
        if callable(title):
            return title()
        return title

    def construct_buttons(self):
        if self.n_views == 1:
            return []
        return [self.view_button]

    def construct_header(self):
        buttons = self.construct_buttons()
        if self.toolbar_position is not None:
            buttons = []
        return widgets.HBox([*buttons, self.title_widget])

    def construct_toolbar(self):
        toolbar = self.fig.canvas.toolbar
        if self.toolbar_position in {"top", "bottom"}:
            toolbar.orientation = "horizontal"
            return widgets.HBox([*self.construct_buttons(), toolbar])
        return widgets.VBox([*self.construct_buttons(), toolbar])

    def construct_box(self):
        titled_box = widgets.HBox([widgets.VBox([self.header, self.fig.canvas])])
        if self.toolbar_position == "top":
            return widgets.VBox([self.toolbar, titled_box])
        if self.toolbar_position == "bottom":
            return widgets.VBox([titled_box, self.toolbar])
        if self.toolbar_position == "left":
            return widgets.HBox([self.toolbar, titled_box])
        if self.toolbar_position == "right":
            return widgets.HBox([titled_box, self.toolbar])
        return titled_box

    def _resize(self, width):
        width += 4  # Correction for main axes margins
        self.header.layout.width = f"{int(width)}px"

    def on_resize(self, event):
        self._resize(event.width)

    def _click(self, coords):
        coords = self.click_fn(coords)
        if coords is None:  # Ignore click
            return
        if self.click_marker is not None:
            self.click_marker.remove()
        self.click_marker = self.ax.scatter(*coords, **self.marker_params, zorder=10)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        # Discard clicks outside the main axes
        if event.inaxes != self.ax:
            return
        if event.button == 1:
            self.click_time = time()

    def on_release(self, event):
        # Discard clicks outside the main axes
        if event.inaxes != self.ax:
            return
        if event.button == 1 and ((time() - self.click_time) < MAX_CLICK_TIME):
            self.click_time = None
            self._click((event.xdata, event.ydata))

    def _unclick(self):
        if self.click_marker is None:
            return
        self.unclick_fn()
        self.click_marker.remove()
        self.click_marker = None
        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if (event.inaxes != self.ax) or (event.key != "escape"):
            return
        self._unclick()

    def set_view(self, view):
        if view < 0 or view >= self.n_views:
            raise ValueError("Unknown view")
        self.current_view = view
        self.redraw()

    def on_view_toggle(self, event):
        _ = event
        if self.is_unclickable:
            self._unclick()
        self.set_view((self.current_view + 1) % self.n_views)

    def set_title(self, title=None):
        title = title or self.title
        self.title_widget.value = TITLE_TEMPLATE.format(style=TITLE_STYLE, title=title)

    def clear(self):
        # Remove all created axes except for the main one if they were created (e.g. a colorbar)
        for ax in self.fig.axes:
            if ax != self.ax:
                ax.remove()
        self.ax.clear()
        # Stretch the axes to its original size
        self.ax.set_axes_locator(None)

    def redraw(self, clear=True):
        if clear:
            self.clear()
        self.set_title()
        if self.plot_fn is not None:
            self.plot_fn(ax=self.ax)

    def plot(self, display_box=True):
        self.redraw(clear=False)
        if self.is_clickable and self.init_click_coords is not None:
            self._click(self.init_click_coords)
        # Init the width of the box
        self._resize(self.fig.get_figwidth() * self.fig.dpi / self.fig.canvas.device_pixel_ratio)
        if display_box:
            display(self.box)


class PairedPlot:
    def __init__(self, orientation="horizontal"):
        if orientation == "horizontal":
            box_type = widgets.HBox
        elif orientation == "vertical":
            box_type = widgets.VBox
        else:
            raise ValueError("Unknown plot orientation, must be one of \{'horizontal', 'vertical'\}")

        self.main = self.construct_main_plot()
        self.aux = self.construct_aux_plot()
        self.box = box_type([self.main.box, self.aux.box])

    def construct_main_plot(self):
        raise NotImplementedError

    def construct_aux_plot(self):
        raise NotImplementedError

    def plot(self):
        self.aux.plot(display_box=False)
        self.main.plot(display_box=False)
        display(self.box)
