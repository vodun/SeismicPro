from time import time

import matplotlib.pyplot as plt
from ipywidgets import widgets


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
    def __init__(self, title="", plot_fn=None, figsize=(4.5, 4.5), toolbar_position="left"):
        toolbar_visible = (toolbar_position is not None)
        if toolbar_position is None:
            toolbar_position = "left"

        with plt.ioff():
            self.fig, self.ax = plt.subplots(figsize=figsize, constrained_layout=True)

        self.fig.canvas.header_visible = False
        self.fig.canvas.toolbar_visible = toolbar_visible
        self.fig.canvas.toolbar_position = toolbar_position
        self.fig.canvas.mpl_connect("resize_event", self.on_resize)

        title = TITLE_TEMPLATE.format(style=TITLE_STYLE, title=title)
        self.title = widgets.HTML(value=title, layout=widgets.Layout(**TEXT_LAYOUT))

        self.header = self.create_header()
        self.box = widgets.VBox([self.header, self.fig.canvas])

        self.plot_fn = plot_fn

    def create_header(self):
        placeholder = widgets.HTML(layout=widgets.Layout(**BUTTON_LAYOUT))
        if self.fig.canvas.toolbar_position == "left":
            return widgets.HBox([placeholder, self.title])
        if self.fig.canvas.toolbar_position == "right":
            return widgets.HBox([self.title, placeholder])
        return self.title

    def _resize(self, width):
        width = width / self.fig.canvas._dpi_ratio  # Remove when fixed in ipympl
        width += 4  # Correction for main axes margins
        if self.fig.canvas.toolbar_visible and (self.fig.canvas.toolbar_position in {"left", "right"}):
            width += 44  #  Correction for an optional toolbar and its margins
        self.header.layout.width = f"{int(width)}px"

    def on_resize(self, event):
        self._resize(event.width)

    def set_title(self, title):
        self.title.value = TITLE_TEMPLATE.format(style=TITLE_STYLE, title=title)

    def plot(self, display=True):
        if display:
            display(self.box)
        self._resize(self.fig.get_figwidth() * self.fig.dpi)  # Init the width of the box
        if self.plot_fn is not None:
            self.plot_fn(ax=self.ax)


class ClickablePlot(InteractivePlot):
    def __init__(self, *args, click_fn=None, allow_unclick=True, unclick_fn=None, init_x=None, init_y=None,
                 color="black", marker="+", **kwargs):
        super().__init__(*args, **kwargs)
        self.click_fn = click_fn
        self.unclick_fn = unclick_fn
        self.color = color
        self.marker = marker
        self.click_time = None
        self.click_scatter = None
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        if allow_unclick:
            self.fig.canvas.mpl_connect("key_press_event", self.on_press)
        if (init_x is not None) and (init_y is not None):
            self._click(init_x, init_y)

    def _click(self, x, y):
        if self.click_fn is not None:
            x, y = self.click_fn(x, y)
        if self.click_scatter is not None:
            self.click_scatter.remove()
        self.click_scatter = self.ax.scatter(x, y, color=self.color, marker=self.marker)
        self.fig.canvas.draw()

    def on_click(self, event):
        # Discard clicks outside the main axes
        if not event.inaxes == self.ax:
            return
        if event.button == 1:
            self.click_time = time()

    def on_release(self, event):
        if not event.inaxes == self.ax:
            return
        if event.button == 1 and ((time() - self.click_time) < MAX_CLICK_TIME):
            self._click(event.xdata, event.ydata)

    def on_press(self, event):
        if (event.inaxes != self.ax) or (event.key != "escape"):
            return
        if self.click_scatter is not None:
            if self.unclick_fn is not None:
                self.unclick_fn()
            self.click_scatter.remove()
            self.click_scatter = None
            self.fig.canvas.draw()


class ToggleClickablePlot(ClickablePlot):
    def __init__(self, *args, toggle_fn=None, toggle_icon=None, **kwargs):
        self.button = widgets.Button(icon=toggle_icon, layout=widgets.Layout(**BUTTON_LAYOUT))
        self.button.on_click(toggle_fn)
        super().__init__(*args, **kwargs)

    def create_header(self):
        placeholder = widgets.HTML(layout=widgets.Layout(**BUTTON_LAYOUT))
        if self.fig.canvas.toolbar_position == "right":
            return widgets.HBox([self.button, placeholder])
        return widgets.HBox([self.button, self.title])


class OptionPlot(InteractivePlot):
    def __init__(self, *args, options, is_lower_better=True, **kwargs):
        self.is_desc = is_lower_better
        self.options = None
        self.curr_option = None

        text_layout = widgets.Layout(height="28px", display="flex", width="100%", justify_content="center", align_items="center")
        button_layout = widgets.Layout(height="28px", width="35px", min_width="35px")

        self.sort = widgets.Button(icon=self.sort_icon, layout=button_layout)
        self.prev = widgets.Button(icon="angle-left", layout=button_layout)
        self.drop = widgets.Dropdown(layout=text_layout)
        self.next = widgets.Button(icon="angle-right", layout=button_layout)

        # Handler definition
        self.sort.on_click(self.reverse_coords)
        self.prev.on_click(self.prev_coords)
        self.drop.observe(self.select_coords, names="value")
        self.next.on_click(self.next_coords)

        super().__init__(*args, **kwargs)

        self.update_state(0, options)

    def create_header(self):
        return widgets.HBox([self.sort, self.prev, self.drop, self.next])

    @property
    def sort_icon(self):
        return "sort-amount-desc" if self.is_desc else "sort-amount-asc"

    def redraw(self):
        self.ax.clear()
        self.plot_fn(*self.options.index[self.curr_option], ax=self.ax)

    def gen_drop_options(self, options):
        return [f"({x}, {y}) - {metric:.05f}" for (x, y), metric in options.iteritems()]

    def update_drop(self, option_ix, options=None):
        self.drop.unobserve(self.select_coords, names="value")
        with self.drop.hold_sync():
            if options is not None:
                self.drop.options = self.gen_drop_options(options)
            self.drop.index = option_ix
        self.drop.observe(self.select_coords, names="value")

    def update_state(self, option_ix, options=None, redraw=True):
        self.curr_option = option_ix
        if options is not None:
            self.options = options
        self.update_drop(option_ix, options)
        self.toggle_prev_next_buttons()
        if redraw:
            self.redraw()
    
    def reverse_coords(self, event):
        self.is_desc = not self.is_desc
        self.sort.icon = self.sort_icon
        self.update_state(len(self.options) - self.curr_option - 1, self.options.iloc[::-1], redraw=False)

    def toggle_prev_next_buttons(self):
        self.prev.disabled = (self.curr_option == 0)
        self.next.disabled = (self.curr_option == (len(self.options) - 1))

    def next_coords(self, event):
        self.update_state(min(self.curr_option + 1, len(self.options) - 1))

    def prev_coords(self, event):
        self.update_state(max(self.curr_option - 1, 0))

    def select_coords(self, change):
        self.update_state(self.drop.index)
