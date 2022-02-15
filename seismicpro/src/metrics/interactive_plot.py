from functools import partial

from sklearn.neighbors import NearestNeighbors

from ..utils import set_text_formatting, MissingModule
from ..utils.interactive_plot_utils import InteractivePlot, ClickablePlot, PairedPlot, TEXT_LAYOUT, BUTTON_LAYOUT

# Safe import of modules for interactive plotting
try:
    from ipywidgets import widgets
except ImportError:
    widgets = MissingModule("ipywidgets")

try:
    from IPython.display import display
except ImportError:
    display = MissingModule("IPython.display")


class MapBinPlot(InteractivePlot):
    def __init__(self, *args, options=None, is_lower_better=True, **kwargs):
        self.is_desc = is_lower_better
        self.options = None
        self.curr_option = None

        self.sort = widgets.Button(icon=self.sort_icon, disabled=True, layout=widgets.Layout(**BUTTON_LAYOUT))
        self.prev = widgets.Button(icon="angle-left", disabled=True, layout=widgets.Layout(**BUTTON_LAYOUT))
        self.drop = widgets.Dropdown(layout=widgets.Layout(**TEXT_LAYOUT))
        self.next = widgets.Button(icon="angle-right", disabled=True, layout=widgets.Layout(**BUTTON_LAYOUT))

        # Handler definition
        self.sort.on_click(self.reverse_coords)
        self.prev.on_click(self.prev_coords)
        self.drop.observe(self.select_coords, names="value")
        self.next.on_click(self.next_coords)

        super().__init__(*args, **kwargs)
        if options is not None:
            self.update_state(0, options)

    def create_header(self):
        if self.fig.canvas.toolbar_position == "right":
            return widgets.HBox([self.prev, self.drop, self.next, self.sort])
        return widgets.HBox([self.sort, self.prev, self.drop, self.next])

    @property
    def sort_icon(self):
        return "sort-amount-desc" if self.is_desc else "sort-amount-asc"

    @property
    def drop_options(self):
        return [f"{metric:.05f} metric at ({x}, {y})" for (x, y), metric in self.options.iteritems()]

    def update_state(self, option_ix, options=None, redraw=True):
        new_options = self.options if options is None else options
        if (new_options is None) or (option_ix < 0) or (option_ix >= len(new_options)):
            return
        self.options = new_options
        self.curr_option = option_ix

        # Unobserve dropdown widget to simultaneously update both options and the currently selected option
        self.drop.unobserve(self.select_coords, names="value")
        with self.drop.hold_sync():
            self.drop.options = self.drop_options
            self.drop.index = self.curr_option
        self.drop.observe(self.select_coords, names="value")

        self.sort.disabled = False
        self.prev.disabled = (self.curr_option == 0)
        self.next.disabled = (self.curr_option == (len(self.options) - 1))

        if redraw:
            self._plot()

    def reverse_coords(self, event):
        _ = event
        self.is_desc = not self.is_desc
        self.sort.icon = self.sort_icon
        self.update_state(len(self.options) - self.curr_option - 1, self.options.iloc[::-1], redraw=False)

    def next_coords(self, event):
        _ = event
        self.update_state(min(self.curr_option + 1, len(self.options) - 1))

    def prev_coords(self, event):
        _ = event
        self.update_state(max(self.curr_option - 1, 0))

    def select_coords(self, change):
        _ = change
        self.update_state(self.drop.index)

    def _plot(self):
        if self.plot_fn is not None:
            self.ax.clear()
            self.plot_fn(self.options.index[self.curr_option], ax=self.ax)

    def plot(self, display_box=True):
        self._resize(self.fig.get_figwidth() * self.fig.dpi)  # Init the width of the box
        self._plot()
        if display_box:
            display(self.box)


class MetricMapPlot(PairedPlot):
    def __init__(self, metric_map, plot_on_click, title=None, x_ticker=None, y_ticker=None, is_lower_better=None,
                 figsize=(4.5, 4.5), fontsize=8, plot_on_click_kwargs=None, **kwargs):
        (x_ticker, y_ticker), kwargs = set_text_formatting(x_ticker, y_ticker, fontsize=fontsize, **kwargs)
        if plot_on_click_kwargs is None:
            plot_on_click_kwargs = {}
        plot_on_click_kwargs["x_ticker"] = {**x_ticker, **plot_on_click_kwargs.get("x_ticker", {})}
        plot_on_click_kwargs["y_ticker"] = {**y_ticker, **plot_on_click_kwargs.get("y_ticker", {})}

        self.metric_map = metric_map
        self.figsize = figsize
        self.title = metric_map.plot_title if title is None else title
        self.plot_map = partial(metric_map.plot, title="", x_ticker=x_ticker, y_ticker=y_ticker,
                                is_lower_better=is_lower_better, **kwargs)
        self.plot_on_click = partial(plot_on_click, **plot_on_click_kwargs)
        self.init_click_coords = metric_map.get_worst_coords(is_lower_better)
        super().__init__()

    def construct_left_plot(self):
        return ClickablePlot(figsize=self.figsize, plot_fn=self.plot_map, click_fn=self.click, allow_unclick=False,
                             title=self.title, init_click_coords=self.init_click_coords)

    def click():
        raise NotImplementedError


class ScatterMapPlot(MetricMapPlot):
    def __init__(self, metric_map, plot_on_click, **kwargs):
        self.coords = metric_map.map_data.index.to_frame().values
        self.coords_neighbors = NearestNeighbors(n_neighbors=1).fit(self.coords)
        super().__init__(metric_map, plot_on_click, **kwargs)

    def construct_right_plot(self):
        return InteractivePlot(toolbar_position="right")

    def click(self, coords):
        coords_ix = self.coords_neighbors.kneighbors([coords], return_distance=False).item()
        coords = tuple(self.coords[coords_ix])
        self.right.set_title(f"{self.metric_map.map_data[coords]:.05f} metric at {coords}")
        self.right.ax.clear()
        self.plot_on_click(coords, ax=self.right.ax)
        return coords


class BinarizedMapPlot(MetricMapPlot):
    def construct_right_plot(self):
        return MapBinPlot(plot_fn=self.plot_on_click, toolbar_position="right")

    def click(self, coords):
        bin_coords = (int(coords[0] + 0.5), int(coords[1] + 0.5))
        contents = self.metric_map.get_bin_contents(bin_coords)
        if contents is None:  # Handle clicks outside bins
            return None
        self.right.update_state(0, contents)
        return bin_coords
