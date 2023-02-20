"""Implements classes for interactive metric map plotting"""

from functools import partial

from sklearn.neighbors import NearestNeighbors

from ..utils import to_list, get_first_defined, get_text_formatting_kwargs, align_args
from ..utils.interactive_plot_utils import InteractivePlot, DropdownOptionPlot, PairedPlot


class NonOverlayingIndicesPlot(InteractivePlot):
    """Construct an interactive plot that passes the last click coordinates to a `plot_fn` of each of its views in
    addition to `ax`."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_coords = None
        self.current_index = None
        self.current_title = None

    @property
    def title(self):
        """str: Return the title of the map data plot."""
        return self.current_title

    @property
    def plot_fn(self):
        """callable: plotter of the current view with the last click coordinates passed."""
        if self.current_coords is None or self.current_index is None:
            return None
        return partial(super().plot_fn, coords=self.current_coords, index=self.current_index)

    def process_map_click(self, coords, indices, titles):
        self.current_coords = coords[0]
        self.current_index = indices[0]
        self.current_title = titles[0]
        self.redraw()


class OverlayingIndicesPlot(DropdownOptionPlot):
    def process_map_click(self, coords, indices, titles):
        options = [{"coords": coord, "index": index, "option_title": title}
                   for coord, index, title in zip(coords, indices, titles)]
        self.update_state(0, options)


class MetricMapPlot(PairedPlot):  # pylint: disable=abstract-method
    """Base class for interactive metric map visualization.

    Two methods should be redefined in a concrete plotter child class:
    * `construct_aux_plot` - construct an interactive plot with map contents at click location,
    * `click` - handle a click on the map plot.
    """
    def __init__(self, metric_map, plot_on_click=None, plot_on_click_kwargs=None, title=None, is_lower_better=None,
                 figsize=(4.5, 4.5), fontsize=8, orientation="horizontal", **kwargs):
        kwargs = {"fontsize": fontsize, **kwargs}
        text_kwargs = get_text_formatting_kwargs(**kwargs)
        plot_on_click, plot_on_click_kwargs = align_args(plot_on_click, plot_on_click_kwargs)
        plot_on_click_kwargs = [{} if plot_kwargs is None else plot_kwargs for plot_kwargs in plot_on_click_kwargs]
        plot_on_click_kwargs = [{**text_kwargs, **plot_kwargs} for plot_kwargs in plot_on_click_kwargs]

        self.figsize = figsize
        self.orientation = orientation

        self.metric_map = metric_map
        self.is_lower_better = is_lower_better
        self.title = metric_map.plot_title if title is None else title
        self.plot_map = partial(metric_map.plot, title="", is_lower_better=is_lower_better, **kwargs)
        self.plot_on_click = [partial(plot_fn, **plot_kwargs)
                              for plot_fn, plot_kwargs in zip(plot_on_click, plot_on_click_kwargs)]
        super().__init__(orientation=orientation)

    def construct_main_plot(self):
        """Construct the metric map plot."""
        return InteractivePlot(plot_fn=self.plot_map, click_fn=self.click, title=self.title, figsize=self.figsize)

    def construct_aux_titles(self, coords, indices, metric_values):
        index_cols = self.metric_map.index_cols
        coords_cols = self.metric_map.coords_cols
        data_cols = index_cols + coords_cols
        keep_cols = [True] * len(index_cols) + [col not in index_cols for col in coords_cols]
        ix_coord_str = [", ".join(f"{col} {val}" for val, col, keep in zip(ix + coord, data_cols, keep_cols) if keep)
                        for ix, coord in zip(indices, coords)]
        return [f"{metric:.03f} metric for {ix_coord}" for metric, ix_coord in zip(metric_values, ix_coord_str)]

    def preprocess_click_coords(self, click_coords):
        _ = click_coords
        raise NotImplementedError

    def click(self, click_coords):
        """Handle a click on the map plot."""
        click_coords = self.preprocess_click_coords(click_coords)
        if click_coords is None:
            return None
        is_ascending = not get_first_defined(self.is_lower_better, self.metric_map.metric.is_lower_better, True)
        click_indices = self.metric_map.get_indices_by_map_coords(click_coords).sort_values(ascending=is_ascending)
        indices, metric_values = zip(*click_indices.items())
        coords = [self.metric_map.get_coords_by_index(index) for index in indices]
        titles = self.construct_aux_titles(coords, indices, metric_values)
        self.aux.process_map_click(coords, indices, titles)
        return click_coords

    def plot(self):
        """Display the map and perform initial clicking."""
        super().plot()
        self.main.click(self.metric_map.get_worst_coords(self.is_lower_better))


class ScatterMapPlot(MetricMapPlot):
    """Construct an interactive plot of a non-aggregated metric map."""

    def __init__(self, metric_map, plot_on_click, **kwargs):
        self.coords = metric_map.map_data.index.to_frame(index=False).to_numpy()
        self.coords_neighbors = NearestNeighbors(n_neighbors=1).fit(self.coords)
        super().__init__(metric_map, plot_on_click, **kwargs)

    def construct_aux_plot(self):
        """Construct an interactive plot with data representation at click location."""
        toolbar_position = "right" if self.orientation == "horizontal" else "left"
        if self.metric_map.has_overlaying_indices:
            return OverlayingIndicesPlot(plot_fn=self.plot_on_click, toolbar_position=toolbar_position)
        return NonOverlayingIndicesPlot(plot_fn=self.plot_on_click, toolbar_position=toolbar_position)

    def preprocess_click_coords(self, click_coords):
        """Return map coordinates closest to coordinates of the click."""
        return tuple(self.coords[self.coords_neighbors.kneighbors([click_coords], return_distance=False).item()])


class BinarizedMapPlot(MetricMapPlot):
    """Construct an interactive plot of a binarized metric map."""

    def construct_aux_plot(self):
        """Construct an interactive plot with map contents at click location."""
        toolbar_position = "right" if self.orientation == "horizontal" else "left"
        return OverlayingIndicesPlot(plot_fn=self.plot_on_click, toolbar_position=toolbar_position)

    def preprocess_click_coords(self, click_coords):
        """Return coordinates of a bin corresponding to click coords. Ignore the click if it was performed outside the
        map."""
        coords = (int(click_coords[0] + 0.5), int(click_coords[1] + 0.5))
        return coords if coords in self.metric_map.map_data else None
