from functools import partial
from sklearn.neighbors import NearestNeighbors

from ..metrics import MetricMap
from ..utils import get_text_formatting_kwargs
from ..utils.interactive_plot_utils import InteractivePlot, DropdownViewPlot, PairedPlot


class FitPlot(PairedPlot):
    def __init__(self, field, refractor_velocity_plot_kwargs=None, figsize=(4.5, 4.5), fontsize=8,
                 orientation="horizontal", **kwargs):
        if field.is_empty:
            raise ValueError("Empty fields do not support interactive plotting")
        if not field.is_fit:
            raise ValueError("Only fields constructed from offset-traveltime data can be plotted")

        kwargs = {"fontsize": fontsize, **kwargs}
        text_kwargs = get_text_formatting_kwargs(**kwargs)
        if refractor_velocity_plot_kwargs is None:
            refractor_velocity_plot_kwargs = {}
        self.refractor_velocity_plot_kwargs = {"title": None, **text_kwargs, **refractor_velocity_plot_kwargs}

        self.field = field
        self.coords = self.field.coords
        self.values = self.field.values
        self.coords_neighbors = NearestNeighbors(n_neighbors=1).fit(self.coords)

        self.x_lim = [0, max(rv.piecewise_offsets[-1] for rv in field.items) * 1.1]
        self.y_lim = [0, max(rv.piecewise_times[-1] for rv in field.items) * 1.1]
        self.titles = (
            ["t0 - Intercept time"] +
            [f"x{i} - Crossover offset {i}" for i in range(1, field.n_refractors)] +
            [f"v{i} - Velocity of refractor {i}" for i in range(1, field.n_refractors + 1)]
        )

        coords_cols = self.field.coords_cols
        if coords_cols is None:
            coords_cols = ["X", "Y"] if self.field.is_geographic else ["INLINE_3D", "CROSSLINE_3D"]
        param_maps = [MetricMap(self.coords, col, coords_cols=coords_cols) for col in self.values.T]
        self.plot_fn = [partial(param_map._plot, title="", **kwargs) for param_map in param_maps]
        self.init_click_coords = param_maps[0].get_worst_coords()

        self.figsize = figsize
        self.orientation = orientation
        super().__init__(orientation=orientation)

    def construct_main_plot(self):
        return DropdownViewPlot(plot_fn=self.plot_fn, click_fn=self.click, title=self.titles,
                                preserve_clicks_on_view_change=True)

    def construct_aux_plot(self):
        toolbar_position = "right" if self.orientation == "horizontal" else "left"
        return InteractivePlot(figsize=self.figsize, toolbar_position=toolbar_position)

    def click(self, coords):
        closest_ix = self.coords_neighbors.kneighbors([coords], return_distance=False).item()
        coords = tuple(self.coords[closest_ix])
        self.aux.set_title(f"Refractor velocity at {int(coords[0]), int(coords[1])}")
        self.aux.clear()
        rv = self.field.item_container[coords]
        rv.plot(ax=self.aux.ax, **self.refractor_velocity_plot_kwargs)
        self.aux.set_xlim(*self.x_lim)
        self.aux.set_ylim(*self.y_lim)
        return coords

    def plot(self):
        super().plot()
        self.main.click(self.init_click_coords)
