from functools import partial
from sklearn.neighbors import NearestNeighbors

from ..metrics import MetricMap
from ..utils.interactive_plot_utils import InteractivePlot, DropdownViewPlot, PairedPlot


class FitPlot(PairedPlot):
    def __init__(self, field, x_ticker=None, y_ticker=None, figsize=(4.5, 4.5), fontsize=8, orientation="horizontal",
                 **kwargs):
        self.field = field
        self.coords = self.field.coords
        self.values = self.field.values
        self.coords_neighbors = NearestNeighbors(n_neighbors=1).fit(self.coords)

        coords_cols = self.field.coords_cols
        if coords_cols is None:
            coords_cols = ["X", "Y"] if self.field.is_geographic else ["INLINE_3D", "CROSSLINE_3D"]
        param_maps = [MetricMap(self.coords, col, coords_cols=coords_cols) for col in self.values.T]
        self.plot_fn = [partial(param_map._plot, title="") for param_map in param_maps]
        self.init_click_coords = param_maps[0].get_worst_coords()

        self.figsize = figsize
        self.orientation = orientation
        super().__init__(orientation=orientation)

    def construct_main_plot(self):
        return DropdownViewPlot(plot_fn=self.plot_fn, click_fn=self.click, title=self.field.param_names,
                                preserve_clicks_on_view_change=True)

    def construct_aux_plot(self):
        """Construct a gather plot."""
        toolbar_position = "right" if self.orientation == "horizontal" else "left"
        return InteractivePlot(figsize=self.figsize, toolbar_position=toolbar_position)

    def click(self, coords):
        closest_ix = self.coords_neighbors.kneighbors([coords], return_distance=False).item()
        coords = tuple(self.coords[closest_ix])
        rv = self.field.item_container[coords]
        self.aux.set_title(f"Refractor velocity at {int(coords[0]), int(coords[1])}")
        self.aux.clear()
        rv.plot(ax=self.aux.ax)
        return coords

    def plot(self):
        super().plot()
        self.main.click(self.init_click_coords)
