from functools import partial

import pandas as pd
from sklearn.neighbors import NearestNeighbors

from ..utils import set_ticks, set_text_formatting, MissingModule
from ..utils.interactive_plot_utils import InteractivePlot, ToggleClickablePlot, PairedPlot

# Safe import of modules for interactive plotting
try:
    from ipywidgets import widgets
except ImportError:
    widgets = MissingModule("ipywidgets")

try:
    from IPython.display import display
except ImportError:
    display = MissingModule("IPython.display")


class SurveyGeometryPlot(PairedPlot):
    def __init__(self, survey, sort_by=None, keep_aspect=False, x_ticker=None, y_ticker=None, figsize=(4.5, 4.5),
                 fontsize=8, gather_plot_kwargs=None, **kwargs):
        (x_ticker, y_ticker), self.map_kwargs = set_text_formatting(x_ticker, y_ticker, fontsize=fontsize, **kwargs)
        if gather_plot_kwargs is None:
            gather_plot_kwargs = {}
        gather_plot_kwargs["x_ticker"] = {**x_ticker, **gather_plot_kwargs.get("x_ticker", {})}
        gather_plot_kwargs["y_ticker"] = {**y_ticker, **gather_plot_kwargs.get("y_ticker", {})}
        self.gather_plot_kwargs = gather_plot_kwargs

        self.survey = survey
        self.sort_by = sort_by
        self.figsize = figsize

        # Calculate source and group indices to speed up gather selection and shot/receiver nearest neighbours to
        # project a click on the closest one
        source_params = self._process_survey(survey, ["SourceX", "SourceY"])
        self.source_ix, self.source_x, self.source_y, self.source_neighbors = source_params
        group_params = self._process_survey(survey, ["GroupX", "GroupY"])
        self.group_ix, self.group_x, self.group_y, self.group_neighbors = group_params

        # Calculate axes limits to fix them to avoid map plot shifting on view toggle
        x_lim = self._get_limits(self.source_x, self.group_x)
        y_lim = self._get_limits(self.source_y, self.group_y)
        self.plot_map = partial(self._plot_map, keep_aspect=keep_aspect, x_lim=x_lim, y_lim=y_lim, x_ticker=x_ticker,
                                y_ticker=y_ticker, **self.map_kwargs)

        self.is_shot_view = True
        self.affected_scatter = None

        super().__init__()

    def construct_left_plot(self):
        return ToggleClickablePlot(figsize=self.figsize, plot_fn=self.plot_map, click_fn=self.click,
                                   unclick_fn=self.unclick, toggle_fn=self.toggle_view, toggle_icon=self.toggle_icon)

    def construct_right_plot(self):
        right = InteractivePlot(figsize=self.figsize, toolbar_position="right")
        right.box.layout.visibility = "hidden"
        return right

    @staticmethod
    def _process_survey(survey, coord_cols):
        from ..index import SeismicIndex  # Avoid cyclic imports, remove when Survey.get_gather is optimized
        index = SeismicIndex(surveys=survey.reindex(coord_cols))
        coords = index.indices.to_frame().values[:, 1:]
        coords_neighbors = NearestNeighbors(n_neighbors=1).fit(coords)
        return index, coords[:, 0], coords[:, 1], coords_neighbors

    @staticmethod
    def _get_limits(source_coords, group_coords):
        min_coord = min(source_coords.min(), group_coords.min())
        max_coord = max(source_coords.max(), group_coords.max())
        margin = 0.05 * (max_coord - min_coord)
        return [min_coord - margin, max_coord + margin]

    def _plot_map(self, ax, keep_aspect, x_lim, y_lim, x_ticker, y_ticker, **kwargs):
        self.left.set_title(self.map_title)
        ax.scatter(self.coord_x, self.coord_y, color=self.main_color, **kwargs)
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.ticklabel_format(style="plain", useOffset=False)
        if keep_aspect:
            ax.set_aspect("equal", adjustable="box")
        set_ticks(ax, "x", self.map_x_label, **x_ticker)
        set_ticks(ax, "y", self.map_y_label, **y_ticker)

    @property
    def index(self):
        return self.source_ix if self.is_shot_view else self.group_ix

    @property
    def coord_x(self):
        return self.source_x if self.is_shot_view else self.group_x

    @property
    def coord_y(self):
        return self.source_y if self.is_shot_view else self.group_y

    @property
    def coords_neighbors(self):
        return self.source_neighbors if self.is_shot_view else self.group_neighbors

    @property
    def affected_coords_cols(self):
        return ["GroupX", "GroupY"] if self.is_shot_view else ["SourceX", "SourceY"]

    @property
    def main_color(self):
        return "tab:red" if self.is_shot_view else "tab:blue"

    @property
    def aux_color(self):
        return "tab:blue" if self.is_shot_view else "tab:red"

    @property
    def toggle_icon(self):
        return "chevron-up" if self.is_shot_view else "chevron-down"

    @property
    def map_title(self):
        return "Shot map" if self.is_shot_view else "Receiver map"

    @property
    def map_x_label(self):
        return "SourceX" if self.is_shot_view else "GroupX"

    @property
    def map_y_label(self):
        return "SourceY" if self.is_shot_view else "GroupY"

    @property
    def gather_title(self):
        return "Common shot gather at " if self.is_shot_view else "Common receiver gather at "

    def click(self, coords):
        closest_ix = self.coords_neighbors.kneighbors([coords], return_distance=False).item()
        x = self.coord_x[closest_ix]
        y = self.coord_y[closest_ix]

        # TODO: Change to gather = survey.get_gather((x, y)) when it is optimized
        tmp_index = self.index.create_subset(pd.MultiIndex.from_tuples([(0, x, y)]))
        gather_headers = tmp_index.headers.droplevel(0)
        gather = self.survey.load_gather(gather_headers, copy_headers=False)

        if self.sort_by is not None:
            gather = gather.sort(by=self.sort_by)
        if self.affected_scatter is not None:
            self.affected_scatter.remove()
        self.affected_scatter = self.left.ax.scatter(*gather[self.affected_coords_cols].T, color=self.aux_color,
                                                     **self.map_kwargs)

        self.right.box.layout.visibility = "visible"
        self.right.set_title(self.gather_title + f"{x, y}")
        self.right.ax.clear()
        gather.plot(ax=self.right.ax, **self.gather_plot_kwargs)
        return x, y

    def unclick(self):
        if self.affected_scatter is not None:
            self.affected_scatter.remove()
            self.affected_scatter = None
        self.right.ax.clear()
        self.right.box.layout.visibility = "hidden"

    def toggle_view(self, event):
        _ = event
        self.is_shot_view = not self.is_shot_view
        self.left.button.icon = self.toggle_icon
        self.left.ax.clear()
        self.plot_map(ax=self.left.ax)
        self.right.ax.clear()
        self.right.box.layout.visibility = "hidden"
