"""Implements a class for interactive plotting of survey geometry"""

from functools import partial

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from ..utils import set_ticks, set_text_formatting, get_text_formatting_kwargs
from ..utils.interactive_plot_utils import InteractivePlot, PairedPlot


class SurveyGeometryPlot(PairedPlot):  # pylint: disable=too-many-instance-attributes
    """Interactive survey geometry plot.

    The plot provides 2 views:
    * Shot view: displays shot locations. Highlights all activated receivers on click and displays the corresponding
      common shot gather.
    * Receiver view: displays receiver locations. Highlights all shots that activated the receiver on click and
      displays the corresponding common receiver gather.
    """
    def __init__(self, survey, keep_aspect=False, sort_by=None, gather_plot_kwargs=None, x_ticker=None, y_ticker=None,
                 figsize=(4.5, 4.5), fontsize=8, orientation="horizontal", **kwargs):
        kwargs = {"fontsize": fontsize, **kwargs}
        (x_ticker, y_ticker), self.scatter_kwargs = set_text_formatting(x_ticker, y_ticker, **kwargs)
        text_kwargs = get_text_formatting_kwargs(**kwargs)
        if gather_plot_kwargs is None:
            gather_plot_kwargs = {}
        self.gather_plot_kwargs = {"title": None, **text_kwargs, **gather_plot_kwargs}

        self.survey = survey
        self.sort_by = sort_by
        self.figsize = figsize
        self.orientation = orientation

        # Calculate source and group indices to speed up gather selection and shot/receiver nearest neighbors to
        # project a click on the closest one
        source_params = self._process_survey(survey, ["SourceX", "SourceY"])
        self.source_ix, self.source_x, self.source_y, self.source_neighbors = source_params
        group_params = self._process_survey(survey, ["GroupX", "GroupY"])
        self.group_ix, self.group_x, self.group_y, self.group_neighbors = group_params

        # Calculate axes limits to fix them to avoid map plot shifting on view toggle
        x_lim = self._get_limits(self.source_x, self.group_x)
        y_lim = self._get_limits(self.source_y, self.group_y)
        self.plot_map = partial(self._plot_map, keep_aspect=keep_aspect, x_lim=x_lim, y_lim=y_lim, x_ticker=x_ticker,
                                y_ticker=y_ticker, **self.scatter_kwargs)
        self.affected_scatter = None

        super().__init__(orientation=orientation)

    def construct_main_plot(self):
        """Construct an interactive plot of shot and receiver locations."""
        return InteractivePlot(plot_fn=[self.plot_map, self.plot_map], click_fn=self.click, unclick_fn=self.unclick,
                               title=["Shot map", "Receiver map"])

    def construct_aux_plot(self):
        """Construct an interactive gather plot."""
        toolbar_position = "right" if self.orientation == "horizontal" else "left"
        aux_plot = InteractivePlot(figsize=self.figsize, toolbar_position=toolbar_position)
        aux_plot.box.layout.visibility = "hidden"
        return aux_plot

    @staticmethod
    def _process_survey(survey, coord_cols):
        # pylint: disable=import-outside-toplevel
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
        if np.isclose(margin, 0):
            margin = 0.05 * max_coord
        return [min_coord - margin, max_coord + margin]

    @property
    def is_shot_view(self):
        """bool: whether the current view displays shot locations."""
        return self.main.current_view == 0

    @property
    def index(self):
        """SeismicIndex: an index to get gathers from, depends on the current view."""
        return self.source_ix if self.is_shot_view else self.group_ix

    @property
    def coord_x(self):
        """np.ndarray: x coordinates of shots or receivers, depend on the current view."""
        return self.source_x if self.is_shot_view else self.group_x

    @property
    def coord_y(self):
        """np.ndarray: y coordinates of shots or receivers, depend on the current view."""
        return self.source_y if self.is_shot_view else self.group_y

    @property
    def coords_neighbors(self):
        """sklearn.neighbors.NearestNeighbors: nearest neighbors of shots or receivers, depend on the current view."""
        return self.source_neighbors if self.is_shot_view else self.group_neighbors

    @property
    def affected_coords_cols(self):
        """list of 2 str: coordinates columns, describing highlighted objects, depend on the current view."""
        return ["GroupX", "GroupY"] if self.is_shot_view else ["SourceX", "SourceY"]

    @property
    def main_color(self):
        """str: color of the plotted objects, depends on the current view."""
        return "tab:red" if self.is_shot_view else "tab:blue"

    @property
    def main_marker(self):
        """str: marker of the plotted objects, depends on the current view."""
        return "*" if self.is_shot_view else "v"

    @property
    def aux_color(self):
        """str: color of the highlighted objects, depends on the current view."""
        return "tab:blue" if self.is_shot_view else "tab:red"

    @property
    def aux_marker(self):
        """str: marker of the highlighted objects, depends on the current view."""
        return "v" if self.is_shot_view else "*"

    @property
    def map_x_label(self):
        """str: label of the x map axis, depends on the current view."""
        return "SourceX" if self.is_shot_view else "GroupX"

    @property
    def map_y_label(self):
        """str: label of the y map axis, depends on the current view."""
        return "SourceY" if self.is_shot_view else "GroupY"

    @property
    def gather_title(self):
        """str: map plot title, depends on the current view."""
        return "Common shot gather at " if self.is_shot_view else "Common receiver gather at "

    def _plot_map(self, ax, keep_aspect, x_lim, y_lim, x_ticker, y_ticker, **kwargs):
        """Plot shot or receiver locations depending on the current view."""
        self.aux.clear()
        self.aux.box.layout.visibility = "hidden"

        ax.scatter(self.coord_x, self.coord_y, color=self.main_color, marker=self.main_marker, **kwargs)
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.ticklabel_format(style="plain", useOffset=False)
        if keep_aspect:
            ax.set_aspect("equal", adjustable="box")
        set_ticks(ax, "x", self.map_x_label, **x_ticker)
        set_ticks(ax, "y", self.map_y_label, **y_ticker)

    def click(self, coords):
        """Highlight affected shot or receiver locations and display a gather."""
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
        self.affected_scatter = self.main.ax.scatter(*gather[self.affected_coords_cols].T, color=self.aux_color,
                                                     marker=self.aux_marker, **self.scatter_kwargs)

        self.aux.box.layout.visibility = "visible"
        self.aux.set_title(self.gather_title + f"{x, y}")
        self.aux.clear()
        gather.plot(ax=self.aux.ax, **self.gather_plot_kwargs)
        return x, y

    def unclick(self):
        """Remove highlighted shot or receiver locations and hide the gather plot."""
        if self.affected_scatter is not None:
            self.affected_scatter.remove()
            self.affected_scatter = None
        self.aux.clear()
        self.aux.box.layout.visibility = "hidden"
