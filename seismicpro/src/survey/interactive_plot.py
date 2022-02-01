import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from ipywidgets import widgets
from IPython.display import display

from ..utils import set_ticks, set_text_formatting
from ..utils.interactive_plot_utils import InteractivePlot, ToggleClickablePlot


class SurveyPlot:
    def __init__(self, survey, sort_by=None, x_ticker=None, y_ticker=None, figsize=(4.5, 4.5), **kwargs):
        self.survey = survey
        self.source_ix, self.source_x, self.source_y, self.source_knn = self._process_survey(survey, ["SourceX", "SourceY"])
        self.group_ix, self.group_x, self.group_y, self.group_knn = self._process_survey(survey, ["GroupX", "GroupY"])
        self.is_shot_view = True
        self.sort_by = sort_by
        self.affected_scatter = None
        (self.x_ticker, self.y_ticker), kwargs = set_text_formatting(x_ticker, y_ticker, **kwargs)

        self.left = ToggleClickablePlot(figsize=figsize, plot_fn=self.plot_map, click_fn=self.click,
                                        unclick_fn=self.unclick, toggle_fn=self.toggle_view,
                                        toggle_icon=self.toggle_icon)
        self.left.ax.ticklabel_format(style="plain", useOffset=False)
        self.right = InteractivePlot(figsize=figsize, toolbar_position="right")
        self.box = widgets.HBox([self.left.box, self.right.box])

    @staticmethod
    def _process_survey(survey, coord_cols):
        from ..index import SeismicIndex
        index = SeismicIndex(surveys=survey.reindex(coord_cols))
        coords = np.stack(index.indices.values)[:, 1:]
        knn = NearestNeighbors(n_neighbors=1).fit(coords)
        return index, coords[:, 0], coords[:, 1], knn

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
    def knn(self):
        return self.source_knn if self.is_shot_view else self.group_knn
    
    @property
    def affected_coords_cols(self):
        return ["GroupX", "GroupY"] if self.is_shot_view else ["SourceX", "SourceY"]

    @property
    def main_color(self):
        return "red" if self.is_shot_view else "blue"

    @property
    def aux_color(self):
        return "blue" if self.is_shot_view else "red"

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

    def plot_map(self, ax):
        self.left.set_title(self.map_title)
        ax.scatter(self.coord_x, self.coord_y, color=self.main_color)
        set_ticks(ax, "x", self.map_x_label, **self.x_ticker)
        set_ticks(ax, "y", self.map_y_label, **self.y_ticker)

    def click(self, coords):
        closest_ix = self.knn.kneighbors([coords], return_distance=False).item()
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
        self.affected_scatter = self.left.ax.scatter(*gather[self.affected_coords_cols].T, color=self.aux_color)

        self.right.ax.clear()
        gather.plot(ax=self.right.ax, x_ticker=self.x_ticker, y_ticker=self.y_ticker)
        self.right.set_title(self.gather_title + f"{x, y}")
        self.right.box.layout.visibility = "visible"
        return x, y

    def unclick(self):
        if self.affected_scatter is not None:
            self.affected_scatter.remove()
            self.affected_scatter = None
        self.right.ax.clear()
        self.right.box.layout.visibility = "hidden"

    def toggle_view(self, event):
        self.is_shot_view = not self.is_shot_view
        self.left.ax.clear()
        self.left.ax.ticklabel_format(style="plain", useOffset=False)
        self.right.ax.clear()
        self.right.box.layout.visibility = "hidden"
        self.plot_map(ax=self.left.ax)
        self.left.button.icon = self.toggle_icon

    def plot(self):
        display(self.box)
        self.left.plot(display_box=False)
        self.right.plot(display_box=False)
        self.right.box.layout.visibility = "hidden"
