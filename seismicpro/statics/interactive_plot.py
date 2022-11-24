from functools import partial

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from ..metrics import MetricMap
from ..utils import to_list, add_colorbar, get_coords_cols
from ..utils import IDWInterpolator, InteractivePlot, DropdownViewPlot, PairedPlot


class ProfilePlot(PairedPlot):
    def __init__(self, model, figsize=(4.5, 4.5), orientation="horizontal", sampling_interval=50, **kwargs):
        self.n_refractors = model.n_refractors
        self.sampling_interval = sampling_interval

        unique_coords = model.field_params[["X", "Y"]].to_numpy()
        surface_elevations = model.field_params["Elevation"].to_numpy()

        layer_thicknesses = model.thicknesses_tensor.detach().cpu().numpy()
        layer_elevations = surface_elevations.reshape(-1, 1) - np.cumsum(layer_thicknesses, axis=1)
        elevations = np.column_stack([surface_elevations, layer_elevations])
        self.min_elevation = elevations.min()
        self.max_elevation = elevations.max()

        weathering_velocity = 1000 / model.weathering_slowness_tensor.detach().cpu().numpy()
        layer_velocities = 1000 / model.slownesses_tensor.detach().cpu().numpy()
        velocities = np.column_stack([weathering_velocity, layer_velocities])
        self.min_velocity = velocities.min()
        self.max_velocity = velocities.max()

        self.interpolator = IDWInterpolator(unique_coords, np.column_stack([elevations, velocities]), neighbors=16)

        self.titles = (
            ["Surface elevation"] +
            [f"Elevation of layer {i+1}" for i in range(model.n_refractors)] +
            [f"Thickness of layer {i+1}" for i in range(model.n_refractors)] +
            ["Velocity of the weathering layer"] +
            [f"Velocity of layer {i+1}" for i in range(model.n_refractors)]
        )

        param_maps = [MetricMap(unique_coords, data_val, coords_cols=["X", "Y"])
                      for data_val in np.column_stack([elevations, layer_thicknesses, velocities]).T]
        self.plot_fn = [partial(param_map._plot, title="", **kwargs) for param_map in param_maps]
        self.init_click_coords = param_maps[0].get_worst_coords()

        self.figsize = figsize
        self.orientation = orientation
        super().__init__(orientation=orientation)

    def construct_main_plot(self):
        """Construct a clickable multi-view plot of parameters of the near-surface velocity model."""
        return DropdownViewPlot(plot_fn=self.plot_fn, slice_fn=self.slice_fn, title=self.titles,
                                preserve_clicks_on_view_change=True)

    def construct_aux_plot(self):
        """Construct a plot of a velocity model at given field coordinates."""
        toolbar_position = "right" if self.orientation == "horizontal" else "left"
        return InteractivePlot(figsize=self.figsize, toolbar_position=toolbar_position)

    def slice_fn(self, start_coords, stop_coords):
        """Display a near-surface velocity model at given field coordinates."""
        x_start, y_start = start_coords
        x_stop, y_stop = stop_coords
        offset = np.sqrt((x_start - x_stop)**2 + (y_start - y_stop)**2)
        n_points = max(int(offset // self.sampling_interval), 2)
        x_linspace = np.linspace(x_start, x_stop, n_points)
        y_linspace = np.linspace(y_start, y_stop, n_points)
        data = self.interpolator(np.column_stack([x_linspace, y_linspace])).T
        elevations = np.row_stack([data[:self.n_refractors + 1], np.full(n_points, self.min_elevation)])
        velocities = data[self.n_refractors + 1:]
        self.aux.clear()
        for i in range(self.n_refractors + 1):
            polygon = self.aux.ax.fill_between(np.arange(n_points), elevations[i], elevations[i+1], lw=0, color="none")
            verts = np.vstack([p.vertices for p in polygon.get_paths()])
            extent = [verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()]
            filling = self.aux.ax.imshow(velocities[i].reshape(1, -1), cmap="coolwarm", aspect="auto",
                                         vmin=self.min_velocity, vmax=self.max_velocity, extent=extent)
            filling.set_clip_path(polygon.get_paths()[0], transform=self.aux.ax.transData)
            self.aux.ax.plot(elevations[i], color="black")
        add_colorbar(self.aux.ax, filling, True)
        self.aux.ax.set_ylim(self.min_elevation, 1.1 * self.max_elevation - 0.1 * self.min_elevation)

    def plot(self):
        """Display the plot and perform initial clicking."""
        super().plot()
        self.main.click(self.init_click_coords)


class StaticsCorrectionPlot(PairedPlot):
    def __init__(self, model, survey_list, datum=0, sort_by=None, figsize=(4.5, 4.5), orientation="horizontal", **kwargs):
        coords_cols = {get_coords_cols(survey.indexed_by for survey in survey_list)}
        if len(coords_cols) > 1:
            raise ValueError
        coords_cols = to_list(coords_cols.pop())
        self.survey_list = [survey.reindex(coords_cols) for survey in survey_list]
        self.nsm = model
        self.datum = datum
        self.sort_by = sort_by

        unique_coords = pd.concat([survey.indices.to_frame() for survey in survey_list], ignore_index=True)
        self.unique_coords = unique_coords.drop_duplicates().to_numpy()
        self.coords_neighbors = NearestNeighbors(n_neighbors=1).fit(self.unique_coords)
        self.gather = None

        self.figsize = figsize
        self.orientation = orientation
        super().__init__(orientation=orientation)

    def construct_main_plot(self):
        return InteractivePlot(plot_fn=lambda ax: ax.scatter(*self.unique_coords.T), click_fn=self.click,
                               unclick_fn=self.unclick, title="Gather locations")

    def construct_aux_plot(self):
        toolbar_position = "right" if self.orientation == "horizontal" else "left"
        return InteractivePlot(plot_fn=[self.plot_gather, partial(self.plot_gather, corrected=True)],
                               title=self.get_gather_title, figsize=self.figsize, toolbar_position=toolbar_position)

    def plot_gather(self, ax, corrected=False):
        """Plot the gather and a hodograph if click has been performed."""
        gather = self.gather
        if corrected:
            gather = self.gather.copy(ignore=["data", "samples"])
            is_uphole = self.nsm.is_uphole
            if is_uphole is None:
                loaded_headers = set(gather.headers.columns) | set(gather.headers.index.names)
                is_uphole = "SourceDepth" in loaded_headers
            shot_depths = gather["SourceDepth"] if is_uphole else 0
            shot_delays = self.nsm.estimate_delays(gather["SourceX", "SourceY"], depths=shot_depths, datum=self.datum)
            rec_delays = self.nsm.estimate_delays(gather["GroupX", "GroupY"], datum=self.datum)
            gather["DT"] = shot_delays + rec_delays
            gather = gather.apply_statics("DT")
        gather.plot(ax=ax)

    def get_gather_title(self):
        if self.aux.current_view == 0:
            return "Gather"
        return "Gather with statics correction applied"

    def click(self, coords):
        closest_ix = self.coords_neighbors.kneighbors([coords], return_distance=False).item()
        coords = tuple(self.unique_coords[closest_ix])
        survey = [survey for survey in self.survey_list if coords in survey.indices][0]
        gather = survey.get_gather(coords, copy_headers=True)
        if self.sort_by is not None:
            gather = gather.sort(by=self.sort_by)
        self.gather = gather

        self.aux.box.layout.visibility = "visible"
        self.aux.redraw()
        return coords

    def unclick(self):
        """Remove highlighted shot or receiver locations and hide the gather plot."""
        self.aux.clear()
        self.aux.box.layout.visibility = "hidden"
