from functools import partial

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from seismicpro.metrics import MetricMap
from seismicpro.utils import to_list, add_colorbar, get_coords_cols
from seismicpro.utils import IDWInterpolator, InteractivePlot, DropdownViewPlot, PairedPlot


class ProfilePlot(PairedPlot):
    def __init__(self, model, figsize=(4.5, 4.5), orientation="horizontal", sampling_interval=50, **kwargs):
        self.n_refractors = model.n_refractors
        self.sampling_interval = sampling_interval

        elevations = np.column_stack([model.surface_elevation_tensor.detach().cpu().numpy(),
                                      model.elevations_tensor.detach().cpu().numpy()])
        self.min_elevation = elevations.min()
        self.max_elevation = elevations.max()

        weathering_velocity = 1000 / model.weathering_slowness_tensor.detach().cpu().numpy()
        layer_velocities = 1000 / model.slownesses_tensor.detach().cpu().numpy()
        velocities = np.column_stack([weathering_velocity, layer_velocities])
        self.min_velocity = velocities.min()
        self.max_velocity = velocities.max()

        self.interpolator = IDWInterpolator(model.coords, np.column_stack([elevations, velocities]), neighbors=16)

        self.titles = (
            ["Surface elevation"] +
            [f"Elevation of layer {i+1}" for i in range(model.n_refractors)] +
            [f"Thickness of layer {i+1}" for i in range(model.n_refractors)] +
            ["Velocity of the weathering layer"] +
            [f"Velocity of layer {i+1}" for i in range(model.n_refractors)]
        )

        param_maps = [MetricMap(model.coords, data_val, coords_cols=["X", "Y"])
                      for data_val in np.column_stack([elevations, -np.diff(elevations, axis=1), velocities]).T]
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
        lim_gap = 0.05 * (self.max_elevation - self.min_elevation)
        x_start, y_start = start_coords
        x_stop, y_stop = stop_coords
        offset = np.sqrt((x_start - x_stop)**2 + (y_start - y_stop)**2)
        n_points = max(int(offset // self.sampling_interval), 2)
        x_linspace = np.linspace(x_start, x_stop, n_points)
        y_linspace = np.linspace(y_start, y_stop, n_points)
        data = self.interpolator(np.column_stack([x_linspace, y_linspace])).T
        elevations = np.row_stack([data[:self.n_refractors + 1], np.full(n_points, self.min_elevation - lim_gap)])
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
        self.aux.ax.set_ylim(self.min_elevation - lim_gap, self.max_elevation + lim_gap)

    def plot(self):
        """Display the plot and perform initial clicking."""
        super().plot()
        self.main.click(self.init_click_coords)
