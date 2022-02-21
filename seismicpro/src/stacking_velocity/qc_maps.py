from functools import partial

from matplotlib import patches

from ..metrics import MetricMap, ScatterMap, BinarizedMap, ScatterMapPlot


class StackingVelocityScatterMap(ScatterMap):
    def _plot_map(self, ax, plot_tri=False, **kwargs):
        if plot_tri:
            coords_x, coords_y = self.velocity_cube.interpolator.coords.T
            simplices = self.velocity_cube.interpolator.tri.simplices
            ax.triplot(coords_x, coords_y, simplices, color="black", linewidth=0.3, zorder=0)
        return super()._plot_map(ax, **kwargs)


class StackingVelocityBinarizedMap(BinarizedMap):
    def _plot_map(self, ax, plot_tri=False, **kwargs):
        # plot_tri argument is not applicable for a binarized map and thus ignored
        _ = plot_tri
        return super()._plot_map(ax, **kwargs)


StackingVelocityMetricMap = partial(MetricMap, scatter_map_class=StackingVelocityScatterMap,
                                    binarized_map_class=StackingVelocityBinarizedMap)


class StackingVelocityScatterMapPlot(ScatterMapPlot):
    def __init__(self, *args, plot_window=False, **kwargs):
        self.plot_window = plot_window
        self.window = None
        super().__init__(*args, **kwargs)

    def click(self, coords):
        coords = super().click(coords)
        if self.window is not None:
            self.window.remove()
        if self.metric_map.is_window_metric and self.plot_window:
            self.window = patches.Circle(coords, self.metric_map.nearest_neighbors.radius, color="blue", alpha=0.3)
            self.left.ax.add_patch(self.window)
        return coords
