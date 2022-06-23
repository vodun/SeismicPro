"""Base class for interactive plotter of depths slices."""

import numpy as np
import matplotlib as mpl

from seismicpro.src.utils import to_list, add_colorbar
from seismicpro.src.utils.interactive_plot_utils import InteractivePlot, PairedPlot

class StaticsMapPlot(InteractivePlot):
    def __init__(self, *, plot_fn=None, drag_fn=None, title="", toolbar_position="left", figsize=(4.5, 4.5)):
        super().__init__(plot_fn=plot_fn, title=title, toolbar_position=toolbar_position, figsize=figsize)
        self.drag_fn = drag_fn
        self.line_marker = None
        self.start_click_coords = None
        self.is_clicked = False

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("key_press_event", self.on_press)

    def on_click(self, event):
        """Remember the mouse button click time to further distinguish between mouse click and hold events."""
        # Discard clicks outside the main axes
        if event.inaxes != self.ax:
            return
        if event.button == 1:
            self.is_clicked = True
            self.start_click_coords = (event.xdata, event.ydata)

    def _update_line(self, end_coords):
        if self.line_marker is not None:
            self.line_marker[0].remove()
        self.line_marker = self.ax.plot([self.start_click_coords[0], end_coords[0]],
                                        [self.start_click_coords[1], end_coords[1]],
                                        color="black", zorder=10)
        self.fig.canvas.draw_idle()

    def on_motion(self, event):
        # Discard clicks outside the main axes
        if event.inaxes != self.ax:
            return
        if event.button == 1 and self.is_clicked:
            self._update_line((event.xdata, event.ydata))

    def on_release(self, event):
        """Handle the mouse button click event if it was short enough to consider it as a single click."""
        # Discard clicks outside the main axes
        if event.inaxes != self.ax:
            return
        if event.button == 1:
            self.is_clicked = False
            end_coords = (event.xdata, event.ydata)
            self._update_line(end_coords)
            if self.drag_fn is not None:
                self.drag_fn(self.start_click_coords, end_coords)

    def _unclick(self):
        if self.line_marker is None:
            return
        self.line_marker[0].remove()
        self.line_marker = None
        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if (event.inaxes == self.ax) and (event.key == "escape"):
            self._unclick()


class StaticsPlot(PairedPlot):
    def __init__(self, mmap, layers_elevations, elevations, velocities, n_points=100, vmin=None, vmax=None):
        self.mmap = mmap
        self.layers_elevations = to_list(layers_elevations)
        self.elevations = elevations
        self.velocities = velocities
        self.n_points = n_points
        self.vmin = vmin
        self.vmax = vmax
        super().__init__()

    def construct_main_plot(self):
        """Construct a clickable semblance plot."""
        return StaticsMapPlot(plot_fn=self.mmap.plot, drag_fn=self.drag)

    def construct_aux_plot(self):
        """Construct a correctable gather plot."""
        return InteractivePlot()

    def drag(self, start_coords, end_coords):
        self.aux.clear()
        start_coords = np.array(list(start_coords)).astype(np.int32)
        end_coords = np.array(list(end_coords)).astype(np.int32)
        xs = np.linspace(start_coords[0], end_coords[0], self.n_points)
        ys = np.linspace(start_coords[1], end_coords[1], self.n_points)

        coords = np.vstack((xs, ys)).T
        elevations = self.elevations(coords)
        velocities = self.velocities(coords).T
        depths = [layer(coords) for layer in self.layers_elevations]
        ax = self.aux.ax

        slice_img = np.full((int(max(elevations))+10, self.n_points), fill_value=np.nan)
        for vel, depth in zip(velocities, [elevations, *depths]):
            ix_matrix = np.arange(slice_img.shape[0]).repeat(slice_img.shape[1]).reshape(slice_img.shape)
            curr_mask = (ix_matrix - depth < 0) * vel
            slice_img = np.where(curr_mask, curr_mask, slice_img)
            ax.plot(depth, "-", c='k', linewidth=5)
        ax.set_ylim(20, max(elevations)+10)
        img = ax.imshow(slice_img, cmap='jet', aspect='auto', vmin=self.vmin, vmax=self.vmax)
        add_colorbar(ax, img, True)
