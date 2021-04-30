import numpy as np
from matplotlib import pyplot as plt

from .velocities import VelocityCube


def set_tickslabels(axis, vel_cube):
    """ convert ticklabels of an axis from grid to lateral coordinates """
    xticks = axis.get_xticks()
    yticks = axis.get_yticks()

    xticks_coords = np.zeros((len(xticks), 2))
    xticks_coords[:, 0] = xticks

    yticks_coords = np.zeros((len(yticks), 2))
    yticks_coords[:, 1] = yticks

    axis.set_xticklabels(vel_cube.from_grid_coords(xticks_coords)[:, 0])
    axis.set_yticklabels(vel_cube.from_grid_coords(yticks_coords)[:, 1])

class VelQC:

    def __init__(self):
        self._fig = None

    def plot(self, path, kernel):
        """ !!. """
        vel_cube = VelocityCube()
        vel_cube.load(path)

        errors = vel_cube.get_vel_increase_errors()

        window_radius = 2
        coords, lateral_disp, center_mape = vel_cube.get_lateral_dispersions(window_radius)
        grid_coords = vel_cube.to_grid_coords(coords)

        fig, ax = plt.subplots(1, 3, figsize=(30, 10))

        img_d = np.full(vel_cube.grid_shape, np.nan)
        img_d[grid_coords[:, 0], grid_coords[:, 1]] = lateral_disp

        cm = ax[0].imshow(img_d.T, origin='lower', cmap='Reds')
        set_tickslabels(ax[0], vel_cube)
        ax[0].set_title("Max 5x5 VL dispersion")
        plt.colorbar(cm, ax=ax[0], fraction=0.021)

        img_c = np.full(vel_cube.grid_shape, np.nan)
        img_c[grid_coords[:, 0], grid_coords[:, 1]] = center_mape

        cm = ax[1].imshow(img_c.T, origin='lower', cmap='Reds')
        set_tickslabels(ax[1], vel_cube)
        ax[1].set_title("MaxAPE center VL vs mean 5x5 VL")
        plt.colorbar(cm, ax=ax[1], fraction=0.021)

        img_e = np.full(vel_cube.grid_shape, np.nan)
        img_e[grid_coords[:, 0], grid_coords[:, 1]] = 1

        err_coords = np.asarray([line[0:2] for line in errors])
        err_points = vel_cube.to_grid_coords(err_coords)
        img_e[err_points[:, 0], err_points[:, 1]] = -1

        ax[2].imshow(img_e.T, origin='lower', cmap='RdYlGn')
        set_tickslabels(ax[2], vel_cube)
        ax[2].set_title("Velocity picking errors")

        self._fig = fig
