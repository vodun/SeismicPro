#!/usr/bin/env python3
""" Script for validating velocities. Ensures velocity increases with time """

import argparse
import csv

import numpy as np
from matplotlib import pyplot as plt

from .velocities import VelocityCube


def process_vel_file():
    """ parse script arguments and run `parse_vel_file`"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--inp_path', type=str, help="Path to velocities file.",
                        required=True)
    parser.add_argument('-v', '--vel_err_path', type=str, help="Path to file with errors in velocities.",
                        default='vel_res.csv')
    parser.add_argument('-m', '--vel_err_map_path', type=str, help="Path to save map of errors.",
                        default='velocities_err_map.png')

    args = parser.parse_args()
    inp_path = args.inp_path
    vel_err_path = args.vel_err_path
    vel_err_map_path = args.vel_err_map_path

    _process_vel_file(inp_path, vel_err_path, vel_err_map_path)

def _process_vel_file(inp_path, vel_err_path, vel_err_map_path):
    """ parse input file and dump found errors """
    print(f"Processing {inp_path}...")

    vc = VelocityCube()

    vc.load(inp_path)

    vel_errors = vc.get_vel_increase_errors()

    _dump_vel_errors_csv(vel_errors, vel_err_path)
    _dump_vel_errors_img(vel_errors, vc, vel_err_map_path)


def _dump_vel_errors_csv(errors, fname):
    """ Dump found errors in velocities to file """

    if len(errors) == 0:
        print("Velocities OK. No res file written")
        return

    with open(fname, 'w', newline='') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(['CoordX', 'CoordY', 'Time_prev', 'Vel_prev', 'Time_curr', 'Vel_curr'])
        for line in errors:
            csvwriter.writerow(line)


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


def _dump_vel_errors_img(errors, vel_cube, imgname):
    """ Draw points from velocities file as a map """

    window_radius = 2
    coords, lateral_disp, center_mape = vel_cube.get_lateral_dispersions(window_radius)
    grid_coords = vel_cube.to_grid_coords(coords)

    _, ax = plt.subplots(1, 3, figsize=(30, 10))

    img_d = np.full(vel_cube.grid_shape, np.nan)
    img_d[grid_coords[:, 0], grid_coords[:, 1]] = lateral_disp

    cm = ax[0].imshow(img_d.T, origin='lower', cmap='Reds')
    set_tickslabels(ax[0], vel_cube)
    ax[0].set_title("Max 5x5 velocity laws dispersion")
    plt.colorbar(cm, ax=ax[0])

    img_c = np.full(vel_cube.grid_shape, np.nan)
    img_c[grid_coords[:, 0], grid_coords[:, 1]] = center_mape

    cm = ax[1].imshow(img_c.T, origin='lower', cmap='Reds')
    set_tickslabels(ax[1], vel_cube)
    ax[1].set_title("Center velocity law vs mean 5x5 velocity law Max APE")
    plt.colorbar(cm, ax=ax[1])

    img_e = np.full(vel_cube.grid_shape, np.nan)
    img_e[grid_coords[:, 0], grid_coords[:, 1]] = 1

    err_coords = np.asarray([line[0:2] for line in errors])
    err_points = vel_cube.to_grid_coords(err_coords)
    img_e[err_points[:, 0], err_points[:, 1]] = -1

    ax[2].imshow(img_e.T, origin='lower', cmap='RdYlGn')
    set_tickslabels(ax[2], vel_cube)
    ax[2].set_title("Velocity picking errors")

    plt.savefig(imgname)



if __name__ == "__main__":
    process_vel_file()
