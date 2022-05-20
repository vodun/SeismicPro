"""Implements the geometry quality control metrics."""

import numpy as np

from ..metrics import PipelineMetric, pass_calc_args
from ..const import HDR_FIRST_BREAK


class GeomControlMetric(PipelineMetric):
    views = ("plot_geom_control", "plot_wv", "plot_crop")
    name = "geom_control"
    is_lower_better = True
    vmin = 0
    vmax = 40

    @classmethod
    def calc(cls, gather, geom_control, **kwargs):
        return geom_control.difference
    
    @pass_calc_args
    def plot_geom_control(cls, gather, geom_control, ax, **kwargs):
        geom_control.plot(ax=ax)

    @pass_calc_args
    def plot_wv(cls, gather, geom_control, ax, **kwargs):
        wv = gather.calculate_weathering_velocity(n_layers=1)
        wv.plot(threshold_times=30, ax=ax, **kwargs)

    @pass_calc_args
    def plot_crop(cls, gather, geom_control, ax, **kwargs):
        gather = gather.copy()
        gather.sort(by='offset')
        wv = gather.calculate_weathering_velocity(n_layers=1)
        gather.apply_lmo(wv)
        cropped = gather.crop(origins=(0,0), crop_shape=(100,100), n_crops=1)
        data = cropped.crops[0].T
        vmax = np.nanquantile(data, .9)
        vmin = np.nanquantile(data, .1)
        ax.imshow(cropped.crops[0].T, vmin=vmin, vmax=vmax, cmap='gray', aspect='auto')
