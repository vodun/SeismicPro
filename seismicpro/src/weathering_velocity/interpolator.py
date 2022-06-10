"""Weathering velocity interpolator."""
import numpy as np
from tqdm.auto import tqdm

from ..utils import DelaunayInterpolator


class WeatheringVelocityInterpolator():
    def __init__(self):
        self.interp = None

    def from_supergathers(self, supergather_survey, first_breaks_col, weathering_velocity=None, wv_kwargs=None):
        """Interpolate wv using supergathers"""
        # TODO: Add oppotunity to set weathering velocity as Uphole_depth / Uphole_time

        coords_to_params = {}
        wv_kwargs = dict() if wv_kwargs is None else wv_kwargs
        wv_kwargs = {"n_layers": 2, "init": {"t0": 0}, **wv_kwargs}
        for ix in tqdm(supergather_survey.headers.index.unique()):
            g = supergather_survey.get_gather(ix)
            wv = g.calculate_weathering_velocity(first_breaks_col=first_breaks_col, **wv_kwargs)
            wv_params = [calculate_crossover(weathering_velocity / 1000, 0, wv.v1 / 1000, wv.t0)]
            # Convert velocity to m/ms
            wv_params += sum([[wv[n_layer][1], wv[n_layer][2] / 1000] for n_layer in range(wv.n_layers)], [])
            coords = g.get_central_gather()[['CDP_X', 'CDP_Y']][0]
            coords_to_params.update({tuple(coords): wv_params})

        self.interp = DelaunayInterpolator(list(coords_to_params.keys()), list(coords_to_params.values()))
        return self

    def __call__(self, coords):
        return self.interp(coords)


def calculate_crossover(v1, t1, v2, t2):
    return ((t2 - t1)*v1*v2) / (v2 - v1)
