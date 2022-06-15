"""Weathering velocity interpolator."""
import numpy as np
from tqdm.auto import tqdm

from ..utils import CloughTocherInterpolator


class WeatheringVelocityInterpolator():
    def __init__(self):
        self.interp = None

    def from_supergathers(self, supergather_survey, first_breaks_col, weathering_velocity=None, wv_kwargs=None):
        """Interpolate wv using supergathers"""
        # TODO: Add oppotunity to set weathering velocity as Uphole_depth / Uphole_time
        coords_to_params = {}
        weathering_velocity = weathering_velocity / 1000 # Convert m/sec to m/ms
        wv_kwargs = dict() if wv_kwargs is None else wv_kwargs
        wv_kwargs = {"n_layers": 2, "init": {"t0": 0}, **wv_kwargs}
        for ix in tqdm(supergather_survey.headers.index.unique()):
            g = supergather_survey.get_gather(ix)
            wv = g.calculate_weathering_velocity(first_breaks_col=first_breaks_col, **wv_kwargs)
            first_crvr = [calculate_crossover(weathering_velocity, 0, wv.v1 / 1000, wv.t0)]
            # Convert velocity to m/ms
            wv_params = np.array(sum([[wv[n_layer][1], wv[n_layer][2] / 1000] for n_layer in range(wv.n_layers)], []))
            wv_params = (list(wv_params[range(0, len(wv_params), 2)]) + [weathering_velocity]
                         + list(wv_params[range(1, len(wv_params), 2)]))
            wv_params = first_crvr + wv_params
            coords = g.get_central_gather()[['CDP_X', 'CDP_Y']][0]
            coords_to_params.update({tuple(coords): wv_params})

        self.interp = CloughTocherInterpolator(list(coords_to_params.keys()), list(coords_to_params.values()))
        return self

    def __call__(self, coords):
        return self.interp(coords)


def calculate_crossover(v1, t1, v2, t2):
    return ((t2 - t1)*v1*v2) / (v2 - v1)
