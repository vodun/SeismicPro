"""Weathering velocity interpolator."""

import numpy as np
from tqdm.auto import tqdm

from ..utils import CloughTocherInterpolator


class RefractorVelocityInterpolator():
    def __init__(self):
        self.interp = None

    def from_supergathers(self, supergather_survey, first_breaks_col, refractor_velocity=None, wv_kwargs=None):
        """Interpolate wv using supergathers"""
        # TODO: Add oppotunity to set refractor velocity as Uphole_depth / Uphole_time
        coords_to_params = {}
        refractor_velocity = refractor_velocity / 1000 # Convert m/sec to m/ms
        wv_kwargs = dict() if wv_kwargs is None else wv_kwargs
        wv_kwargs = {"n_refractors": 2, "init": {"t0": 0}, **wv_kwargs}
        for ix in tqdm(supergather_survey.headers.index.unique()):
            g = supergather_survey.get_gather(ix)
            wv = g.calculate_refractor_velocity(first_breaks_col=first_breaks_col, **wv_kwargs)
            first_crvr = [calculate_crossover(refractor_velocity, 0, wv.v1 / 1000, wv.t0)]
            # Convert velocity to m/ms
            wv_params = list(wv.params.values())[1:]
            velocities = [refractor_velocity] + list(np.array(wv_params[-wv.n_refractors:]) / 1000)
            wv_params = first_crvr + wv_params[:-wv.n_refractors] + velocities
            coords = g.get_central_gather()[['CDP_X', 'CDP_Y']][0]
            coords_to_params.update({tuple(coords): wv_params})

        self.interp = CloughTocherInterpolator(list(coords_to_params.keys()), list(coords_to_params.values()))
        return self

    def __call__(self, coords):
        return self.interp(coords)


def calculate_crossover(v1, t1, v2, t2):
    return ((t2 - t1)*v1*v2) / (v2 - v1)
