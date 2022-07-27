"""Weathering velocity interpolator."""

import numpy as np
from tqdm.auto import tqdm

from ..utils import CloughTocherInterpolator, Coordinates
from .refractor_velocity_field import RefractorVelocityField


class RefractorVelocityInterpolator():
    def __init__(self, refractor_velocity):
        self.field = RefractorVelocityField()
        self.refractor_velocity = refractor_velocity / 1000 # Convert m/sec to m/ms

    def from_supergathers(self, supergather_survey, first_breaks_col, wv_kwargs=None):
        """Interpolate wv using supergathers"""
        # TODO: Add option to set refractor velocity as Uphole_depth / Uphole_time
        wv_kwargs = dict() if wv_kwargs is None else wv_kwargs
        wv_kwargs = {"n_refractors": 2, "init": {"t0": 0}, **wv_kwargs}
        for ix in tqdm(supergather_survey.headers.index.unique()):
            g = supergather_survey.get_gather(ix)
            rv = g.calculate_refractor_velocity(first_breaks_col=first_breaks_col, **wv_kwargs)
            rv.coords = Coordinates(coords=g.get_central_gather()[["CDP_X", "CDP_Y"]][0], names=["CDP_X", "CDP_Y"])
            self.field.update(rv)
        self.field.create_interpolator('ct')
        return self

    def __call__(self, coords):
        coords = np.array(coords, dtype=np.float32)
        is_1d_coords = coords.ndim == 1
        coords = np.atleast_2d(coords)
        values = self.field.interpolate(coords)
        n_refractors = self.field.n_refractors
        t0 = values[:, 0]
        v1 = values[:, -n_refractors] / 1000
        first_crvrs = np.array([calculate_crossovers(self.refractor_velocity, 0, v1, t0)]).reshape(-1, 1)
        refractor_velocity = np.zeros((len(coords), 1)) + self.refractor_velocity
        res = np.hstack((first_crvrs, values[:, 1: -n_refractors], refractor_velocity, values[:, -n_refractors:]))
        if is_1d_coords:
            return res[0]
        return res

def calculate_crossovers(v1, t1, v2, t2):
    return ((t2 - t1)*v1*v2) / (v2 - v1)
