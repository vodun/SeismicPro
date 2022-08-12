"""Weathering velocity interpolator."""

import numpy as np
from tqdm.auto import tqdm

from ..utils import CloughTocherInterpolator, Coordinates, get_cols
from .refractor_velocity_field import RefractorVelocityField, RefractorVelocity


class RefractorVelocityInterpolator():
    def __init__(self, refractor_velocity, smoothing_radius=None):
        self.field = RefractorVelocityField()
        self.refractor_velocity = refractor_velocity / 1000 # Convert m/sec to m/ms
        self.smoothing_radius = smoothing_radius

    def from_supergathers(self, supergather_survey, first_breaks_col, rv_kwargs=None):
        """Interpolate rv using supergathers"""
        # TODO: Add option to set refractor velocity as Uphole_depth / Uphole_time
        rv_kwargs = dict() if rv_kwargs is None else rv_kwargs
        rv_kwargs = {"n_refractors": 2, "init": {"t0": 0}, **rv_kwargs}
        grouped_headers = supergather_survey.headers.groupby(["SUPERGATHER_INLINE_3D", "SUPERGATHER_CROSSLINE_3D"])
        for _, sub_headers in tqdm(grouped_headers):
            rv = RefractorVelocity().from_first_breaks(offsets=sub_headers["offset"].values,
                                                      fb_times=sub_headers[first_breaks_col].values, **rv_kwargs)
            sp_coords = get_cols(sub_headers, ["SUPERGATHER_INLINE_3D", "SUPERGATHER_CROSSLINE_3D"])
            mask = np.all(sub_headers[["INLINE_3D", "CROSSLINE_3D"]].values == sp_coords, axis=1)
            rv.coords = Coordinates(coords=sub_headers[mask][["CDP_X", "CDP_Y"]].values[0], names=["CDP_X", "CDP_Y"])
            self.field.update(rv)
        if self.smoothing_radius is not None:
            self.field = self.field.smooth(self.smoothing_radius)
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
        res = np.hstack((first_crvrs, values[:, 1: -n_refractors], refractor_velocity, values[:, -n_refractors:]/1000))
        if is_1d_coords:
            return res[0]
        return res

def calculate_crossovers(v1, t1, v2, t2):
    return ((t2 - t1)*v1*v2) / (v2 - v1)
