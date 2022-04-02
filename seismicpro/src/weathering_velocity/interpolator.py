"""Weathering velocity interpolator."""
import numpy as np
from tqdm.auto import tqdm
from sklearn.neighbors import NearestNeighbors


class WeatheringVelocityInterpolator():
    def __init__(self):
        self.wv_params = None
        self.nn = None

    def from_supergathers(self, survey, supergather_kwargs, first_breaks_col, weathering_velocity=None,
                          wv_kwargs=None):
        """Interpolate wv using supergathers"""
        # TODO: Add oppotunity to set weathering velocity as Uphole_depth / Uphole_time

        # Move source on the surface, if wv is not passed to find correct value of V0.
        # All operations with passed weathering_velocity are applied under the assumption that weathering_velocity
        # is a direct wave.
        supergather_survey = survey.generate_supergathers(**supergather_kwargs)
        if weathering_velocity is None:
            supergather_survey.apply(lambda FirstBreak, SourceUpholeTime: np.clip(FirstBreak - SourceUpholeTime, 1, None),
                                     cols=[first_breaks_col, "SourceUpholeTime"],
                                     res_cols=first_breaks_col,
                                     inplace=True,
                                     unpack_args=True)

        coords_to_params = {}
        wv_kwargs = dict() if wv_kwargs is None else wv_kwargs
        wv_kwargs = {"n_layers": 2, "init": {"t0": 0}, **wv_kwargs}
        for ix in tqdm(supergather_survey.headers.index.unique()):
            g = supergather_survey.get_gather(ix)
            wv = g.calculate_weathering_velocity(first_breaks_col=first_breaks_col, **wv_kwargs)
            wv_params = [getattr(wv, param) for param in wv._valid_keys[1:]]
            if weathering_velocity is not None:
                wv_params.insert(0, calculate_crossover(weathering_velocity, 0, wv.v1, wv.t0))
                wv_params.insert(len(wv._valid_keys) - wv_kwargs.get("n_layers"), weathering_velocity)

            coords = survey.headers.loc[ix][["CDP_X", "CDP_Y"]].values[0]
            coords_to_params.update({tuple(coords): wv_params})

        self.wv_params = np.array(list(coords_to_params.values()))
        self.nn = NearestNeighbors(n_neighbors=1).fit(list(coords_to_params.keys()))
        return self

    def __call__(self, coords):
        ixs = self.nn.kneighbors(coords, return_distance=False).reshape(-1)
        return self.wv_params[ixs]


def calculate_crossover(v1, t1, v2, t2):
    return ((t2 - t1)*v1*v2) / (v2 - v1)
