import numpy as np
import cv2
from scipy.interpolate import interp1d, LinearNDInterpolator
from sklearn.neighbors import NearestNeighbors


class VelocityInterpolator:
    def __init__(self, laws):
        self.laws = laws
        self.coords = np.stack(list(laws.keys()))
        self.coords_hull = cv2.convexHull(self.coords, returnPoints=True)

        self.nearest_interpolator = NearestNeighbors(n_neighbors=1)
        self.nearest_interpolator.fit(self.coords)

        min_i, min_x = np.min(self.coords, axis=0) - 1
        max_i, max_x = np.max(self.coords, axis=0) + 1
        fake_laws = [self._interpolate_nearest(i, x)
                     for i, x in [(min_i, min_x), (min_i, max_x), (max_i, min_x), (max_i, max_x)]]
        law_data = np.concatenate([np.concatenate([[(law.inline, law.crossline)] * len(law.model), law.model], axis=1)
                                   for law in list(laws.values()) + fake_laws])
        self.linear_interpolator = LinearNDInterpolator(law_data[:, :-1], law_data[:, -1])

    def is_in_hull(self, inline, crossline):
        return cv2.pointPolygonTest(self.coords_hull, (inline, crossline), measureDist=True) >= 0

    def _interpolate_linear(self, inline, crossline):
        velocity_interpolator = lambda times: self.linear_interpolator(inline, crossline, times)
        return VelocityLaw(inline, crossline, interpolator=velocity_interpolator)

    def _interpolate_nearest(self, inline, crossline):
        index = self.nearest_interpolator.kneighbors([(inline, crossline),], return_distance=False).item()
        nearest_inline, nearest_crossline = self.coords[index].tolist()
        nearest_law = self.laws[(nearest_inline, nearest_crossline)]
        return VelocityLaw(inline, crossline, nearest_law.model, nearest_law.interpolator)

    def __call__(self, inline, crossline):
        if self.is_in_hull(inline, crossline):
            return self._interpolate_linear(inline, crossline)
        return self._interpolate_nearest(inline, crossline)


class VelocityLaw:
    def __init__(self, inline, crossline, model=None, interpolator=None):
        self.inline = inline
        self.crossline = crossline
        self.model = model
        if self.model is not None:
            self.model = np.array(model)
            self.interpolator = interp1d(self.model[:, 0], self.model[:, 1], fill_value="extrapolate")
        else:
            self.interpolator = interpolator
        if self.interpolator is None:
            raise ValueError("Both model and interpolator cannot be None")

    def __call__(self, times):
        return self.interpolator(times)


class VelocityCube:
    def __init__(self):
        self.laws = {}
        self.velocity_interpolator = None

    def update(self, law):
        self.laws[(law.inline, law.crossline)] = law
        return self

    def create_velocity_interpolator(self):
        # TODO: interpolate times to a given range
        self.velocity_interpolator = VelocityInterpolator(self.laws)
        return self

    def get_law(self, inline, crossline):
        if self.velocity_interpolator is None:
            raise ValueError("Velocity interpolator must be created first")
        return self.velocity_interpolator(inline, crossline)
