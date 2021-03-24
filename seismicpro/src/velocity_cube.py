import numpy as np
from scipy.interpolate import interp1d, LinearNDInterpolator
from sklearn.neighbors import NearestNeighbors


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


class NearestInterpolator:
    def __init__(self, laws):
        self.laws = laws
        self.coords = np.stack(list(laws.keys()))
        self.nn = NearestNeighbors(n_neighbors=1, n_jobs=-1)
        self.nn.fit(self.coords)

    def __call__(self, inline, crossline):
        index = self.nn.kneighbors([(inline, crossline),], return_distance=False).item()
        nn_inline, nn_crossline = self.coords[index].tolist()
        nn_interpolator = self.laws[(nn_inline, nn_crossline)]
        return VelocityLaw(nn_inline, nn_crossline, interpolator=nn_interpolator)


class LinearInterpolator:
    def __init__(self, laws):
        law_data = np.concatenate([np.concatenate([[(law.inline, law.crossline)] * len(law.model), law.model], axis=1)
                                   for law in laws.values()])
        self.interpolator = LinearNDInterpolator(law_data[:, :-1], law_data[:, -1])

    def is_in_hull(self, inline, crossline):
        res = self.interpolator(inline, crossline, 0).item()
        return not np.isnan(res)

    def __call__(self, inline, crossline):
        return VelocityLaw(inline, crossline, interpolator=lambda times: self.interpolator(inline, crossline, times))


class VelocityCube:
    def __init__(self):
        self.laws = {}
        self.linear_interpolator = None
        self.nearest_interpolator = None

    def update(self, law):
        self.laws[(law.inline, law.crossline)] = law
        return self

    def create_interpolator(self):
        # TODO: interpolate times to a given range
        self.linear_interpolator = LinearInterpolator(self.laws)
        self.nearest_interpolator = NearestInterpolator(self.laws)
        return self

    def get_law(self, inline, crossline):
        if self.linear_interpolator is None:
            raise ValueError("Interpolator must be created first")
        if self.linear_interpolator.is_in_hull(inline, crossline):
            return self.linear_interpolator(inline, crossline)
        return self.nearest_interpolator(inline, crossline)
