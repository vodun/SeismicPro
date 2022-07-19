import numpy as np

from .refractor_velocity import RefractorVelocity
from ..field import SpatialField
from ..utils import to_list


class RefractorVelocityField(SpatialField):
    item_class = RefractorVelocity

    def __init__(self, items=None, n_refractors=None, survey=None, is_geographic=None):
        self.n_refractors = n_refractors
        super().__init__(items, survey, is_geographic)

    @property
    def param_names(self):
        if self.n_refractors is None:
            raise ValueError("The number of refractors is undefined")
        return ["t0"] + [f"x{i}" for i in range(1, self.n_refractors)] + [f"v{i+1}" for i in range(self.n_refractors)]

    def validate_items(self, items):
        super().validate_items(items)
        if len({item.n_refractors for item in items}) != 1:
            raise ValueError("Each RefractorVelocity instance must describe the same number of refractors")

    def update(self, items):
        items = to_list(items)
        super().update(items)
        if items:
            self.n_refractors = items[0].n_refractors
        return self

    @staticmethod
    def item_to_values(item):
        return np.array(list(item.params.values()))

    def _interpolate(self, coords):
        values = self.interpolator(coords)

        # Ensure that t0 is non-negative
        np.clip(values[:, 0], 0, None, out=values[:, 0])

        # Ensure that velocities of refractors are non-negative and increasing
        velocities = values[:, self.n_refractors:]
        np.clip(velocities[:, 0], 0, None, out=velocities[:, 0])
        np.maximum.accumulate(velocities, axis=1, out=velocities)

        # Ensure that crossover offsets are non-negative and increasing
        if self.n_refractors > 1:
            cross_offsets = values[:, 1:self.n_refractors]
            np.clip(cross_offsets[:, 0], 0, None, out=cross_offsets[:, 0])
            np.maximum.accumulate(cross_offsets, axis=1, out=cross_offsets)

        return values

    def construct_item(self, values, coords):
        return self.item_class.from_params(dict(zip(self.param_names, values)), coords=coords)
