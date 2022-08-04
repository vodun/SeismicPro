from functools import partial

import numpy as np

from .refractor_velocity import RefractorVelocity
from ..field import SpatialField
from ..utils import to_list, IDWInterpolator


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

    @staticmethod
    def _postprocess_values(self, values):
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

    def _interpolate(self, coords):
        values = self.interpolator(coords)
        self._postprocess_values(values)
        return values

    def construct_item(self, values, coords):
        return self.item_class.from_params(dict(zip(self.param_names, values)), coords=coords)

    def smooth(self, radius, min_refractor_points=10):
        coords = self.coords
        values = self.values
        smoothed_values = np.empty_like(values)
        smoother = partial(IDWInterpolator, radius=radius, dist_transform=0)

        ignore_mask = np.zeros((self.n_items, self.n_refractors), dtype=bool)
        for i, rv in enumerate(self.item_container.values()):
            if rv.offsets is not None:
                n_refractor_points = np.histogram(rv.offsets, rv.piecewise_offsets, density=False)[0]
                ignore_mask[i] = n_refractor_points < min_refractor_points

        # If a refractor is empty for all items of a field, smooth its values anyway
        ignore_refractors = ignore_mask.all(axis=0)
        ignore_mask[:, ignore_refractors] = False

        # Smooth t0 using only items with well-fitted first refractor
        smoothed_values[:, 0] = smoother(coords[~ignore_mask[:, 0]], values[~ignore_mask[:, 0], 0])(coords)

        # Smooth crossover offsets using only items with well-fitted neighboring refractors
        for i in range(1, self.n_refractors):
            proper_items_mask = ~(ignore_mask[:, i - 1] | ignore_mask[:, i])
            smoothed_values[:, i] = smoother(coords[proper_items_mask], values[proper_items_mask, i])(coords)

        # Smooth velocities using only items with well-fitted corresponding refractor
        for i in range(self.n_refractors, 2 * self.n_refractors):
            proper_items_mask = ~ignore_mask[:, i - self.n_refractors]
            smoothed_values[:, i] = smoother(coords[proper_items_mask], values[proper_items_mask, i])(coords)

        self._postprocess_values(smoothed_values)

        smoothed_items = []
        for rv, val in zip(field.item_container.values(), smoothed_values):
            item = self.construct_item(val, rv.coords)
            item.offsets = rv.offsets
            item.fb_times = rv.fb_times
            smoothed_items.append(item)
        
        return type(self)(smoothed_items, n_refractors=self.n_refractors, survey=self.survey,
                          is_geographic=self.is_geographic)
