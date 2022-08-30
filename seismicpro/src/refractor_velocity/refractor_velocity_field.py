from functools import partial, cached_property

import numpy as np
from tqdm.auto import tqdm

from .refractor_velocity import RefractorVelocity
from .interactive_plot import FitPlot
from .utils import get_param_names, postprocess_params
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
        return get_param_names(self.n_refractors)

    @cached_property
    def is_fit(self):
        return all(item.is_fit for item in self.items)

    @cached_property
    def max_offset(self):
        max_offsets = [item.max_offset for item in self.items if item.max_offset is not None]
        if max_offsets:
            return np.mean(max_offsets)
        return None

    @cached_property
    def mean_velocity(self):
        return self.construct_item(self.values.mean(axis=0), coords=None)

    def validate_items(self, items):
        super().validate_items(items)
        n_refractors_set = {item.n_refractors for item in items}
        if self.n_refractors is not None:
            n_refractors_set.add(self.n_refractors)
        if len(n_refractors_set) != 1:
            raise ValueError("Each RefractorVelocity must describe the same number of refractors as the field")

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
        return postprocess_params(values)

    def construct_item(self, values, coords):
        return self.item_class(**dict(zip(self.param_names, values)), max_offset=self.max_offset, coords=coords)

    def _get_refined_values(self, interpolator_class, min_refractor_points=0, min_refractor_points_quantile=0):
        coords = self.coords
        values = self.values
        refined_values = np.empty_like(values)

        # Calculate the number of point in each refractor for velocity models that were fit
        n_refractor_points = np.full((self.n_items, self.n_refractors), fill_value=np.nan)
        for i, rv in enumerate(self.item_container.values()):
            if rv.is_fit:
                n_refractor_points[i] = np.histogram(rv.offsets, rv.piecewise_offsets, density=False)[0]

        # Calculate minimum acceptable number of points in each refractor, should be at least 2
        min_refractor_points = np.maximum(np.nanquantile(n_refractor_points, min_refractor_points_quantile, axis=0),
                                          max(2, min_refractor_points))
        ignore_mask = n_refractor_points < min_refractor_points

        # If a refractor is ignored for all items of a field, use it anyway
        ignore_refractors = ignore_mask.all(axis=0)
        ignore_mask[:, ignore_refractors] = False

        # Refine t0 using only items with well-fitted first refractor
        refined_values[:, 0] = interpolator_class(coords[~ignore_mask[:, 0]], values[~ignore_mask[:, 0], 0])(coords)

        # Refine crossover offsets using only items with well-fitted neighboring refractors
        for i in range(1, self.n_refractors):
            proper_items_mask = ~(ignore_mask[:, i - 1] | ignore_mask[:, i])
            refined_values[:, i] = interpolator_class(coords[proper_items_mask], values[proper_items_mask, i])(coords)

        # Refine velocities using only items with well-fitted corresponding refractor
        for i in range(self.n_refractors, 2 * self.n_refractors):
            proper_items_mask = ~ignore_mask[:, i - self.n_refractors]
            refined_values[:, i] = interpolator_class(coords[proper_items_mask], values[proper_items_mask, i])(coords)

        # Postprocess refined values
        return postprocess_params(refined_values)

    def create_interpolator(self, interpolator, min_refractor_points=0, min_refractor_points_quantile=0, **kwargs):
        """Create a field interpolator. Chooses appropriate interpolator type by its name defined by `interpolator` and
        a mapping returned by `self.available_interpolators`."""
        interpolator_class = self._get_interpolator_class(interpolator)
        values = self._get_refined_values(interpolator_class, min_refractor_points, min_refractor_points_quantile)
        self.interpolator = self._get_interpolator_class(interpolator)(self.coords, values, **kwargs)
        self.is_dirty_interpolator = False
        return self

    def smooth(self, radius=None, neighbors=10, min_refractor_points=0, min_refractor_points_quantile=0):
        if self.is_empty:
            return type(self)(survey=self.survey, is_geographic=self.is_geographic)
        if radius is None:
            radius = self.default_neighborhood_radius
        smoother = partial(IDWInterpolator, radius=radius, neighbors=neighbors, dist_transform=0)
        values = self._get_refined_values(smoother, min_refractor_points, min_refractor_points_quantile)

        smoothed_items = []
        for rv, val in zip(self.items, values):
            item = self.construct_item(val, rv.coords)

            # Copy all fit-related items from the parent field
            item.is_fit = rv.is_fit
            item.fit_result = rv.fit_result
            item.init = rv.init
            item.bounds = rv.bounds
            item.offsets = rv.offsets
            item.times = rv.times

            smoothed_items.append(item)

        return type(self)(smoothed_items, n_refractors=self.n_refractors, survey=self.survey,
                          is_geographic=self.is_geographic)

    def refine(self, radius=None, neighbors=10, min_refractor_points=0, min_refractor_points_quantile=0,
               relative_bounds_size=0.25, bar=True):
        if not self.is_fit:
            raise ValueError("Only fields that were constructed using offset-traveltime data can be refined")
        smoothed_field = self.smooth(radius, neighbors, min_refractor_points, min_refractor_points_quantile)
        bounds_size = smoothed_field.values.ptp(axis=0) * relative_bounds_size / 2
        params_bounds = np.stack([smoothed_field.values - bounds_size, smoothed_field.values + bounds_size], axis=2)

        # Clip t0 bounds to be positive and all crossover bounds to be no greater than max offset
        params_bounds[:, 0] = np.maximum(params_bounds[:, 0], 0)
        params_bounds[:, 1:self.n_refractors] = np.minimum(params_bounds[:, 1:self.n_refractors], self.max_offset)

        refined_items = []
        for rv, bounds in tqdm(zip(smoothed_field.items, params_bounds), total=self.n_items,
                               desc="Velocity models refined", disable=not bar):
            rv = RefractorVelocity.from_first_breaks(rv.offsets, rv.times, bounds=dict(zip(self.param_names, bounds)),
                                                     max_offset=self.max_offset, coords=rv.coords)
            refined_items.append(rv)
        return type(self)(refined_items, n_refractors=self.n_refractors, survey=self.survey,
                          is_geographic=self.is_geographic)

    def plot_fit(self, **kwargs):
        FitPlot(self, **kwargs).plot()
