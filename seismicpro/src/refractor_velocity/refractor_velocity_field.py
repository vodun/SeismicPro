from functools import partial

import numpy as np
import pandas as pd

from .refractor_velocity import RefractorVelocity
from .interactive_plot import FitPlot
from .utils import get_param_names, postprocess_params, calc_df_to_dump, load_rv, dump_rv
from ..field import SpatialField
from ..utils import to_list, Coordinates, IDWInterpolator


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

    @property
    def max_offset(self):
        return np.max([item.max_offset for coords, item in self.item_container.items()])

    def validate_items(self, items):
        super().validate_items(items)
        n_refractors_set = {item.n_refractors for item in items}
        if self.n_refractors is not None:
            n_refractors_set.add(self.n_refractors)
        if len(n_refractors_set) != 1:
            raise ValueError("Each RefractorVelocity must describe the same number of refractors as the field")

    @classmethod
    def from_file(cls, path, is_geographic=None, encoding="UTF-8"):
        """Load RefractorVelocityField from a file.

        File should have coords and parameters of a single RefractorVelocity with next structure:
         - The first row contains name_x, name_y, coord_x, coord_y, and parameter names ("t0", "x1"..."x{n-1}",
        "v1"..."v{n}", "max_offset").
         - Each next line contains row contains the coords names, coords values, and parameters values of one
        RefractorVelocity.

        File example:
         name_x     name_y    coord_x    coord_y        t0        x1        v1        v2 max_offset
        SourceX    SourceY    1111100    2222220     50.00   1000.00   1500.00   2000.00    2000.00
        ...
        SourceX    SourceY    1111200    2222240     60.00   1050.00   1550.00   1950.00    2050.00

        Parameters
        ----------
        path : str,
            path to the file.
        encoding : str, defaults to "UTF-8"
            File encoding.

        Returns
        -------
        self : RefractorVelocityField
            RefractorVelocityField instance created from a file.
        """
        coords_list, params_list, max_offset_list = load_rv(path, encoding)
        rv_list = []
        for coords, params, max_offset in zip(coords_list, params_list, max_offset_list):
            rv = RefractorVelocity(max_offset=max_offset, coords=coords, **params)
            rv_list.append(rv)
        return cls(rv_list, is_geographic=is_geographic)

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
        return self.item_class(**dict(zip(self.param_names, values)), coords=coords)

    def smooth(self, radius, min_refractor_points=10):
        coords = self.coords
        values = self.values
        smoothed_values = np.empty_like(values)
        smoother = partial(IDWInterpolator, radius=radius, dist_transform=0)

        ignore_mask = np.zeros((self.n_items, self.n_refractors), dtype=bool)
        for i, rv in enumerate(self.item_container.values()):
            if rv.is_fit:
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

        # Postprocess smoothed params and construct a new field
        smoothed_values = postprocess_params(smoothed_values)
        smoothed_items = []
        for rv, val in zip(self.item_container.values(), smoothed_values):
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

    def dump(self, path, encoding="UTF-8", min_col_size=11):
        """Save the RefractorVelocityField instance to a file.

        The resulting file have the coordinates and parameters of a single RefractorVelocity with the following
        structure:
         - The first line contains name_x, name_y, coord_x, coord_y, and parameter names ("t0", "x1"..."x{n-1}",
        "v1"..."v{n}", "max_offset").
         - Each next line contains the coords names, coords values, and parameters values corresponding to one
        RefractorVelocity in the resulting RefractorVelocityField.

        File example:
         name_x     name_y    coord_x    coord_y        t0        x1        v1        v2 max_offset
        SourceX    SourceY    1111100    2222220     50.00   1000.00   1500.00   2000.00    2000.00
        ...
        SourceX    SourceY    1111200    2222240     60.00   1050.00   1550.00   1950.00    2050.00

        Parameters
        ----------
        path : str
            Path to the file.
        encoding : str, optional, defaults to "UTF-8"
            File encoding.
        min_col_size : int, defaults to 11
            Minimum size of each columns in the resulting file.

        Returns
        -------
        self : RefractorVelocityField
            RefractorVelocityField unchanged.

        Raises
        ------
        ValueError
            If RefractorVelocityField is empty.
        """
        if self.is_empty:
            raise ValueError("Field is empty. Could not dump empty field.")
        df_list = [calc_df_to_dump(rv) for rv in self.item_container.values()]
        dump_rv(df_list, path=path, encoding=encoding, min_col_size=min_col_size)
        return self

    def plot_fit(self, **kwargs):
        FitPlot(self, **kwargs).plot()
