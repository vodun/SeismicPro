from functools import partial

import numpy as np
import pandas as pd

from .refractor_velocity import RefractorVelocity
from .interactive_plot import FitPlot
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
        return ["t0"] + [f"x{i}" for i in range(1, self.n_refractors)] + [f"v{i+1}" for i in range(self.n_refractors)]

    @property
    def max_offset(self):
        return np.max([item.max_offset for coords, item in self.item_container.items()])

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
    def _postprocess_values(values):
        n_refractors = values.shape[1] // 2

        # Ensure that t0 is non-negative
        np.clip(values[:, 0], 0, None, out=values[:, 0])

        # Ensure that velocities of refractors are non-negative and increasing
        velocities = values[:, n_refractors:]
        np.clip(velocities[:, 0], 0, None, out=velocities[:, 0])
        np.maximum.accumulate(velocities, axis=1, out=velocities)

        # Ensure that crossover offsets are non-negative and increasing
        if n_refractors > 1:
            cross_offsets = values[:, 1:n_refractors]
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
        for rv, val in zip(self.item_container.values(), smoothed_values):
            item = self.construct_item(val, rv.coords)
            item.offsets = rv.offsets
            item.fb_times = rv.fb_times
            smoothed_items.append(item)

        return type(self)(smoothed_items, n_refractors=self.n_refractors, survey=self.survey,
                          is_geographic=self.is_geographic)

    def dump(self, path, encoding="UTF-8", col_size=11):
        """Save the RefractorVelocityField instance to a file.

        File example:
        SourceX   SourceY        t0        x1        v1        v2 max_offset
        1111100   2222220     50.00   1000.00   1500.00   2000.00    2000.00
        ...
        1111200   2222240     60.00   1050.00   1550.00   1950.00    2050.00

        Parameters
        ----------
        path : str
            Path to the file.
        encoding : str, optional, defaults to "UTF-8"
            File encoding.
        col_size : int, defaults to 10
            Size of each columns in file. `col_size` will be increased for coordinate columns if coordinate names
            are longer.

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
        new_col_size = max(col_size, max(len(name) for name in self.coords_cols) + 1)

        columns = list(self.coords_cols) + list(self.param_names) + ["max_offset"]
        cols_format = '{:>{new_col_size}}' * 2 + '{:>{col_size}}' * (len(columns) - 2)
        cols_str = cols_format.format(*columns, new_col_size=new_col_size, col_size=col_size)

        values = np.array([list(item.params.values()) for coords, item in self.item_container.items()])
        # print(np.vstack([item.max_offset for coords, item in self.item_container.items()]))
        max_offsets = np.vstack([item.max_offset for coords, item in self.item_container.items()])
        data = np.hstack((self.coords, values, max_offsets))
        print(data[0])
        data_format = ('\n' + "{:>{new_col_size}.0f}" * 2 + '{:>{col_size}.2f}' * (len(columns) - 2)) * data.shape[0]
        data_str = data_format.format(*data.ravel(), new_col_size=new_col_size, col_size=col_size)

        with open(path, 'w', encoding=encoding) as f:
            f.write(cols_str + data_str)
        return self

    @classmethod
    def load(cls, path, encoding="UTF-8"):
        """Load RefractorVelocityField from a file.

        File example:
        SourceX   SourceY        t0        x1        v1        v2 max_offset
        1111100   2222220     50.00   1000.00   1500.00   2000.00    2000.00
        ...
        1111200   2222240     60.00   1050.00   1550.00   1950.00    2050.00

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
        self = cls()
        df = pd.read_csv(path, sep=r'\s+', encoding=encoding)
        self.n_refractors = (len(df.columns) - 2) // 2
        rv_list = []
        for row in df.to_numpy():
            coords = Coordinates(names=tuple(df.columns[:2]), coords=tuple(row[:2]))
            params = dict(zip(self.param_names, row[2:-1]))
            max_offset = row[-1]
            rv = RefractorVelocity.from_params(params=params, coords=coords, max_offset=max_offset)
            rv_list.append(rv)
        self.update(rv_list)
        return self

    def plot_fit(self, **kwargs):
        FitPlot(self, **kwargs).plot()
