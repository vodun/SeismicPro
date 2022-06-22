import warnings

import numpy as np

from .utils import to_list, read_vfunc, dump_vfunc, Coordinates
from .utils.interpolation import IDWInterpolator, DelaunayInterpolator, CloughTocherInterpolator, RBFInterpolator


class Field:
    item_class = None

    def __init__(self, survey=None, is_geographic=None):
        self.survey = survey
        self.item_container = {}
        self.is_geographic = is_geographic
        self.coords_cols = None
        self.interpolator = None
        self.is_dirty_interpolator = True

    @property
    def has_survey(self):
        """bool: Whether a survey is defined for the filed."""
        return self.interpolator is not None

    @property
    def has_interpolator(self):
        """bool: Whether the field interpolator was created."""
        return self.interpolator is not None

    @property
    def is_empty(self):
        """bool: Whether the field is empty."""
        return len(self.item_container) == 0

    @property
    def available_interpolators(self):
        """dict: A mapping from an available interpolator name to the corresponding class."""
        return {}

    @property
    def coords(self):
        """np.ndarray: spatial coordinates of field items."""
        return np.stack(list(self.item_container.keys()))

    @property
    def values(self):
        """np.ndarray or None: values to be passed to the field interpolator class."""
        raise NotImplementedError

    def create_interpolator(self, interpolator, **kwargs):
        if self.is_empty:
            raise ValueError("Interpolator cannot be created for an empty field")
        interpolator_class = self.available_interpolators.get(interpolator)
        if interpolator_class is None:
            raise ValueError(f"Unknown interpolator {interpolator}. Available options are: "
                             f"{', '.join(self.available_interpolators.keys())}")
        self.interpolator = interpolator_class(self.coords, self.values, **kwargs)
        self.is_dirty_interpolator = False
        return self

    def transform_coords(self, coords, to_geographic=None):
        if to_geographic is None:
            to_geographic = self.is_geographic

        coords_arr = np.array(coords)
        is_1d_coords = coords_arr.ndim == 1
        if is_1d_coords:
            coords = [coords]
        coords_arr = np.atleast_2d(coords_arr)
        if coords_arr.ndim != 2 or coords_arr.shape[1] != 2:
            raise ValueError("Wrong shape of passed coordinates")

        need_cast_mask = np.zeros(len(coords), dtype=bool)
        for i, coord in enumerate(coords):
            if isinstance(coord, Coordinates):
                need_cast_mask[i] = coord.is_geographic is not to_geographic

        if need_cast_mask.any():
            # TODO: cast coords to field coords using survey transforms
            raise ValueError("Both coords and field must represent either geographic or line coordinates")

        return coords_arr, coords, is_1d_coords

    def update(self, items):
        items = to_list(items)
        if not items:
            return self
        if not all(isinstance(item, self.item_class) for item in items):
            raise TypeError(f"The field can be updated only with instances of {self.item_class} class")

        # Infer is_geographic and coords_cols during the first update
        is_geographic = self.is_geographic
        if self.is_geographic is None:
            is_geographic = items[0].coords.is_geographic

        coords_cols_set = {item.coords.names for item in items}
        coords_cols = coords_cols_set.pop() if len(coords_cols_set) == 1 else None
        if not self.is_empty and coords_cols != self.coords_cols:
            coords_cols = None

        # Update the field
        field_coords, _, _ = self.transform_coords([item.coords for item in items], to_geographic=is_geographic)
        for coords, item in zip(field_coords, items):
            self.item_container[tuple(coords)] = item
        self.is_dirty_interpolator = True
        self.is_geographic = is_geographic
        self.coords_cols = coords_cols
        return self

    def validate_interpolator(self):
        """Validate that field interpolator is created and warn if it's dirty."""
        if not self.has_interpolator:
            raise ValueError("Field interpolator was not created, call create_interpolator method first")
        if self.is_dirty_interpolator:
            warnings.warn("The field was updated after its interpolator was created", RuntimeWarning)

    def construct_items(self, field_coords, return_coords):
        _ = field_coords, return_coords
        raise NotImplementedError

    def __call__(self, coords):
        self.validate_interpolator()
        field_coords, return_coords, is_1d_coords = self.transform_coords(coords)
        if self.coords_cols is None and not all(isinstance(coords, Coordinates) for coords in return_coords):
            raise ValueError("Names of field coordinates are undefined, so only Coordinates instances are allowed")
        return_coords = [coords if isinstance(coords, Coordinates) else Coordinates(coords, names=self.coords_cols)
                         for coords in return_coords]
        items = self.construct_items(field_coords, return_coords)
        if is_1d_coords:
            return items[0]
        return items


class SpatialField(Field):
    @property
    def available_interpolators(self):
        interpolators = {
            "idw": IDWInterpolator,
            "delaunay": DelaunayInterpolator,
            "ct": CloughTocherInterpolator,
            "rbf": RBFInterpolator,
        }
        return interpolators

    def interpolate(self, coords):
        self.validate_interpolator()
        field_coords, _, is_1d_coords = self.transform_coords(coords)
        values = self.interpolator(field_coords)
        if is_1d_coords:
            return values[0]
        return values

    def construct_item(self, values, coords):
        _ = values, coords
        raise NotImplementedError

    def construct_items(self, field_coords, return_coords):
        values = self.interpolator(field_coords)
        return [self.construct_item(vals, coords) for vals, coords in zip(values, return_coords)]


class ValuesAgnosticField(Field):
    @property
    def available_interpolators(self):
        interpolators = {
            "idw": IDWInterpolator,
            "delaunay": DelaunayInterpolator,
        }
        return interpolators

    @property
    def values(self):
        """None: ValuesAgnosticField does not require values to be passed to the interpolator class."""
        return None

    def construct_item(self, base_items, weights, coords):
        _ = base_items, weights, coords
        raise NotImplementedError

    def construct_items(self, field_coords, return_coords):
        weighted_coords_list = self.interpolator.get_weighted_coords(field_coords)
        items = []
        for weighted_coords, coords in zip(weighted_coords_list, return_coords):
            base_items = [self.item_container[coords] for coords in weighted_coords.keys()]
            weights = list(weighted_coords.values())
            items.append(self.construct_item(base_items, weights, coords))
        return items


class VFUNCMixin:
    @classmethod
    def from_file(cls, path, coords_cols=("INLINE_3D", "CROSSLINE_3D"), survey=None):
        items = [cls.item_class.from_points(data_x, data_y, coords=coords)
                 for coords, data_x, data_y in read_vfunc(path, coords_cols=coords_cols)]
        return cls(survey=survey).update(items)

    def dump(self, path):
        dump_vfunc(path, [(coords, *item.vfunc_data) for coords, item in self.item_container.items()])
