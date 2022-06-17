import warnings

import numpy as np

from .utils import to_list, read_vfunc, Coordinates, IDWInterpolator, DelaunayInterpolator


class Field:
    field_item_class = None

    def __init__(self, survey=None):
        self.survey = survey
        self.item_container = {}
        self.is_geographic = None
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
    def coords(self):
        """2d np.ndarray or None: Spatial coordinates of field items. `None` for empty field."""
        if self.is_empty:
            return None
        return np.stack(list(self.item_container.keys()))

    def create_interpolator(self, interpolator, **kwargs):
        if self.is_empty:
            raise ValueError("Interpolator cannot be created for an empty field")
        interpolator_name_to_class = {
            "idw": IDWInterpolator,
            "delaunay": DelaunayInterpolator,
        }
        interpolator_class = interpolator_name_to_class.get(interpolator)
        if interpolator_class is None:
            raise ValueError(f"Unknown interpolator {interpolator}. Available options are: "
                             f"{', '.join(interpolator_name_to_class.keys())}")
        self.interpolator = interpolator_class(self.coords, **kwargs)
        self.is_dirty_interpolator = False
        return self

    def update(self, items):
        items = to_list(items)
        if not items:
            return self

        item_class = self.field_item_class or type(items[0])
        if not all(isinstance(item, item_class) for item in items):
            raise TypeError("Wrong types of items being added to the field")
        self.field_item_class = item_class

        # TODO: check for coords consistency in items

        for item in items:
            self.object_container[tuple(item.coords)] = item
        self.is_dirty_interpolator = True
        return self

    def __call__(self, coords):
        if not self.has_interpolator:
            raise ValueError("Field interpolator was not created, call create_interpolator method first")
        if self.is_dirty_interpolator:
            warnings.warn("The field was updated after its interpolator was created", RuntimeWarning)
        if isinstance(coords, Coordinates) and coords.is_geographic is not self.is_geographic:
            raise ValueError("Both coords and field must represent either geographic or line coordinates")
        weighted_coords = self.interpolator.get_weighted_coords(coords)
        items = [self.item_container[coords] for coords in weighted_coords.keys()]
        weights = list(weighted_coords.values())

        # TODO: decide what to do with tuple names
        return self.field_item_class.from_weighted_instances(items, weights=weights, coords=coords)


class VFUNCField(Field):
    @classmethod
    def from_file(cls, path, survey=None):
        coords_cols = ("INLINE_3D", "CROSSLINE_3D")
        objects = [cls.field_object_class.from_points(data_x, data_y, coords=Coordinates(coords, names=coords_cols))
                   for *coords, data_x, data_y in read_vfunc(path)]
        return cls(survey=survey).update(objects)

    # def dump(self, path):
    #     vfunc_list = []
    #     for (inline, crossline), stacking_velocity in self.stacking_velocities_dict.items():
    #         vfunc_list.append((inline, crossline, stacking_velocity.times, stacking_velocity.velocities))
    #     dump_vfunc(path, vfunc_list)
