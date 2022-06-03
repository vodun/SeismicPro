import warnings

import numpy as np

from .utils import to_list, read_vfunc, Coordinates, IDWInterpolator, DelaunayInterpolator


class Field:
    field_object_class = None

    def __init__(self, survey=None):
        self.object_container = {}
        self.interpolator = None
        self.is_dirty_interpolator = True
        self.survey = survey

    @property
    def coords(self):
        return np.stack(list(self.object_container.keys()))

    @property
    def has_interpolator(self):
        """bool: Whether the field interpolator was created."""
        return self.interpolator is not None

    @property
    def has_survey(self):
        """bool: Whether a survey is defined for the filed."""
        return self.interpolator is not None

    def create_interpolator(self, interpolator, **kwargs):
        if not self.object_container:
            raise ValueError("Interpolator cannot be created for an empty field")
        interpolator_name_to_class = {
            "idw": IDWInterpolator,
            "delaunay": DelaunayInterpolator,
        }
        interpolator_class = interpolator_name_to_class[interpolator]
        self.interpolator = interpolator_class(self.coords, **kwargs)
        self.is_dirty_interpolator = False
        return self

    def update(self, objects):
        objects = to_list(objects)
        if not objects:
            return self

        object_class = self.field_object_class or type(objects[0])
        if not all(isinstance(obj, object_class) for obj in objects):
            raise TypeError("Wrong types of objects being added to the field")
        # TODO: check for coords cols and define them for the field
        self.field_object_class = object_class
        for obj in objects:
            self.object_container[tuple(obj.coords)] = obj  # TODO: cast coords to an x-y format
        self.is_dirty_interpolator = True
        return self

    def __call__(self, coords):
        if not self.has_interpolator:
            raise ValueError("Field interpolator was not created, call create_interpolator method first")
        if self.is_dirty_interpolator:
            warnings.warn("The field was updated after its interpolator was created", RuntimeWarning)
        # TODO: validate coords, cast them to field coords if needed
        weighted_coords = self.interpolator.get_weighted_coords(coords)
        objects = [self.object_container[coords] for coords in weighted_coords.keys()]
        weights = list(weighted_coords.values())
        return self.field_object_class.from_weighted_instances(objects, weights=weights, coords=coords)


class VFUNCField(Field):
    @classmethod
    def from_file(cls, path, coords_cols=("INLINE_3D", "CROSSLINE_3D"), survey=None):
        objects = [cls.field_object_class.from_points(data_x, data_y, coords=Coordinates(coord_x, coord_y, names=coords_cols))
                   for coord_x, coord_y, data_x, data_y in read_vfunc(path)]
        return cls(survey=survey).update(objects)

    # def dump(self, path):
    #     vfunc_list = []
    #     for (inline, crossline), stacking_velocity in self.stacking_velocities_dict.items():
    #         vfunc_list.append((inline, crossline, stacking_velocity.times, stacking_velocity.velocities))
    #     dump_vfunc(path, vfunc_list)
