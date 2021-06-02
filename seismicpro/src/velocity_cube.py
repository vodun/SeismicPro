import warnings

import numpy as np
import cv2
from scipy.interpolate import interp1d, LinearNDInterpolator
from sklearn.neighbors import NearestNeighbors

from .utils import to_list, read_vfunc, read_single_vfunc


class VelocityInterpolator:
    def __init__(self, stacking_velocities_dict, tmin=-np.inf, tmax=np.inf):
        if not stacking_velocities_dict:
            raise ValueError("No stacking velocities passed")

        # Set the time range of all stacking velocities to [tmin, tmax] in order to ensure that the convex hull of the
        # point cloud passed to LinearNDInterpolator covers all timestamps from tmin to tmax
        self.stacking_velocities_dict = {}
        for coord, stacking_velocity in stacking_velocities_dict:
            self.stacking_velocities_dict[coord] = stacking_velocity.set_time_range(tmin, tmax)

        # Calculate the convex hull of given stacking velocity coordinates to further select appropriate interpolator
        self.coords = np.stack(list(stacking_velocities_dict.keys()))
        self.coords_hull = cv2.convexHull(self.coords, returnPoints=True)

        self.nearest_interpolator = NearestNeighbors(n_neighbors=1)
        self.nearest_interpolator.fit(self.coords)

        # Create artificial stacking velocities in the corners of given coordinate grid in order for
        # LinearNDInterpolator to work with a full rank matrix of coordinates
        min_i, min_x = np.min(self.coords, axis=0) - 1
        max_i, max_x = np.max(self.coords, axis=0) + 1
        fake_velocities = [self._interpolate_nearest(i, x)
                           for i, x in [(min_i, min_x), (min_i, max_x), (max_i, min_x), (max_i, max_x)]]
        vel_data = [np.concatenate([[vel.get_coords()] * len(vel.times), vel.times, vel.velocities], axis=1)
                    for vel in list(stacking_velocities_dict.values()) + fake_velocities]
        vel_data = np.concatenate(vel_data)
        self.linear_interpolator = LinearNDInterpolator(vel_data[:, :-1], vel_data[:, -1])

    def is_in_hull(self, inline, crossline):
        return cv2.pointPolygonTest(self.coords_hull, (inline, crossline), measureDist=True) >= 0

    def _interpolate_linear(self, inline, crossline):
        velocity_interpolator = lambda times: self.linear_interpolator(inline, crossline, times)
        return StackingVelocity.from_interpolator(velocity_interpolator, inline, crossline)

    def _interpolate_nearest(self, inline, crossline):
        index = self.nearest_interpolator.kneighbors([(inline, crossline),], return_distance=False).item()
        nearest_inline, nearest_crossline = self.coords[index].tolist()
        nearest_stacking_velocity = self.stacking_velocities_dict[(nearest_inline, nearest_crossline)]
        return StackingVelocity.from_points(nearest_stacking_velocity.times, nearest_stacking_velocity.velocities,
                                            inline=inline, crossline=crossline)

    def __call__(self, inline, crossline):
        if self.is_in_hull(inline, crossline):
            return self._interpolate_linear(inline, crossline)
        return self._interpolate_nearest(inline, crossline)


class StackingVelocity:
    def __init__(self):
        self.interpolator = lambda times: np.zeros_like(times)
        self.times = None
        self.velocities = None
        self.inline = None
        self.crossline = None

    @classmethod
    def from_points(cls, times, velocities, inline=None, crossline=None):
        times = np.array(times)
        velocities = np.array(velocities)
        if times.ndim != 1 or times.shape != velocities.shape:
            raise ValueError("Incosistent shapes of times and velocities")
        if (velocities < 0).any():
            raise ValueError("Velocity values must be positive")
        self = cls.from_interpolator(interp1d(times, velocities, fill_value="extrapolate"), inline, crossline)
        self.times = times
        self.velocities = velocities
        return self

    @classmethod
    def from_file(cls, path):
        inline, crossline, times, velocities = read_single_vfunc(path)
        return cls.from_points(times, velocities, inline, crossline)

    @classmethod
    def from_interpolator(cls, interpolator, inline=None, crossline=None):
        self = cls()
        self.interpolator = interpolator
        self.inline = inline
        self.crossline = crossline
        return self

    def set_time_range(self, tmin, tmax):
        if not self.has_points:
            raise ValueError("Time range can be set only for StackingVelocity instances, created from points")
        valid_time_mask = (self.times > tmin) & (self.times < tmax)
        times = np.array([tmin, *self.times[valid_time_mask], tmax])
        times = times[~np.isinf(times)]
        return self.from_points(times, self(times), self.inline, self.crossline)

    @property
    def has_points(self):
        return (self.times is not None) and (self.velocities is not None)

    @property
    def has_coords(self):
        return (self.inline is not None) and (self.crossline is not None)

    def get_coords(self):
        return (self.inline, self.crossline)

    def __call__(self, times):
        return np.maximum(self.interpolator(times), 0)


class VelocityCube:
    def __init__(self, tmin=-np.inf, tmax=np.inf, path=None):
        self.stacking_velocities_dict = {}
        self.interpolator = None
        self.is_dirty_interpolator = True
        self.tmin = tmin
        self.tmax = tmax
        if path is not None:
            self.load(path)

    @property
    def has_interpolator(self):
        return self.interpolator is not None

    def load(self, path):
        for inline, crossline, times, velocities in read_vfunc(path):
            stacking_velocity = StackingVelocity.from_points(times, velocities, inline, crossline)
            self.stacking_velocities_dict[(inline, crossline)] = stacking_velocity
        self.is_dirty_interpolator = True
        return self

    def update(self, stacking_velocities):
        stacking_velocities = to_list(stacking_velocities)
        if not all(isinstance(vel, StackingVelocity) for vel in stacking_velocities):
            raise ValueError("The cube can be updated only with `StackingVelocity` instances")
        if not all(vel.has_coords for vel in stacking_velocities):
            raise ValueError("All passed `StackingVelocity` instances must have not-None coordinates")
        for vel in stacking_velocities:
            self.stacking_velocities_dict[vel.get_coords()] = vel
        if stacking_velocities:
            self.is_dirty_interpolator = True
        return self

    def create_interpolator(self):
        self.interpolator = VelocityInterpolator(self.stacking_velocities_dict, self.tmin, self.tmax)
        self.is_dirty_interpolator = False
        return self

    def get_stacking_velocity(self, inline, crossline, create_interpolator=True):
        if create_interpolator and (not self.has_interpolator or self.is_dirty_interpolator):
            self.create_interpolator()
        elif not create_interpolator:
            if not self.has_interpolator:
                raise ValueError("Velocity interpolator must be created first")
            if self.is_dirty_interpolator:
                warnings.warn("Dirty interpolator is being used", RuntimeWarning)
        return self.interpolator(inline, crossline)
