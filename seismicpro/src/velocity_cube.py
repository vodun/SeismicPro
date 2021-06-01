import numpy as np
import cv2
from scipy.interpolate import interp1d, LinearNDInterpolator
from sklearn.neighbors import NearestNeighbors

from .utils import to_list, read_vfunc


class VelocityInterpolator:
    def __init__(self, stacking_velocities_dict):
        self.stacking_velocities_dict = stacking_velocities_dict
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
        return StackingVelocity(interpolator=velocity_interpolator, inline=inline, crossline=crossline)

    def _interpolate_nearest(self, inline, crossline):
        index = self.nearest_interpolator.kneighbors([(inline, crossline),], return_distance=False).item()
        nearest_inline, nearest_crossline = self.coords[index].tolist()
        nearest_stacking_velocity = self.stacking_velocities_dict[(nearest_inline, nearest_crossline)]
        return StackingVelocity(nearest_stacking_velocity.times, nearest_stacking_velocity.velocities,
                                nearest_stacking_velocity.interpolator, inline=inline, crossline=crossline)

    def __call__(self, inline, crossline):
        if self.is_in_hull(inline, crossline):
            return self._interpolate_linear(inline, crossline)
        return self._interpolate_nearest(inline, crossline)


class StackingVelocity:
    def __init__(self, times=None, velocities=None, interpolator=None, inline=None, crossline=None):
        self.times = np.array(times)
        self.velocities = np.array(velocities)
        if (self.times is not None) or (self.velocities is not None):
            if self.times.ndim != 1 or self.times.shape != self.velocities.shape:
                raise ValueError("Incosistent shapes of times and velocities")
            self.interpolator = interp1d(self.times, self.velocities, fill_value="extrapolate")
        else:
            self.interpolator = interpolator
        if self.interpolator is None:
            raise ValueError("Either times and velocities or interpolator must be not-None")
        self.inline = inline
        self.crossline = crossline

    def __call__(self, times):
        return np.maximum(self.interpolator(times), 0)

    def get_coords(self):
        return (self.inline, self.crossline)


class VelocityCube:
    def __init__(self, path=None):
        self.stacking_velocities_dict = {}
        self.interpolator = None
        if path is not None:
            self.load(path)

    def load(self, path):
        for inline, crossline, times, velocities in read_vfunc(path):
            stacking_velocity = StackingVelocity(times=times, velocities=velocities,
                                                 inline=inline, crossline=crossline)
            self.stacking_velocities_dict[(inline, crossline)] = stacking_velocity
        return self

    def update(self, stacking_velocities):
        stacking_velocities = to_list(stacking_velocities)
        if not all(isinstance(vel, StackingVelocity) for vel in stacking_velocities):
            raise ValueError("The cube can be updated only with `StackingVelocity` instances")
        if any((vel.inline is None) or (vel.crossline is None) for vel in stacking_velocities):
            raise ValueError("All passed `StackingVelocity` instances must have not-None coordinates")
        for vel in stacking_velocities:
            self.stacking_velocities_dict[vel.get_coords()] = vel
        return self

    def create_interpolator(self):
        # TODO: interpolate times to a given range
        self.interpolator = VelocityInterpolator(self.stacking_velocities_dict)
        return self

    def get_stacking_velocity(self, inline, crossline):
        if self.interpolator is None:
            raise ValueError("Velocity interpolator must be created first")
        return self.interpolator(inline, crossline)
