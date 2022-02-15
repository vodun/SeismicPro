import cv2
import numpy as np
from numba import njit, prange
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.qhull import Delaunay, QhullError  #pylint: disable=no-name-in-module

from .stacking_velocity import StackingVelocity


class VelocityInterpolator:
    """A class for stacking velocity interpolation over the whole field.

    Velocity interpolator accepts a dict of stacking velocities and constructs a convex hull of their coordinates.
    After that, given an inline and a crossline of unknown stacking velocity to get, interpolation is performed in the
    following way:
    1. If spatial coordinates lie within the constructed convex hull, linear barycentric interpolation for each time
       over spatially Delaunay-triangulated data is calculated,
    2. Otherwise, the closest known stacking velocity in Euclidean distance is returned.

    Parameters
    ----------
    stacking_velocities_dict : dict
        A dict of stacking velocities whose keys are tuples with their spatial coordinates and values are the stacking
        velocities themselves.

    Attributes
    ----------
    stacking_velocities_dict : dict
        A dict of stacking velocities whose keys are tuples with their spatial coordinates and values are the stacking
        velocities themselves.
    coords : 2d np.ndarray
        Stacked coordinates of velocities from `stacking_velocities_dict`.
    coords_hull : 3d np.ndarray
        Convex hull of coordinates of stacking velocities. Later used to choose the interpolation strategy for the
        requested coordinates.
    nearest_interpolator : NearestNeighbors
        An estimator of the closest stacking velocity to the passed spatial coordinates.
    tri : Delaunay
        Delaunay triangulation of `coords`.
    """
    def __init__(self, stacking_velocities_dict):
        self.stacking_velocities_dict = stacking_velocities_dict

        # Calculate the convex hull of given stacking velocity coordinates to further select appropriate interpolator
        self.coords = np.stack(list(self.stacking_velocities_dict.keys()))
        self.coords_hull = cv2.convexHull(self.coords, returnPoints=True)

        self.nearest_interpolator = NearestNeighbors(n_neighbors=1)
        self.nearest_interpolator.fit(self.coords)

        try:
            self.tri = Delaunay(self.coords, incremental=False)
        except QhullError:
            # Create artificial stacking velocities in the corners of given coordinate grid in order for Delaunay to
            # work with a full rank matrix of coordinates
            min_i, min_x = np.min(self.coords, axis=0) - 1
            max_i, max_x = np.max(self.coords, axis=0) + 1
            corner_velocities_coords = [(min_i, min_x), (min_i, max_x), (max_i, min_x), (max_i, max_x)]
            self.tri = Delaunay(np.concatenate([self.coords, corner_velocities_coords]), incremental=False)

        # Perform the first auxilliary call of the tri for it to work properly in different processes.
        # Otherwise VelocityCube.__call__ may fail if called in a pipeline with prefetch with mpc target.
        _ = self.tri.find_simplex((0, 0))

    def _is_in_hull(self, inline, crossline):
        """Check if given `inline` and `crossline` lie within a convex hull of spatial coordinates of stacking
        velocities passed during interpolator creation."""
        # Cast inline and crossline to pure python types for latest versions of opencv to work correctly
        inline = np.array(inline).item()
        crossline = np.array(crossline).item()
        return cv2.pointPolygonTest(self.coords_hull, (inline, crossline), measureDist=True) >= 0

    def _get_simplex_info(self, coords):
        """Return indices of simplex vertices and corresponding barycentric coordinates for each of passed `coords`."""
        simplex_ix = self.tri.find_simplex(coords)
        if np.any(simplex_ix < 0):
            raise ValueError("Some passed coords are outside convex hull of known stacking velocities coordinates")
        transform = self.tri.transform[simplex_ix]
        transition = transform[:, :2]
        bias = transform[:, 2]
        bar_coords = np.sum(transition * np.expand_dims(coords - bias, axis=1), axis=-1)
        bar_coords = np.column_stack([bar_coords, 1 - bar_coords.sum(axis=1)])
        return self.tri.simplices[simplex_ix], bar_coords

    def _interpolate_barycentric(self, inline, crossline):
        """Perform linear barycentric interpolation of stacking velocity at given `inline` and `crossline`."""
        coords = [(inline, crossline)]
        (simplex,), (bar_coords,) = self._get_simplex_info(coords)
        non_zero_coords = ~np.isclose(bar_coords, 0)
        bar_coords = bar_coords[non_zero_coords]
        vertices = self.tri.points[simplex][non_zero_coords].astype(np.int32)
        vertex_velocities = [self.stacking_velocities_dict[tuple(ver)] for ver in vertices]

        def velocity_interpolator(times):
            times = np.array(times)
            velocities = (np.column_stack([vel(times) for vel in vertex_velocities]) * bar_coords).sum(axis=1)
            return velocities.item() if times.ndim == 0 else velocities

        return StackingVelocity.from_interpolator(velocity_interpolator, inline, crossline)

    def _get_nearest_velocities(self, coords):
        """Return the closest known stacking velocity to each of passed `coords`."""
        nearest_indices = self.nearest_interpolator.kneighbors(coords, return_distance=False)[:, 0]
        nearest_coords = self.coords[nearest_indices]
        velocities = [self.stacking_velocities_dict[tuple(coord)] for coord in nearest_coords]
        return velocities

    def _interpolate_nearest(self, inline, crossline):
        """Return the closest known stacking velocity to given `inline` and `crossline`."""
        nearest_stacking_velocity = self._get_nearest_velocities([(inline, crossline),])[0]
        return StackingVelocity.from_points(nearest_stacking_velocity.times, nearest_stacking_velocity.velocities,
                                            inline=inline, crossline=crossline)

    @staticmethod
    @njit(nogil=True, parallel=True)
    def _interp(simplices, bar_coords, velocities):
        res = np.zeros((len(simplices), velocities.shape[-1]), dtype=velocities.dtype)
        for i in prange(len(simplices)):  #pylint: disable=not-an-iterable
            for vel_ix, bar_coord in zip(simplices[i], bar_coords[i]):
                res[i] += velocities[vel_ix] * bar_coord
        return res

    def interpolate(self, coords, times):
        """Interpolate stacking velocity at given coords and times.

        Interpolation over a regular grid of times allows for implementing a much more efficient computation strategy
        than simply iteratively calling the interpolator for each of `coords` and that evaluating it at given `times`.

        Parameters
        ----------
        coords : 2d array-like
            Coordinates to interpolate stacking velocities at. Has shape `(n_coords, 2)`.
        times : 1d array-like
            Times to interpolate stacking velocities at.

        Returns
        -------
        velocities : 2d np.ndarray
            Interpolated stacking velocities at given `coords` and `times`. Has shape `(len(coords), len(times))`.
        """
        coords = np.array(coords)
        times = np.array(times)
        velocities = np.empty((len(coords), len(times)))

        knn_mask = np.array([not self._is_in_hull(*coord) for coord in coords])
        coords_nearest = coords[knn_mask]
        coords_barycentric = coords[~knn_mask]

        if len(coords_nearest):
            velocities[knn_mask] = np.array([vel(times) for vel in self._get_nearest_velocities(coords_nearest)])

        if len(coords_barycentric):
            # Determine a simplex and barycentric coordinates inside it for all the coords
            simplices, bar_coords = self._get_simplex_info(coords_barycentric)

            # Calculate stacking velocities for all required simplex vertices and given times
            base_velocities = np.empty((len(self.tri.points), len(times)))
            vertex_indices = np.unique(simplices)
            vertex_coords = self.tri.points[vertex_indices].astype(np.int32)
            for vert_ix, vert_coords in zip(vertex_indices, vertex_coords):
                vel = self.stacking_velocities_dict.get(tuple(vert_coords))
                if vel is not None:
                    base_velocities[vert_ix] = vel(times)

            velocities[~knn_mask] = self._interp(simplices, bar_coords, base_velocities)

        return velocities

    def __call__(self, inline, crossline):
        """Interpolate stacking velocity at given `inline` and `crossline`.

        If `inline` and `crossline` lie within a convex hull of spatial coordinates of known stacking velocities,
        perform linear barycentric interpolation. Otherwise return stacking velocity closest to coordinates passed.

        Parameters
        ----------
        inline : int
            An inline to interpolate stacking velocity at.
        crossline : int
            A crossline to interpolate stacking velocity at.

        Returns
        -------
        stacking_velocity : StackingVelocity
            Interpolated stacking velocity at (`inline`, `crossline`).
        """
        if self._is_in_hull(inline, crossline):
            return self._interpolate_barycentric(inline, crossline)
        return self._interpolate_nearest(inline, crossline)
