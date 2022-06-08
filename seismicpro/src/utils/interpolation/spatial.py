"""Implements classes and functions for 2d interpolation and extrapolation"""

import cv2
import numpy as np
from scipy import interpolate
from scipy.spatial.qhull import Delaunay, QhullError
from sklearn.neighbors import NearestNeighbors


def parse_inputs(coords, values=None):
    coords = np.array(coords, dtype=np.float64, order="C")
    is_coords_1d = (coords.ndim == 1)
    coords = np.atleast_2d(coords)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (n_coords, 2) or (2,)")
    if values is None:
        return coords, is_coords_1d, None, None

    values = np.array(values, dtype=np.float64, order="C")
    is_values_1d = (values.ndim == 1)
    if is_values_1d and is_coords_1d:
        is_values_1d = False
        values = np.atleast_2d(coords)
    if is_values_1d:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or len(values) != len(coords):
        raise ValueError("values must have shape (n_coords,) or (n_coords, n_values)")
    return coords, is_coords_1d, values, is_values_1d


class SpatialInterpolator:
    def __init__(self, coords, values=None):
        self.coords, _, self.values, self.is_values_1d = parse_inputs(coords, values)

    @property
    def has_values(self):
        return self.values is not None

    def _interpolate(self, coords):
        _ = coords
        raise NotImplementedError

    def __call__(self, coords):
        if not self.has_values:
            raise ValueError("The interpolator requires values to be passed to be callable")
        coords, is_coords_1d, _, _ = parse_inputs(coords)
        values = self._interpolate(coords)
        if self.is_values_1d:
            values = values[:, 0]
        if is_coords_1d:
            return values[0]
        return values


class ValuesAgnosticInterpolator(SpatialInterpolator):
    def _get_weighted_coords(self, coords):
        _ = coords
        raise NotImplementedError

    def get_weighted_coords(self, coords):
        coords, is_coords_1d, _, _ = parse_inputs(coords)
        weighted_coords = self._get_weighted_coords(coords)
        if is_coords_1d:
            return weighted_coords[0]
        return weighted_coords


class IDWInterpolator(ValuesAgnosticInterpolator):
    def __init__(self, coords, values=None, radius=None, neighbors=None, dist_transform=2, smoothing=0):
        super().__init__(coords, values)
        if neighbors is None:
            neighbors = len(self.coords)
        neighbors = min(neighbors, len(self.coords))
        self.use_radius = radius is not None
        self.nearest_interpolator = NearestNeighbors(n_neighbors=neighbors, radius=radius).fit(self.coords)
        self.dist_transform = dist_transform
        self.smoothing = smoothing

    def _distances_to_weights(self, dist):
        is_1d_dist = (dist.ndim == 1)
        dist = np.atleast_2d(dist)

        # Transform distances according to dist_transform and smooth them if needed
        if callable(self.dist_transform):
            dist = self.dist_transform(dist)
        else:
            dist **= self.dist_transform
        dist += self.smoothing

        # Calculate weights from distances: correctly handle case of interpolating at known coords
        zero_mask = np.isclose(dist, 0)
        dist[zero_mask] = 1  # suppress division by zero warning
        weights = 1 / dist
        weights[zero_mask.any(axis=1)] = 0
        weights[zero_mask] = 1
        weights /= weights.sum(axis=1, keepdims=True)

        if is_1d_dist:
            return weights[0]
        return weights

    def _get_neighbors_weights(self, coords):
        dist, indices = self.nearest_interpolator.kneighbors(coords, return_distance=True)
        return indices, self._distances_to_weights(dist)

    def _interpolate_neighbors(self, coords):
        if len(coords) == 0:
            return np.empty((0, self.values.shape[1]), dtype=self.values.dtype)
        base_indices, base_weights = self._get_neighbors_weights(coords)
        base_values = self.values[base_indices]
        return (base_values * base_weights[:, :, None]).sum(axis=1).astype(self.values.dtype)

    def _get_weighted_coords_neighbors(self, coords):
        if len(coords) == 0:
            return np.empty(0, dtype=object)
        base_indices, base_weights = self._get_neighbors_weights(coords)
        base_coords = self.coords[base_indices]
        non_zero_mask = ~np.isclose(base_weights, 0)
        weighted_coords = [{tuple(coord): weight for coord, weight in zip(coords[mask], weights[mask])}
                           for coords, weights, mask in zip(base_coords, base_weights, non_zero_mask)]
        return np.array(weighted_coords, dtype=object)

    def _get_radius_weights(self, coords):
        dist, indices = self.nearest_interpolator.radius_neighbors(coords, return_distance=True)
        empty_radius_mask = np.array([len(ix) == 0 for ix in indices])
        indices = indices[~empty_radius_mask]
        dist = dist[~empty_radius_mask]
        weights = np.empty_like(dist, dtype=object)
        for i, dist_item in enumerate(dist):  # dist is an array of arrays thus direct iteration is required
            weights[i] = self._distances_to_weights(dist_item)
        return indices, weights, empty_radius_mask

    def _interpolate_radius(self, coords):
        base_indices, base_weights, empty_radius_mask = self._get_radius_weights(coords)
        values = np.empty((len(coords), self.values.shape[1]), dtype=self.values.dtype)
        values[empty_radius_mask] = self._interpolate_neighbors(coords[empty_radius_mask])
        if len(base_indices):
            values[~empty_radius_mask] = [(self.values[indices] * weights[:, None]).sum(axis=0)
                                           for indices, weights in zip(base_indices, base_weights)]
        return values

    def _get_weighted_coords_radius(self, coords):
        base_indices, base_weights, empty_radius_mask = self._get_radius_weights(coords)
        values = np.empty(len(coords), dtype=object)
        values[empty_radius_mask] = self._get_weighted_coords_neighbors(coords[empty_radius_mask])

        weighted_coords = np.empty(len(base_indices), dtype=object)
        for i, (indices, weights) in enumerate(zip(base_indices, base_weights)):
            non_zero_mask = ~np.isclose(weights, 0)
            coords = self.coords[indices[non_zero_mask]]
            weights = weights[non_zero_mask]
            weighted_coords[i] = {tuple(coord): weight for coord, weight in zip(coords, weights)}
        values[~empty_radius_mask] = weighted_coords
        return values

    def _interpolate(self, coords):
        if self.use_radius:
            return self._interpolate_radius(coords)
        return self._interpolate_neighbors(coords)

    def _get_weighted_coords(self, coords):
        if self.use_radius:
            return self._get_weighted_coords_radius(coords)
        return self._get_weighted_coords_neighbors(coords)


class BaseDelaunayInterpolator(SpatialInterpolator):
    def __init__(self, coords, values=None, neighbors=3, dist_transform=2):
        super().__init__(coords, values)

        # Construct a convex hull of passed coords. Cast coords to float32, otherwise cv2 may fail
        self.coords_hull = cv2.convexHull(self.coords.astype(np.float32), returnPoints=True)

        # Construct an IDW interpolator to use outside the constructed hull
        self.idw_interpolator = IDWInterpolator(coords, values, neighbors=neighbors, dist_transform=dist_transform)

        # Triangulate input points
        try:
            self.tri = Delaunay(self.coords, incremental=False)
        except QhullError:
            # Delaunay fails in case of linearly dependent coordinates. Create artificial points in the corners of
            # given coordinate grid in order for Delaunay to work with a full rank matrix.
            min_x, min_y = np.min(self.coords, axis=0) - 1
            max_x, max_y = np.max(self.coords, axis=0) + 1
            corner_coords = [(min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y)]
            self.coords = np.concatenate([self.coords, corner_coords])
            if self.values is not None:
                mean_values = np.mean(self.values, axis=0, keepdims=True, dtype=self.values.dtype)
                corner_values = np.repeat(mean_values, 4, axis=0)
                self.values = np.concatenate([self.values, corner_values])
            self.tri = Delaunay(self.coords, incremental=False)

        # Perform the first auxiliary call of the tri for it to work properly in different processes.
        # Otherwise interpolation may fail if called in a pipeline with prefetch with mpc target.
        _ = self.tri.find_simplex((0, 0))

    def _is_in_hull(self, coords):
        # Cast coords to float32 to match the type of points in the convex hull
        return cv2.pointPolygonTest(self.coords_hull, np.array(coords, dtype=np.float32), measureDist=False) >= 0

    def _interpolate_inside_hull(self, coords):
        _ = coords
        raise NotImplementedError

    def _interpolate(self, coords):
        inside_hull_mask = np.array([self._is_in_hull(coord) for coord in coords])
        values = np.empty((len(coords), self.values.shape[1]), dtype=self.values.dtype)
        values[inside_hull_mask] = self._interpolate_inside_hull(coords[inside_hull_mask])
        # pylint: disable-next=protected-access
        values[~inside_hull_mask] = self.idw_interpolator._interpolate(coords[~inside_hull_mask])
        return values


class DelaunayInterpolator(BaseDelaunayInterpolator, ValuesAgnosticInterpolator):
    def _get_simplex_info(self, coords):
        """Return indices of simplex vertices and corresponding barycentric coordinates for each of passed `coords`."""
        simplex_ix = self.tri.find_simplex(coords)
        if np.any(simplex_ix < 0):
            raise ValueError("Some passed coords are outside convex hull of known coordinates")
        transform = self.tri.transform[simplex_ix]
        transition = transform[:, :2]
        bias = transform[:, 2]
        bar_coords = np.sum(transition * np.expand_dims(coords - bias, axis=1), axis=-1)
        bar_coords = np.column_stack([bar_coords, 1 - bar_coords.sum(axis=1)])
        return self.tri.simplices[simplex_ix], bar_coords

    def _interpolate_inside_hull(self, coords):
        simplices_indices, bar_coords = self._get_simplex_info(coords)
        return (self.values[simplices_indices] * bar_coords[:, :, None]).sum(axis=1)

    def _get_weighted_coords(self, coords):
        inside_hull_mask = np.array([self._is_in_hull(coord) for coord in coords])
        weights = np.empty(len(coords), dtype=object)
        # pylint: disable-next=protected-access
        weights[~inside_hull_mask] = self.idw_interpolator._get_weighted_coords(coords[~inside_hull_mask])

        simplices_indices, bar_coords = self._get_simplex_info(coords[inside_hull_mask])
        simplices_coords = self.coords[simplices_indices]
        non_zero_mask = ~np.isclose(bar_coords, 0)
        weights[inside_hull_mask] = [{tuple(point): weight for point, weight in zip(simplices[mask], weights[mask])}
                                     for simplices, weights, mask in zip(simplices_coords, bar_coords, non_zero_mask)]
        return weights


class CloughTocherInterpolator(BaseDelaunayInterpolator):
    def __init__(self, coords, values, neighbors=3, dist_transform=2, **kwargs):
        super().__init__(coords, values, neighbors=neighbors, dist_transform=dist_transform)
        self.ct_interpolator = interpolate.CloughTocher2DInterpolator(self.tri, self.values, **kwargs)

    def _interpolate_inside_hull(self, coords):
        return self.ct_interpolator(coords)


class RBFInterpolator(SpatialInterpolator):
    def __init__(self, coords, values, neighbors=None, smoothing=0, **kwargs):
        super().__init__(coords, values)
        self.rbf_interpolator = interpolate.RBFInterpolator(self.coords, self.values, neighbors=neighbors,
                                                            smoothing=smoothing, **kwargs)

    def _interpolate(self, coords):
        return self.rbf_interpolator(coords).astype(self.values.dtype)
