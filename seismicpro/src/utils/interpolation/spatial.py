import cv2
import numpy as np
from scipy import interpolate
from scipy.spatial.qhull import Delaunay, QhullError
from sklearn.neighbors import NearestNeighbors


def parse_inputs(coords, values=None):
    coords = np.array(coords)
    is_coords_1d = (coords.ndim == 1)
    coords = np.atleast_2d(coords)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (n_coords, 2) or (2,)")
    if values is None:
        return coords, is_coords_1d, None, None

    values = np.array(values)
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


class CoordsOnlyInterpolator(SpatialInterpolator):
    def _get_weighted_coords(self, coords):
        _ = coords
        raise NotImplementedError

    def get_weighted_coords(self, coords):
        coords, is_coords_1d, _, _ = parse_inputs(coords)
        weighted_coords = self._get_weighted_coords(coords)
        if is_coords_1d:
            return weighted_coords[0]
        return weighted_coords


class IDWInterpolator(CoordsOnlyInterpolator):
    def __init__(self, coords, values=None, radius=None, n_neighbors=10, dist_transform=2):
        super().__init__(coords, values)
        n_neighbors = min(n_neighbors, len(self.coords))
        self.use_radius = radius is not None
        self.nearest_interpolator = NearestNeighbors(n_neighbors=n_neighbors, radius=radius).fit(self.coords)
        if not callable(dist_transform):
            dist_transform = lambda x, dist_transform=dist_transform: x**dist_transform
        self.dist_transform = dist_transform

    def _get_neighbors_weights(self, coords):
        dist, indices = self.nearest_interpolator.kneighbors(coords, return_distance=True)

        # Calculate weights from distances: correctly handle case of interpolating in known coords
        weights = self.dist_transform(dist)
        zero_mask = np.isclose(weights, 0)
        zero_coords = zero_mask.any(axis=1)
        to_zero_mask = np.zeros_like(zero_mask, dtype=bool)
        to_zero_mask[zero_coords] = ~zero_mask[zero_coords]

        weights[zero_mask] = 1
        weights = 1 / weights
        weights[to_zero_mask] = 0
        weights /= weights.sum(axis=1, keepdims=True)
        return indices, weights

    def _interpolate_neighbors(self, coords):
        if len(coords) == 0:
            return np.empty((0, self.values.shape[1]), dtype=self.values.dtype)
        indices, weights = self._get_neighbors_weights(coords)
        return (self.values[indices] * weights[:, :, None]).sum(axis=1)

    def _get_weighted_coords_neighbors(self, coords):
        if len(coords) == 0:
            return np.empty(0, dtype=object)
        indices, weights = self._get_neighbors_weights(coords)
        nearest_coords = self.coords[indices]
        non_zero_mask = ~np.isclose(weights, 0)
        weighted_coords = [{tuple(coord): weight for coord, weight in zip(i_coords[i_mask], i_weights[i_mask])}
                           for i_coords, i_weights, i_mask in zip(nearest_coords, weights, non_zero_mask)]
        return np.array(weighted_coords, dtype=object)

    def _get_radius_weights(self, coords):
        dist, indices = self.nearest_interpolator.radius_neighbors(coords, return_distance=True)
        use_neighbors_mask = np.array([len(ix) == 0 for ix in indices])
        dist = dist[~use_neighbors_mask]
        indices = indices[~use_neighbors_mask]
        coords_weights = np.empty(np.sum(~use_neighbors_mask), dtype=object)
        for i, (d, ix) in enumerate(zip(dist, indices)):
            weights = self.dist_transform(d)
            zero_mask = np.isclose(weights, 0)
            weights[zero_mask] = 1
            weights = 1 / weights
            if zero_mask.any():
                weights[~zero_mask] = 0
            coords_weights[i] = weights / weights.sum()
        return indices, coords_weights, use_neighbors_mask

    def _interpolate_radius(self, coords):
        indices, coords_weights, use_neighbors_mask = self._get_radius_weights(coords)
        values = np.empty((len(coords), self.values.shape[1]), dtype=self.values.dtype)
        values[use_neighbors_mask] = self._interpolate_neighbors(coords[use_neighbors_mask])
        values[~use_neighbors_mask] = [(self.values[ix] * weights[:, None]).sum(axis=0)
                                       for ix, weights in zip(indices, coords_weights)]
        return values

    def _get_weighted_coords_radius(self, coords):
        indices, coords_weights, use_neighbors_mask = self._get_radius_weights(coords)
        weighted_coords = np.empty(len(indices), dtype=object)
        for i, (ix, weights) in enumerate(zip(indices, coords_weights)):
            non_zero_mask = ~np.isclose(weights, 0)
            weighted_coords[i] = {tuple(coord): weight for coord, weight in zip(self.coords[ix[non_zero_mask]], weights[non_zero_mask])}

        values = np.empty(len(coords), dtype=object)
        values[use_neighbors_mask] = self._get_weighted_coords_neighbors(coords[use_neighbors_mask])
        values[~use_neighbors_mask] = weighted_coords
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
    def __init__(self, coords, values=None, n_neighbors=3, dist_transform=2):
        super().__init__(coords, values)
        self.coords_hull = cv2.convexHull(self.coords, returnPoints=True)
        self.idw_interpolator = IDWInterpolator(coords, values, n_neighbors=n_neighbors, dist_transform=dist_transform)

        try:
            self.tri = Delaunay(self.coords, incremental=False)
        except QhullError:
            # Create artificial coordinates in the corners of given coordinate grid in order for Delaunay to
            # work with a full rank matrix of coordinates
            min_x, min_y = np.min(self.coords, axis=0) - 1
            max_x, max_y = np.max(self.coords, axis=0) + 1
            corner_coords = [(min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y)]
            self.tri = Delaunay(np.concatenate([self.coords, corner_coords]), incremental=False)
            if self.values is not None:
                self.values = np.concatenate([self.values, np.zeros((4, self.values.shape[1]), dtype=self.values.dtype)])

        # Perform the first auxiliary call of the tri for it to work properly in different processes.
        # Otherwise interpolation may fail if called in a pipeline with prefetch with mpc target.
        _ = self.tri.find_simplex((0, 0))

    def _is_in_hull(self, coords):
        # Cast both coords to pure python types for latest versions of opencv to work correctly
        return cv2.pointPolygonTest(self.coords_hull, np.array(coords).tolist(), measureDist=False) >= 0

    def _interpolate_inside_hull(self, coords):
        raise NotImplementedError

    def _interpolate(self, coords):
        inside_hull_mask = np.array([self._is_in_hull(coord) for coord in coords])
        values = np.empty((len(coords), self.values.shape[1]), dtype=self.values.dtype)
        values[inside_hull_mask] = self._interpolate_inside_hull(coords[inside_hull_mask])
        values[~inside_hull_mask] = self.idw_interpolator._interpolate(coords[~inside_hull_mask])
        return values


class DelaunayInterpolator(BaseDelaunayInterpolator, CoordsOnlyInterpolator):
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
        weights[~inside_hull_mask] = self.idw_interpolator._get_weighted_coords(coords[~inside_hull_mask])

        simplices_indices, bar_coords = self._get_simplex_info(coords[inside_hull_mask])
        simplices_coords = self.coords[simplices_indices]
        non_zero_mask = ~np.isclose(bar_coords, 0)
        weights[inside_hull_mask] = [{tuple(point): weight for point, weight in zip(simplices[mask], weights[mask])}
                                     for simplices, weights, mask in zip(simplices_coords, bar_coords, non_zero_mask)]
        return weights


class CloughTocherInterpolator(BaseDelaunayInterpolator):
    def __init__(self, coords, values, n_neighbors=3, dist_transform=2, **kwargs):
        super().__init__(coords, values, n_neighbors=n_neighbors, dist_transform=dist_transform)
        self.ct_interpolator = interpolate.CloughTocher2DInterpolator(self.tri, self.values, **kwargs)

    def _interpolate_inside_hull(self, coords):
        return self.ct_interpolator(coords)


class RBFInterpolator(SpatialInterpolator):
    def __init__(self, coords, values, **kwargs):
        super().__init__(coords, values)
        self.rbf_interpolator = interpolate.RBFInterpolator(self.coords, self.values, **kwargs)

    def _interpolate(self, coords):
        return self.rbf_interpolator(coords)
