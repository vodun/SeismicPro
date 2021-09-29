"""Implements classes for velocity analysis: StackingVelocity and VelocityCube"""

import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from numba import njit, prange
from scipy.spatial.qhull import Delaunay, QhullError  #pylint: disable=no-name-in-module
from sklearn.neighbors import NearestNeighbors

from .metrics import MetricsMap
from .utils import to_list, read_vfunc, read_single_vfunc, dump_vfunc, velocity_qc
from .utils.interpolation import interp1d


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


class StackingVelocity:
    """A class representing stacking velocity in a certain place of a field.

    Stacking velocity is the value of the seismic velocity obtained from the best fit of the traveltime curve by a
    hyperbola for each timestamp. It is used to correct the arrival times of reflection events in the traces for their
    varying offsets prior to stacking.

    It can be created from three different types of data by calling a corresponding `classmethod`:
    * `from_points` - create a stacking velocity from 1d arrays of times and velocities,
    * `from_file` - create a stacking velocity from a file in VFUNC format with time-velocity pairs,
    * `from_interpolator` - create a stacking velocity from a callable that returns velocity value by given time.

    However, usually a stacking velocity instance is not created directly, but is obtained as a result of calling the
    following methods:
    * :func:`~semblance.Semblance.calculate_stacking_velocity` - to run an automatic algorithm for stacking velocity
      computation,
    * :func:`VelocityCube.__call__` - to interpolate a stacking velocity at passed field coordinates given a created or
      loaded velocity cube.

    The resulting object is callable and returns stacking velocities for given times.

    Examples
    --------
    Stacking velocity can be automatically calculated for a CDP gather by its semblance:
    >>> survey = Survey(path, header_index=["INLINE_3D", "CROSSLINE_3D"], header_cols="offset")
    >>> gather = survey.sample_gather().sort(by="offset")
    >>> semblance = gather.calculate_semblance(velocities=np.linspace(1400, 5000, 200), win_size=8)
    >>> velocity = semblance.calculate_stacking_velocity()

    Or it can be interpolated from a velocity cube (loaded from a file in our case):
    >>> cube = VelocityCube(path=cube_path).create_interpolator()
    >>> velocity = cube(inline, crossline)

    Attributes
    ----------
    interpolator : callable
        An interpolator returning velocity value by given time.
    times : 1d array-like
        An array with time values for which stacking velocity was picked. Measured in milliseconds. Can be `None` if
        the instance was created from interpolator.
    velocities : 1d array-like
        An array with stacking velocity values, matching the length of `times`. Measured in meters/seconds. Can be
        `None` if the instance was created from interpolator.
    inline : int
        An inline of the stacking velocity. If `None`, the created instance won't be able to be added to a
        `VelocityCube`.
    crossline : int
        A crossline of the stacking velocity. If `None`, the created instance won't be able to be added to a
        `VelocityCube`.
    """
    def __init__(self):
        self.interpolator = lambda times: np.zeros_like(times, dtype=np.float32)
        self.times = None
        self.velocities = None
        self.inline = None
        self.crossline = None

    @classmethod
    def from_points(cls, times, velocities, inline=None, crossline=None):
        """Init stacking velocity from arrays of times and corresponding velocities.

        The resulting object performs linear velocity interpolation between points given. Linear extrapolation is
        performed outside of the defined times range.

        Parameters
        ----------
        times : 1d array-like
            An array with time values for which stacking velocity was picked. Measured in milliseconds.
        velocities : 1d array-like
            An array with stacking velocity values, matching the length of `times`. Measured in meters/seconds.
        inline : int, optional, defaults to None
            An inline of the created stacking velocity. If `None`, the created instance won't be able to be added to a
            `VelocityCube`.
        crossline : int, optional, defaults to None
            A crossline of the created stacking velocity. If `None`, the created instance won't be able to be added to
            a `VelocityCube`.

        Returns
        -------
        self : StackingVelocity
            Created stacking velocity instance.

        Raises
        ------
        ValueError
            If shapes of `times` and `velocities` are inconsistent or some velocity values are negative.
        """
        times = np.array(times, dtype=np.float32)
        velocities = np.array(velocities, dtype=np.float32)
        if times.ndim != 1 or times.shape != velocities.shape:
            raise ValueError("Inconsistent shapes of times and velocities")
        if (velocities < 0).any():
            raise ValueError("Velocity values must be positive")
        self = cls.from_interpolator(interp1d(times, velocities), inline, crossline)
        self.times = times
        self.velocities = velocities
        return self

    @classmethod
    def from_file(cls, path):
        """Init stacking velocity from a file with vertical functions in Paradigm Echos VFUNC format.

        The file must have exactly one record with the following structure:
        VFUNC [inline] [crossline]
        [time_1] [velocity_1] [time_2] [velocity_2] ... [time_n] [velocity_n]

        The resulting object performs linear velocity interpolation between points given. Linear extrapolation is
        performed outside of the defined times range.

        Parameters
        ----------
        path : str
            A path to the source file.

        Returns
        -------
        self : StackingVelocity
            Loaded stacking velocity instance.
        """
        inline, crossline, times, velocities = read_single_vfunc(path)
        return cls.from_points(times, velocities, inline, crossline)

    @classmethod
    def from_interpolator(cls, interpolator, inline=None, crossline=None):
        """Init stacking velocity from velocity interpolator.

        Parameters
        ----------
        interpolator : callable
            An interpolator returning velocity value by given time.
        inline : int, optional, defaults to None
            An inline of the created stacking velocity. If `None`, the created instance won't be able to be added to a
            `VelocityCube`.
        crossline : int, optional, defaults to None
            A crossline of the created stacking velocity. If `None`, the created instance won't be able to be added to
            a `VelocityCube`.

        Returns
        -------
        self : StackingVelocity
            Created stacking velocity instance.
        """
        self = cls()
        self.interpolator = interpolator
        self.inline = inline
        self.crossline = crossline
        return self

    def dump(self, path):
        """Dump stacking velocities to a file in VFUNC format.

        Notes
        -----
        See more about the format in :func:`~utils.file_utils.dump_vfunc`.

        Parameters
        ----------
        path : str
            A path to the created file.
        """
        if not self.has_coords or not self.has_points:
            raise ValueError("StackingVelocity instance can be dumped only if it was created from time and velocity "
                             "pairs with not-None inline and crossline")
        dump_vfunc(path, [(self.inline, self.crossline, self.times, self.velocities)])

    @property
    def has_points(self):
        """bool: Whether the instance was created from time and velocity pairs."""
        return (self.times is not None) and (self.velocities is not None)

    @property
    def has_coords(self):
        """bool: Whether stacking velocity inline and crossline are not-None."""
        return (self.inline is not None) and (self.crossline is not None)

    def get_coords(self):
        """Get spatial coordinates of the stacking velocity.

        Returns
        -------
        coords : tuple with 2 elements
            Stacking velocity spatial coordinates.
        """
        return (self.inline, self.crossline)

    def __call__(self, times):
        """Return stacking velocities for given `times`.

        Parameters
        ----------
        times : 1d array-like
            An array with time values. Measured in milliseconds.

        Returns
        -------
        velocities : 1d array-like
            An array with stacking velocity values, matching the length of `times`. Measured in meters/seconds.
        """
        return np.maximum(self.interpolator(times), 0)


class VelocityCube:
    """A class for storing and interpolating stacking velocity data over a field.

    Velocities used for seismic cube stacking are usually picked on a sparse grid of inlines and crosslines and then
    interpolated over the whole field in order to reduce computational costs. Such interpolation can be performed by
    `VelocityCube` class which provides an interface to obtain a stacking velocity at given spatial coordinates via its
    `__call__` method. The cube can either be loaded from a file of vertical functions or created empty and iteratively
    updated with calculated stacking velocities.

    After all velocities are added, velocity interpolator should be created. It can be done either manually by
    calling :func:`~VelocityCube.create_interpolator` method or automatically during the first call to the cube. Manual
    interpolator creation is useful when the cube should be passed to different proccesses (e.g. in a pipeline with
    prefetch with `mpc` target) since otherwise the interpolator will be independently created in all the processes.

    Examples
    --------
    The cube can either be loaded from a file:
    >>> cube = VelocityCube(path=cube_path)

    Or created empty and updated with instances of `StackingVelocity` class:
    >>> cube = VelocityCube()
    >>> velocity = StackingVelocity.from_points(times=[0, 1000, 2000, 3000], velocities=[1500, 2000, 2800, 3400],
    ...                                         inline=20, crossline=40)
    >>> cube.update(velocity)

    Cube creation should be finalized with `create_interpolator` method call:
    >>> cube.create_interpolator()

    Parameters
    ----------
    path : str, optional
        A path to the source file with vertical functions to load the cube from. If not given, an empty cube is
        created.
    create_interpolator : bool, optional, defaults to True
        Whether to create an interpolator immediately if the cube was loaded from a file.

    Attributes
    ----------
    stacking_velocities_dict : dict
        A dict of stacking velocities in the cube whose keys are tuples with their spatial coordinates and values are
        the instances themselves.
    interpolator : VelocityInterpolator
        Velocity interpolator over the field.
    is_dirty_interpolator : bool
        Whether the cube was updated after the interpolator was created.
    """
    def __init__(self, path=None, create_interpolator=True):
        self.stacking_velocities_dict = {}
        self.interpolator = None
        self.is_dirty_interpolator = True
        if path is not None:
            self.load(path)
            if create_interpolator:
                self.create_interpolator()

    @property
    def has_interpolator(self):
        """bool: Whether the velocity interpolator was created."""
        return self.interpolator is not None

    def load(self, path):
        """Load a velocity cube from a file with vertical functions in Paradigm Echos VFUNC format.

        The file may have one or more records with the following structure:
        VFUNC [inline] [crossline]
        [time_1] [velocity_1] [time_2] [velocity_2] ... [time_n] [velocity_n]

        Parameters
        ----------
        path : str
            A path to the source file.

        Returns
        -------
        self : VelocityCube
            Self with loaded stacking velocities. Changes `stacking_velocities_dict` inplace and sets the
            `is_dirty_interpolator` flag to `True`.
        """
        for inline, crossline, times, velocities in read_vfunc(path):
            stacking_velocity = StackingVelocity.from_points(times, velocities, inline, crossline)
            self.stacking_velocities_dict[(inline, crossline)] = stacking_velocity
        self.is_dirty_interpolator = True
        return self

    def dump(self, path):
        """Dump all the vertical functions of the cube to a file in VFUNC format.

        Notes
        -----
        See more about the format in :func:`~utils.file_utils.dump_vfunc`.

        Parameters
        ----------
        path : str
            A path to the created file.
        """
        vfunc_list = []
        for (inline, crossline), stacking_velocity in self.stacking_velocities_dict.items():
            vfunc_list.append((inline, crossline, stacking_velocity.times, stacking_velocity.velocities))
        dump_vfunc(path, vfunc_list)

    def update(self, stacking_velocities):
        """Update a velocity cube with given stacking velocities.

        Notes
        -----
        All passed `StackingVelocity` instances must have not-None coordinates.

        Parameters
        ----------
        stacking_velocities : StackingVelocity or list of StackingVelocity
            Stacking velocities to update the cube with.

        Returns
        -------
        self : VelocityCube
            Self with loaded stacking velocities. Changes `stacking_velocities_dict` inplace and sets the
            `is_dirty_interpolator` flag to `True` if passed `stacking_velocities` is not empty.

        Raises
        ------
        TypeError
            If wrong type of `stacking_velocities` was passed.
        ValueError
            If any of the passed stacking velocities has `None` coordinates.
        """
        stacking_velocities = to_list(stacking_velocities)
        if not all(isinstance(vel, StackingVelocity) for vel in stacking_velocities):
            raise TypeError("The cube can be updated only with `StackingVelocity` instances")
        if not all(vel.has_coords for vel in stacking_velocities):
            raise ValueError("All passed `StackingVelocity` instances must have not-None coordinates")
        for vel in stacking_velocities:
            self.stacking_velocities_dict[vel.get_coords()] = vel
        if stacking_velocities:
            self.is_dirty_interpolator = True
        return self

    def create_interpolator(self):
        """Create velocity interpolator from stacking velocities in the cube.

        Returns
        -------
        self : VelocityCube
            Self with created interploator. Updates `interpolator` inplace and sets the `is_dirty_interpolator` flag
            to `False`.

        Raises
        ------
        ValueError
            If velocity cube is empty.
        """
        if not self.stacking_velocities_dict:
            raise ValueError("No stacking velocities passed")
        self.interpolator = VelocityInterpolator(self.stacking_velocities_dict)
        self.is_dirty_interpolator = False
        return self

    def __call__(self, inline, crossline, create_interpolator=True):
        """Interpolate stacking velocity at given `inline` and `crossline`.

        Parameters
        ----------
        inline : int
            An inline to interpolate stacking velocity at.
        crossline : int
            A crossline to interpolate stacking velocity at.
        create_interpolator : bool, optional, defaults to True
            Whether to create a velocity interpolator if it does not exist.

        Returns
        -------
        stacking_velocity : StackingVelocity
            Interpolated stacking velocity at (`inline`, `crossline`).

        Raises
        ------
        ValueError
            If velocity interpolator does not exist and `create_interpolator` flag is set to `False`.
        """
        if create_interpolator and (not self.has_interpolator or self.is_dirty_interpolator):
            self.create_interpolator()
        elif not create_interpolator:
            if not self.has_interpolator:
                raise ValueError("Velocity interpolator must be created first")
            if self.is_dirty_interpolator:
                warnings.warn("Dirty interpolator is being used", RuntimeWarning)
        return self.interpolator(inline, crossline)

    def qc(self, win_radius, times, coords=None, metrics_names=None, n_workers=None):
        if metrics_names is None:
            metrics_names = velocity_qc.VELOCITY_QC_METRICS
        metrics_funcs = []
        for name in to_list(metrics_names):
            if name in velocity_qc.VELOCITY_QC_METRICS:
                func = getattr(velocity_qc, name)
            elif callable(name):
                if name.__name__ == "<lambda>":
                    raise ValueError("Lambda expressions are not allowed")
                func = name
            else:
                raise ValueError(f"Unknown metric {name}")
            metrics_funcs.append(func)

        # Calculate stacking velocities at given times for each of coords
        if coords is None:
            coords = list(self.stacking_velocities_dict.keys())
        coords = np.array(coords)
        if not self.has_interpolator:
            self.create_interpolator()
        velocities = self.interpolator.interpolate(coords, np.array(times))

        # Select all neighbouring stacking velocities for each of coords
        coords_knn = NearestNeighbors(radius=win_radius).fit(coords)
        _, windows_indices = coords_knn.radius_neighbors(coords, return_distance=True, sort_results=True)

        # Calculate requested metrics
        def calculate_window_metrics(window_indices):
            window_velocities = velocities[window_indices]
            return [func(window_velocities) for func in metrics_funcs]

        if n_workers is None:
            n_workers = os.cpu_count()
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            metrics = executor.map(calculate_window_metrics, windows_indices)
        metrics = {func.__name__: np.array(val) for func, val in zip(metrics_funcs, zip(*metrics))}
        return MetricsMap(coords, **metrics)
