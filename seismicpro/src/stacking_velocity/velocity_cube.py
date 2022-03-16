"""Implements a VelocityCube class which stores stacking velocities calculated at different field locations and allows
for spatial velocity interpolation"""

import os
import warnings

import numpy as np
from tqdm.contrib.concurrent import thread_map
from sklearn.neighbors import NearestNeighbors

from .stacking_velocity import StackingVelocity
from .velocity_interpolator import VelocityInterpolator
from .metrics import VELOCITY_QC_METRICS, StackingVelocityMetric
from ..utils import to_list, read_vfunc, dump_vfunc


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

    The cube provides an interface to quality control via its `qc` method, which calculates maps for several
    spatial-window-based metrics calculated for its stacking velocities evaluated at passed times. These maps may be
    interactively visualized to evaluate the cube quality in detail.

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

    Quality control can be performed by calling `qc` method and visualizing the resulting maps:
    >>> metrics_maps = cube.qc(win_radius=40, times=np.arange(0, 3000, 2))
    >>> for metric_map in metrics_maps:
    >>>     metric_map.plot(interactive=True)

    Parameters
    ----------
    path : str, optional
        A path to the source file with vertical functions to load the cube from. If not given, an empty cube is
        created.
    create_interpolator : bool, optional, defaults to True
        Whether to create an interpolator immediately if the cube is loaded from a file.

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
            self.stacking_velocities_dict[tuple(vel.coords)] = vel
        if stacking_velocities:
            self.is_dirty_interpolator = True
        return self

    def create_interpolator(self):
        """Create velocity interpolator from stacking velocities in the cube.

        Returns
        -------
        self : VelocityCube
            Self with created interpolator. Updates `interpolator` inplace and sets the `is_dirty_interpolator` flag to
            `False`.

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

    def qc(self, win_radius, times, coords=None, metrics=None, n_workers=None, bar=True): #pylint: disable=invalid-name
        """Perform quality control of the velocity cube by calculating spatial-window-based metrics for its stacking
        velocities evaluated at given `times`.

        If `coords` are specified, QC will be performed for stacking velocities, interpolated for each of them.
        Otherwise, stacking velocities stored in the cube are used directly.

        By default, the following metrics are calculated:
        * Presence of segments with velocity decrease in time,
        * Maximal deviation of instantaneous acceleration from the mean acceleration over all times,
        * Maximal spatial velocity standard deviation in a window over all times,
        * Maximal absolute relative difference between central stacking velocity and the average of all remaining
          velocities in the window over all times.

        Parameters
        ----------
        win_radius : positive float
            Spatial window radius (Euclidean distance).
        times : 1d array-like
            Times to calculate metrics for. Measured in milliseconds.
        coords : 2d np.array or None, optional
            Spatial coordinates of stacking velocities to calculate metrics for. If not given, stacking velocities
            stored in the cube are used without extra interpolation step.
        metrics : StackingVelocityMetric or list of StackingVelocityMetric, optional
            Metrics to calculate. Defaults to those defined in `~metrics.VELOCITY_QC_METRICS`.
        n_workers : int, optional
            The number of threads to be spawned to calculate metrics. Defaults to the number of cpu cores.
        bar : bool, optional, defaults to True
            Whether to show a progress bar.

        Returns
        -------
        metrics_maps : StackingVelocityMetricMap or list of StackingVelocityMetricMap
            Calculated metrics maps. Has the same shape as `metrics`.
        """
        is_single_metric = isinstance(metrics, type) and issubclass(metrics, StackingVelocityMetric)
        if metrics is None:
            metrics = VELOCITY_QC_METRICS
        metrics = to_list(metrics)
        if not metrics:
            raise ValueError("At least one metric should be passed")
        if not all(isinstance(metric, type) and issubclass(metric, StackingVelocityMetric) for metric in metrics):
            raise ValueError("All passed metrics must be subclasses of StackingVelocityMetric")

        # Calculate stacking velocities at given times for each of coords
        if not self.has_interpolator:
            self.create_interpolator()
        if coords is None:
            coords = list(self.stacking_velocities_dict.keys())
        coords = np.array(coords)
        times = np.array(times)
        velocities = self.interpolator.interpolate(coords, times)

        # Select all neighboring stacking velocities for each of coords
        if n_workers is None:
            n_workers = os.cpu_count()
        coords_neighbors = NearestNeighbors(radius=win_radius, n_jobs=n_workers).fit(coords)
        # Sort results to guarantee that central stacking velocity of each window will have index 0
        _, windows_indices = coords_neighbors.radius_neighbors(coords, return_distance=True, sort_results=True)

        # Initialize metrics and calculate them
        def calc_metrics(window_indices):
            window = velocities[window_indices]
            return [metric.calc(window if metric.is_window_metric else window[0], times) for metric in metrics]

        metrics = [metric(times, velocities, coords_neighbors) for metric in metrics]
        results = thread_map(calc_metrics, windows_indices, max_workers=n_workers,
                             desc="Coordinates processed", disable=not bar)
        coords_cols = ["INLINE_3D", "CROSSLINE_3D"]
        metrics_maps = [metric.map_class(coords, metric_values, coords_cols=coords_cols, metric=metric)
                        for metric, metric_values in zip(metrics, zip(*results))]
        if is_single_metric:
            return metrics_maps[0]
        return metrics_maps
