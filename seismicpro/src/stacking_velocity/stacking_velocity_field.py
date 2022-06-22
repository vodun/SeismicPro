"""Implements a VelocityCube class which stores stacking velocities calculated at different field locations and allows
for spatial velocity interpolation"""

import os
from functools import reduce

import numpy as np
from tqdm.contrib.concurrent import thread_map
from sklearn.neighbors import NearestNeighbors

from .stacking_velocity import StackingVelocity
from .metrics import VELOCITY_QC_METRICS, StackingVelocityMetric
from ..field import ValuesAgnosticField, VFUNCMixin
from ..utils import to_list


class StackingVelocityField(ValuesAgnosticField, VFUNCMixin):
    """A class for storing and interpolating stacking velocity data over a field.

    Velocities used for seismic cube stacking are usually picked on a sparse grid of inlines and crosslines and then
    interpolated over the whole field in order to reduce computational costs. Such interpolation can be performed by
    `VelocityCube` class which provides an interface to obtain a stacking velocity at given spatial coordinates via its
    `__call__` method. The cube can either be loaded from a file of vertical functions or created empty and iteratively
    updated with calculated stacking velocities.

    After all velocities are added, velocity interpolator should be created. It can be done either manually by
    calling :func:`~VelocityCube.create_interpolator` method or automatically during the first call to the cube. Manual
    interpolator creation is useful when the cube should be passed to different processes (e.g. in a pipeline with
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

    Cube creation must be finalized with `create_interpolator` method call:
    >>> cube.create_interpolator("idw")

    Now the field allows for velocity interpolation at given coordinates:
    >>> ...

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
    item_class = StackingVelocity

    def construct_item(self, base_items, weights, coords):
        return self.item_class.from_stacking_velocities(base_items, weights, coords=coords)

    def get_mean_velocity(self):
        return self.item_class.from_stacking_velocities(list(self.item_container.values()))

    def interpolate(self, coords, times):
        self.validate_interpolator()
        field_coords, _, _ = self.transform_coords(coords)
        times = np.atleast_1d(times)
        weighted_coords = self.interpolator.get_weighted_coords(field_coords)
        base_velocities_coords = set.union(*[set(coord_weights.keys()) for coord_weights in weighted_coords])
        base_velocities = {coords: self.item_container[coords](times) for coords in base_velocities_coords}

        res = np.zeros((len(field_coords), len(times)))
        for i, coord_weights in enumerate(weighted_coords):
            for coord, weight in coord_weights.items():
                res[i] += base_velocities[coord] * weight
        return res

    def qc(self, radius, metrics=None, coords=None, times=None, n_workers=None, bar=True):
        """Perform quality control of the velocity field by calculating spatial-window-based metrics for its stacking
        velocities evaluated at given `coords` and `times`.

        If `coords` are not given, coordinates of items in the field are used. If `times` are not given, samples of the
        underlying `Survey` are used if it is defined.

        By default, the following metrics are calculated:
        * Presence of segments with velocity decrease in time,
        * Maximal deviation of instantaneous acceleration from the mean acceleration over all times,
        * Maximal spatial velocity standard deviation in a window over all times,
        * Maximal absolute relative difference between central stacking velocity and the average of all remaining
          velocities in the window over all times.

        Parameters
        ----------
        radius : positive float
            Spatial window radius (Euclidean distance).
        metrics : StackingVelocityMetric or list of StackingVelocityMetric, optional
            Metrics to calculate. Defaults to those defined in `~metrics.VELOCITY_QC_METRICS`.
        coords : 2d np.array or list of Coordinates or None, optional
            Spatial coordinates of stacking velocities to calculate metrics for. If not given, coordinates of items in
            the field are used.
        times : 1d array-like
            Times to calculate metrics for. Measured in milliseconds.
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

        # Set default coords and times
        if coords is None:
            coords = self.coords
        if times is None:
            if not self.has_survey:
                raise ValueError("times must be passed if the field is not linked with a survey")
            times = self.survey.times

        # Calculate stacking velocities at given times for each of coords
        velocities = self.interpolate(coords, times)

        # Select all neighboring stacking velocities for each of coords
        if n_workers is None:
            n_workers = os.cpu_count()
        coords_neighbors = NearestNeighbors(radius=radius, n_jobs=n_workers).fit(coords)
        # Sort results to guarantee that central stacking velocity of each window will have index 0
        _, windows_indices = coords_neighbors.radius_neighbors(coords, return_distance=True, sort_results=True)

        # Initialize metrics and calculate them
        def calc_metrics(window_indices):
            window = velocities[window_indices]
            return [metric.calc(window if metric.is_window_metric else window[0], times) for metric in metrics]

        metrics = [metric(times, velocities, coords_neighbors) for metric in metrics]
        results = thread_map(calc_metrics, windows_indices, max_workers=n_workers,
                             desc="Coordinates processed", disable=not bar)
        metrics_maps = [metric.map_class(coords, metric_values, coords_cols=self.coords_cols, metric=metric)
                        for metric, metric_values in zip(metrics, zip(*results))]
        if is_single_metric:
            return metrics_maps[0]
        return metrics_maps
