"""Implements a StackingVelocity class which allows for velocity interpolation at given times"""

import numpy as np

from ..utils.interpolation import interp1d
from ..utils import to_list, read_single_vfunc, dump_vfunc, Coordinates


class StackingVelocity:
    """A class representing stacking velocity at a certain point of a field.

    Stacking velocity is the value of the seismic velocity obtained from the best fit of the traveltime curve by a
    hyperbola for each timestamp. It is used to correct the arrival times of reflection events in the traces for their
    varying offsets prior to stacking.

    It can be created from four different types of data by calling a corresponding `classmethod`:
    * `from_points` - from 1d arrays of times and velocities,
    * `from_file` - from a file in VFUNC format with time-velocity pairs,
    * `from_weighted_instances` - from other stacking velocities with given weights,
    * `from_constant_velocity` - from a single value which will be returned for all times.

    However, usually a stacking velocity instance is not created directly, but is obtained as a result of calling the
    following methods:
    * :func:`~semblance.Semblance.calculate_stacking_velocity` - to run an automatic algorithm for stacking velocity
      computation by vertical velocity semblance,
    * :func:`StackingVelocityField.__call__` - to interpolate a stacking velocity at passed field coordinates given a
      created or loaded velocity cube.

    The resulting object is callable and returns stacking velocities for given times.

    Examples
    --------
    Stacking velocity can be automatically calculated for a CDP gather by its semblance:
    >>> survey = Survey(path, header_index=["INLINE_3D", "CROSSLINE_3D"], header_cols="offset")
    >>> gather = survey.sample_gather().sort(by="offset")
    >>> semblance = gather.calculate_semblance(velocities=np.linspace(1400, 5000, 200), win_size=8)
    >>> velocity = semblance.calculate_stacking_velocity()

    Or it can be interpolated from a velocity cube (loaded from a file in this case):
    >>> cube = VelocityCube(path=cube_path).create_interpolator()
    >>> velocity = cube(inline, crossline)

    Attributes
    ----------
    times : 1d array-like
        An array with time values for which stacking velocity was picked. Measured in milliseconds.
    velocities : 1d array-like
        An array with stacking velocity values, matching the length of `times`. Measured in meters/seconds.
    interpolator : callable
        An interpolator returning velocity value by given time.
    coords : Coordinates or None
        Spatial coordinates of the stacking velocity. If `None`, the created instance won't be able to be added to a
        `StackingVelocityField`.
    """
    def __init__(self):
        self.times = None
        self.velocities = None
        self.interpolator = None
        self.coords = None

    @classmethod
    def from_points(cls, times, velocities, coords=None):
        """Init stacking velocity from arrays of times and corresponding velocities.

        The resulting object performs linear velocity interpolation between given points. Linear extrapolation is
        performed outside of the defined times range.

        Parameters
        ----------
        times : 1d array-like
            An array with time values for which stacking velocity was picked. Measured in milliseconds.
        velocities : 1d array-like
            An array with stacking velocity values, matching the length of `times`. Measured in meters/seconds.
        coords : Coordinates or None, optional, defaults to None
            Spatial coordinates of the created stacking velocity. If `None`, the created instance won't be able to be
            added to a `StackingVelocityField`.

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

        self = cls()
        self.times = times
        self.velocities = velocities
        self.interpolator = interp1d(times, velocities)
        self.coords = coords
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
        *coords, times, velocities = read_single_vfunc(path)
        return cls.from_points(times, velocities, coords=Coordinates(coords, names=("INLINE_3D", "CROSSLINE_3D")))

    @classmethod
    def from_weighted_instances(cls, instances, weights=None, coords=None):
        """Init stacking velocity from other stacking velocities with given weights.

        Parameters
        ----------
        instances : ...
            ...
        weights : ...
            ...
        coords : Coordinates or None, optional, defaults to None
            Spatial coordinates of the created stacking velocity. If `None`, the created instance won't be able to be
            added to a `StackingVelocityField`.

        Returns
        -------
        self : StackingVelocity
            Created stacking velocity instance.
        """
        instances = to_list(instances)
        if weights is None:
            weights = np.ones_like(instances) / len(instances)
        weights = np.array(weights)
        times = np.unique(np.concatenate([inst.times for inst in instances]))
        velocities = (np.stack([inst(times) for inst in instances]) * weights[:, None]).sum(axis=0)
        return cls.from_points(times, velocities, coords=coords)

    @classmethod
    def from_constant_velocity(cls, velocity, coords=None):
        """Init stacking velocity from a single velocity returned for all times.

        Parameters
        ----------
        velocity : float
            Stacking velocity returned for all times.
        coords : Coordinates or None, optional, defaults to None
            Spatial coordinates of the created stacking velocity. If `None`, the created instance won't be able to be
            added to a `StackingVelocityField`.

        Returns
        -------
        self : StackingVelocity
            Created stacking velocity instance.
        """
        return cls.from_points([0, 10000], [velocity, velocity], coords=coords)

    @property
    def has_coords(self):
        """bool: Whether stacking velocity coordinates are not-None."""
        return self.coords is not None

    def dump(self, path):
        """Dump stacking velocity to a file in VFUNC format.

        Notes
        -----
        See more about the format in :func:`~utils.file_utils.dump_vfunc`.

        Parameters
        ----------
        path : str
            A path to the created file.
        """
        if not self.has_coords:
            raise ValueError("StackingVelocity instance can be dumped only if it has well-defined coordinates")
        dump_vfunc(path, [(*self.coords, self.times, self.velocities)])

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
        if self.interpolator is None:
            raise ValueError("StackingVelocity instance must be created by calling one of its from_* constructors")
        return np.maximum(self.interpolator(times), 0)
