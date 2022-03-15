"""Implements a StackingVelocity class which allows for velocity interpolation at given times"""

from collections import namedtuple

import numpy as np

from ..utils.interpolation import interp1d
from ..utils import read_single_vfunc, dump_vfunc


class StackingVelocity:
    """A class representing stacking velocity at a certain point of a field.

    Stacking velocity is the value of the seismic velocity obtained from the best fit of the traveltime curve by a
    hyperbola for each timestamp. It is used to correct the arrival times of reflection events in the traces for their
    varying offsets prior to stacking.

    It can be created from four different types of data by calling a corresponding `classmethod`:
    * `from_points` - create a stacking velocity from 1d arrays of times and velocities,
    * `from_file` - create a stacking velocity from a file in VFUNC format with time-velocity pairs,
    * `from_interpolator` - create a stacking velocity from a callable that returns velocity value by given time,
    * `from_constant_velocity` - create a stacking velocity which returns a single value for all times.

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

    Or it can be interpolated from a velocity cube (loaded from a file in this case):
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

    @classmethod
    def from_constant_velocity(cls, velocity, inline=None, crossline=None):
        """Init stacking velocity from a single velocity returned for all times.

        Parameters
        ----------
        velocity : float
            A single velocity returned for all times.
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
        interpolator = interp1d([0, 10000], [velocity, velocity])
        return cls.from_interpolator(interpolator, inline=inline, crossline=crossline)

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

    def get_coords(self, *args, **kwargs):
        """Get spatial coordinates of the stacking velocity.

        Ignores all passed arguments but accept them to preserve general `get_coords` interface.

        Returns
        -------
        coords : tuple with 2 elements
            Stacking velocity spatial coordinates.
        """
        _ = args, kwargs
        return namedtuple("Coordinates", ["INLINE_3D", "CROSSLINE_3D"])(self.inline, self.crossline)

    @property
    def coords(self):
        """namedtuple with 2 elements: Spatial coordinates of the stacking velocity."""
        return self.get_coords()

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
