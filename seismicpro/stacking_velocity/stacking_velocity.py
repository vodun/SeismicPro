"""Implements a StackingVelocity class which allows for velocity interpolation at given times"""

import numpy as np

from .velocity_model import calculate_stacking_velocity
from ..muter import Muter
from ..decorators import batch_method
from ..utils import VFUNC


class StackingVelocity(VFUNC):
    """A class representing stacking velocity at a certain point of a field.

    Stacking velocity is the value of the seismic velocity obtained from the best fit of the traveltime curve by a
    hyperbola for each timestamp. It is used to correct the arrival times of reflection events in the traces for their
    varying offsets prior to stacking.

    It can be instantiated directly by passing arrays of times and velocities defining knots of a piecewise linear
    velocity function or created from other types of data by calling a corresponding `classmethod`:
    * `from_file` - from a file in VFUNC format with time-velocity pairs,
    * `from_constant_velocity` - from a single velocity returned for all times,
    * `from_stacking_velocities` - from other stacking velocities with given weights.

    However, usually a stacking velocity instance is not created directly, but is obtained as a result of calling the
    following methods:
    * :func:`~velocity_spectrum.VerticalVelocitySpectrum.calculate_stacking_velocity` - to run an automatic algorithm
      for stacking velocity computation by vertical velocity spectrum,
    * :func:`StackingVelocityField.__call__` - to interpolate a stacking velocity at passed field coordinates given a
      created or loaded velocity field.

    The resulting object is callable and returns stacking velocities for given times.

    Examples
    --------
    Stacking velocity can be automatically calculated for a CDP gather by its velocity spectrum:
    >>> survey = Survey(path, header_index=["INLINE_3D", "CROSSLINE_3D"], header_cols="offset")
    >>> gather = survey.sample_gather().sort(by="offset")
    >>> velocity_spectrum = gather.calculate_vertical_velocity_spectrum()
    >>> velocity = velocity_spectrum.calculate_stacking_velocity()

    Or it can be interpolated from a velocity field (loaded from a file in this case):
    >>> field = StackingVelocityField.from_file(field_path).create_interpolator("idw")
    >>> coords = (inline, crossline)
    >>> velocity = field(coords)

    Parameters
    ----------
    times : 1d array-like
        An array with time values for which stacking velocity was picked. Measured in milliseconds.
    velocities : 1d array-like
        An array with stacking velocity values, matching the length of `times`. Measured in meters/seconds.
    coords : Coordinates or None, optional, defaults to None
        Spatial coordinates of the stacking velocity. If not given, the created instance won't be able to be added to a
        `StackingVelocityField`.

    Attributes
    ----------
    data_x : 1d np.ndarray
        An array with time values for which stacking velocity was picked. Measured in milliseconds.
    data_y : 1d np.ndarray
        An array with stacking velocity values, matching the length of `data_x`. Measured in meters/seconds.
    interpolator : callable
        An interpolator returning velocity value by given time.
    coords : Coordinates or None
        Spatial coordinates of the stacking velocity.
    """
    def __init__(self, times, velocities, coords=None):
        super().__init__(times, velocities, coords=coords)

    @property
    def times(self):
        """1d np.ndarray: An array with time values for which stacking velocity was picked. Measured in
        milliseconds."""
        return self.data_x

    @property
    def velocities(self):
        """1d np.ndarray: An array with stacking velocity values, matching the length of `times`. Measured in
        meters/seconds."""
        return self.data_y

    def validate_data(self):
        """Validate whether `times` and `velocities` are 1d arrays of the same shape and all stacking velocities are
        positive."""
        super().validate_data()
        if (self.velocities < 0).any():
            raise ValueError("Velocity values must be positive")

    @classmethod
    def from_stacking_velocities(cls, velocities, weights=None, coords=None):
        """Init stacking velocity by averaging other stacking velocities with given weights.

        Parameters
        ----------
        velocities : StackingVelocity or list of StackingVelocity
            Stacking velocities to be aggregated.
        weights : float or list of floats, optional
            Weight of each item in `velocities`. Normalized to have sum of 1 before aggregation. If not given, equal
            weights are assigned to all items and thus mean stacking velocity is calculated.
        coords : Coordinates, optional
            Spatial coordinates of the created stacking velocity. If not given, the created instance won't be able to
            be added to a `StackingVelocityField`.

        Returns
        -------
        self : StackingVelocity
            Created stacking velocity instance.
        """
        return cls.from_vfuncs(velocities, weights, coords)

    @classmethod
    def from_constant_velocity(cls, velocity, coords=None):
        """Init stacking velocity from a single velocity returned for all times.

        Parameters
        ----------
        velocity : float
            Stacking velocity returned for all times.
        coords : Coordinates, optional
            Spatial coordinates of the created stacking velocity. If not given, the created instance won't be able to
            be added to a `StackingVelocityField`.

        Returns
        -------
        self : StackingVelocity
            Created stacking velocity instance.
        """
        return cls([0, 10000], [velocity, velocity], coords=coords)

    @classmethod
    def from_vertical_velocity_spectrum(cls, spectrum, init=None, bounds=None, relative_margin=0.2,
                                        acceleration_bounds="auto", times_step=100, max_offset=5000,
                                        hodograph_correction_step=10, max_n_skips=2):
        kwargs = {"init": init, "bounds": bounds, "relative_margin": relative_margin,
                  "acceleration_bounds": acceleration_bounds, "times_step": times_step, "max_offset": max_offset,
                  "hodograph_correction_step": hodograph_correction_step, "max_n_skips": max_n_skips}
        times, velocities = calculate_stacking_velocity(spectrum, **kwargs)
        return cls(times, velocities, coords=spectrum.coords)

    @batch_method(target="for", copy_src=False)
    def create_muter(self, max_stretch_factor=0.65):
        """ Create a muter from a stacking velocity.
        This muter is supposed to attenuate the effect of waveform stretching after the nmo correction.

        Parameters
        ----------
        max_stretch_factor : float, defaults to 0.65
            Maximum allowed stretch factor.

        Returns
        -------
        self : Muter
            Created muter.
        """
        return Muter.from_stacking_velocity(self, max_stretch_factor=max_stretch_factor)

    def __call__(self, times):
        """Return stacking velocities for given `times`.

        Parameters
        ----------
        times : 1d array-like
            An array with time values. Measured in milliseconds.

        Returns
        -------
        velocities : 1d np.ndarray
            An array with stacking velocity values, matching the length of `times`. Measured in meters/seconds.
        """
        return np.maximum(super().__call__(times), 0)
