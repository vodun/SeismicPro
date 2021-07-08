"""Implements Muter class"""
import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

from .utils import read_single_vfunc


class Muter:
    """A class contains various approaches for muting operation.

    A muter can be created from three different types of data by calling a corresponding classmethod:
    * `from_points` - create muter from the 1d arrays of times and offsets,
    * `from_file` - create muter from the file muting points,
    * `from_first_breaks` - create muter from the 1d arrays of times and offsets that represent first break points.

    Found muter returns the time of muting for given offsets and may be called via `__call__` method.

    Attributes
    ----------
    muter : callable
        Returns the times for the given offset value. The input of the callable must be numeric or 1d array-like.
    """
    def __init__(self):
        self.muter = lambda offsets: np.zeros_like(offsets)

    @classmethod
    def from_points(cls, offsets, times, fill_value="extrapolate"):
        """ Create muter callable from the 1d arrays of times and offsets.

        Resulted muter is a 1d interpolater, which interpolates time values between given points and extrapolates
        values outside of the offsets' range.

        Parameters
        ----------
        offsets : 1d array-like
            Array with offset values. Measured in meters.
        times : 1d array-like
            Array with muting times. Measured in milliseconds.
        fill_value : str, optional, by default "extrapolate"
            Argument from `scipy.interp1d`.

        Returns
        -------
        self : Muter
            Muter with 1d interpolator.
        """
        self = cls()
        self.muter = interp1d(offsets, times, fill_value=fill_value)
        return self

    @classmethod
    def from_file(cls, path, **kwargs):
        """ Create a muter from the file with VFUNC format.

        See more about the format in :func:`~utils.file_utils.read_single_vfunc`. This function returns times and
        offsets and redirect call to :func:`.from_points`.

        Parameters
        ----------
        path : str
            Path to the file with muting.

        Returns
        -------
        self : Muter
            Muter with 1d interpolator.
        """
        _, _, offsets, times = read_single_vfunc(path)
        return cls.from_points(offsets, times, **kwargs)

    @classmethod
    def from_first_breaks(cls, offsets, times, velocity_reduction=0):
        """ Create muter from the 1d arrays of times and offsets that represent to first break points.

        This muter uses first break times to emulate velocity of weathering layer using linear regression.

        Parameters
        ----------
        offsets : 1d array-like
            An array with offset values. Measured in meters.
        times : 1d array-like
            An array with muting times. Measured in milliseconds.
        velocity_reduction : int, optional, by default 0
            A number for shifting the found velocity of weathering layer. If positive, the velocity will be reduced by
            given value, otherwise increased. Measured in meters/seconds.

        Returns
        -------
        self : Muter
            Muter with callable that represents a velocity of weathering layer.
        """
        velocity_reduction = velocity_reduction / 1000  # from m/s to m/ms
        lin_reg = LinearRegression(fit_intercept=True)
        lin_reg.fit(np.array(times).reshape(-1, 1), np.array(offsets))

        # The fitted velocity is reduced by velocity_reduction in order to mute amplitudes near first breaks
        intercept = lin_reg.intercept_
        velocity = lin_reg.coef_ - velocity_reduction

        self = cls()
        self.muter = lambda offsets: (offsets - intercept) / velocity
        return self

    def __call__(self, offsets):
        """Returns times for given offsets based on found muter.

        Notes
        -----
        If muter is not calculated, a zero will be returned for every offset.

        Parameters
        ----------
         offsets : 1d array-like
            Array with offset values. Measured in meters.

        Returns
        -------
        times : 1d array-like
            Array with muting times. Measured in milliseconds.
        """
        return self.muter(offsets)
