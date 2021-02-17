""" The file contains classes for velocity analysis. """
# pylint: disable=not-an-iterable
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from scipy.interpolate import interp1d
from matplotlib import colors as mcolors

from .plot_utils import _set_ticks


def use_docs_from(method_from):
    """ Decorator that adds the documentation from a `method_from` to `method_to`.

    Parameters
    ----------
    method_from : function or Class
    An instance to get documentation from.

    Returns
    -------
    decorator : callable
        Class decorator.
    """
    def decorator(method_to):
        """ Returned decorator. """
        from_name = method_from.__qualname__
        message = '|  For clarity, the docstrings of the `{}` are shown below.  |'.format(from_name)
        line = '\n' + '-' * len(message) + '\n'
        support_string = line + message + line
        method_to.__doc__ += support_string + method_from.__doc__
        return method_to
    return decorator


class BaseSemblance:
    """ Base class for velocity analysis.

    Attributes
    ----------
    _seismogram : array-like
        Data for calculating semblance. The attribute is stored in a transposed form due to performance reasons,
        so that `_seismogram.shape` is (num_traces, trace_lenght).
    _times : array-like
        An array containing the recording time for each trace value.
        Measured in milliseconds.
    _offsets : array-like
        The distance from the source to the receiver for each trace.
        Measured in meters.
    _sample_rate : int or float
        Step in milliseconds between signal amplitude measurements during shooting.
        Measured in milliseconds.
    _win_size : int
        Window size for smoothing the semblance.
        Measured in samples.
    """
    def __init__(self, seismogram, times, offsets, win_size):
        self._seismogram = np.ascontiguousarray(seismogram.T)
        self._times = times # ms
        self._offsets = offsets # m
        self._sample_rate = self._times[1] - self._times[0]
        self._win_size = win_size

    @staticmethod
    @njit(nogil=True, fastmath=True, parallel=True)
    def base_calc_semblance(calc_nmo_func, seismogram, times, offsets, velocity, sample_rate, # pylint: disable=too-many-arguments
                            win_size, t_min, t_max):
        """ Calculate semblance for specified velocity in the preset time window from `t_min` to `t_max`.

        Parameters
        ----------
        calc_nmo_func : njitted callable
            Callable that calculates normal moveout corrected seismogram for specified time and velocity values
            and range of offsets.
        seismogram : np.ndarray
            Data for calculating semblance.
        times : array-like
            An array containing the recording time for each trace value.
        offsets : array-like
            The distance from the source to the receiver for each trace.
        velocity : array-like
            Velocity value for semblance computation.
        sample_rate : int
            Step in milliseconds between signal amplitude measurements during shooting.
        _win_size : int
            Window size for smoothing the semblance.
            Measured in samples.
        t_min : int
            Time value to start compute semblance from.
            Measured in samples.
        t_max : int
            The last time value for semblance computation.
            Measured in samples.

        Returns
        -------
        slice_semblance : 1d array
            Semblance values for a specified `veloicty` in time range from `t_min` to `t_max`.
        """
        t_win_size_min = max(0, t_min - win_size)
        t_win_size_max = min(len(times) - 1, t_max + win_size)

        nmo = np.empty((t_win_size_max - t_win_size_min + 1, seismogram.shape[1]))
        for i in prange(t_win_size_min, t_win_size_max):
            nmo[i - t_win_size_min] = calc_nmo_func(seismogram, times[i], offsets, velocity, sample_rate)

        numerator = np.sum(nmo, axis=1)**2
        denominator = np.sum(nmo**2, axis=1)
        slice_semblance = np.zeros(t_max-t_min)
        for t in prange(t_min, t_max):
            t_rel = t - t_win_size_min
            ix_from = max(0, t_rel - win_size)
            ix_to = min(len(nmo) - 1, t_rel + win_size)
            slice_semblance[t-t_min] = (np.sum(numerator[ix_from : ix_to]) /
                                        (len(offsets) * np.sum(denominator[ix_from : ix_to])
                                         + 1e-6))
        return slice_semblance

    @staticmethod
    @njit(nogil=True, fastmath=True)
    def base_calc_nmo(seismogram, time, offsets, velocity, sample_rate):
        r""" Default approach for normal moveout computation for single hodograph. Corrected seismogram calculates
        as following:
        :math:`t_c = \sqrt{t^2 + \frac{l^2}{v^2}}`, where
            t_c - corrected time value.
            t - specified time value.
            l - distance from the source to receiver.
            v - velocity.

        Parameters
        ----------
        seismogram : np.ndarray
            Data for calculating normal moveout.
        time : int
            Time value to calculate normal moveout.
        offsets : array-like
            The distance from the source to the receiver for each trace.
        velocity : array-like
            Velocity value for NMO computation.
        sample_rate : int
            Step in milliseconds between signal amplitude measurements during shooting.

        Returns
        -------
        corrected_seismogram : 1d array
            NMO corrected hodograph.
        """
        corrected_seismogram = np.zeros(len(offsets))
        corrected_times = (np.sqrt(time**2 + offsets**2/velocity**2) / sample_rate).astype(np.int32)
        for i in range(len(offsets)):
            corrected_time = corrected_times[i]
            if corrected_time < len(seismogram):
                corrected_seismogram[i] = seismogram[corrected_time, i]
        return corrected_seismogram

    def plot(self, semblance, ticks_range_x, ticks_range_y, xlabel='', title='', figsize=(15, 12), # pylint: disable=too-many-arguments
             fontsize=11, grid=None, x_points=None, y_points=None, save_to=None, dpi=300, **kwargs):
        """ Base plotter for vertical velocity semblance. The plotter adds level lines, colors the graph, signs axes
        and values, also draw a stacking velocity, if specified (via `x_points` and `y_points`).

        Parameters
        ----------
        semblance : 2-d np.ndarray
            Array with vertical velocity or residual semblance.
        ticks_range_x : array-like with length 2, optional
            Min and max value of labels on the x-axis.
        ticks_range_y : array-like with length 2, optional
            Min and max value of labels on the y-axis.
        xlabel : str, optional, by default ''
            The label of the x-axis.
        title : str, optional, by default ''
            Plot title.
        figsize : tuple, optional, by default (15, 12)
            Output plot size.
        grid : bool, optional, by default False
            If given, add a gird to the graph.
        x_points : array-like, optional
            Points of stacking velocity by the x-axis. The point is an index in semblance that corresponds to the
            current velocity.
        y_points : array-like, optional
            Points of stacking velocity by the y-axis. The point is an index in semblance that corresponds to the
            current time.
        save_to : str, optional
            If given, save the plot to the path specified.
        dpi : int, optional, by default 300
            Resolution for the saved figure.

        Note
        ----
        1. Kwargs passed into the :func:`._set_ticks`.
        """
        # Split range of semblance amplitudes on 16 discrete levels. Arguable approach, but
        # we find the result based on these levels the most attractive.
        max_val = np.max(semblance)
        levels = (np.logspace(0, 1, num=16, base=500)/500) * max_val
        levels[0] = 0
        xlist = np.arange(0, semblance.shape[1])
        ylist = np.arange(0, semblance.shape[0])
        x_grid, y_grid = np.meshgrid(xlist, ylist)

        # Add the level lines and colorize the graph.
        fig, ax = plt.subplots(figsize=figsize)
        norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=256)
        ax.contour(x_grid, y_grid, semblance, levels, colors='k', linewidths=.5, alpha=.5)
        img = ax.imshow(semblance, norm=norm, aspect='auto', cmap='seismic')
        fig.colorbar(img, ticks=levels[1::2])

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Time')

        if title:
            ax.set_title(title, fontsize=fontsize)

        # Change marker of velocity points if they are set at distance from each other.
        # It avoids dots in every point if velocity law is set for every time.
        if x_points is not None and y_points is not None:
            marker = 'o' if np.min(np.diff(np.sort(y_points))) > 50 else ''
            plt.plot(x_points, y_points, c='#fafcc2', linewidth=2.5, marker=marker)

        _set_ticks(ax, img_shape=semblance.T.shape, ticks_range_x=ticks_range_x,
                   ticks_range_y=ticks_range_y, **kwargs)
        ax.set_ylim(semblance.shape[0], 0)
        if grid:
            ax.grid(c='k')
        if save_to:
            plt.savefig(save_to, bbox_inches='tight', pad_inches=0.1, dpi=dpi)
        plt.show()


@use_docs_from(BaseSemblance)
class Semblance(BaseSemblance):
    r""" Semblance is a normalized output-input energy ratio for CDP seismogram.

    The higher the values of semblance are, the more coherent the signal is along a hyperbolic trajectory over the
    entire spread length of the CDP gather.

    The semblance is computed by:
    :math:`S(k, v) = \frac{\sum^{k+N/2}_{i=k-N/2}(\sum^{M-1}_{j=0} f_{j}(i, v))^2}
                          {M \sum^{k+N/2}_{i=k-N/2}\sum^{M-1}_{j=0} f_{j}(i, v)^2}`, where
    S - semblance value for starting time point `k` and velocity `v`,
    M - number of traces in gather,
    N - Window size,
    f_{j}(i, v) - the amplitude value on the i-th trace at NMO corrected time j.

    Vector f(i, v) represents a normal moveout correction with velocity `v` for i-th trace.
    :math:`f_{j}(i, v) = \sqrt{t_0^2 + \frac{l_i^2}{v^2}}`, where
    :math:`t_0` - start time of hyperbola assosicated with velocity v,
    :math:`l_i` - distance from the gather to the i-th trace (offset),
    :math:`v` - velocity value.

    The resulted matrix contains vertical velocity semblance values based on hyperbolas with each combination of the
    started point :math:`k` and velocity :math:`v`.
    This matrix has a shape (time_length, velocity_length).

    Attributes
    ----------
    semblance : 2d np.ndarray
         Array with vertical velocity semblance.
    _velocities : array-like
        Array of velocity values defined the limits for semblance computation.
        Measured in meters/seconds.

    See other attributes described in :class:`~BaseSemblance`.

    Note
    ----
    1. Detailed description of the vertical velocity semblance computation is presented
       in the method :func:`~Semblance._calc_semblance`.
    """
    def __init__(self, seismogram, times, offsets, velocities, win_size=25):
        super().__init__(seismogram=seismogram, times=times, offsets=offsets, win_size=win_size)
        self._semblance = None
        self._velocities = velocities # m/s

        self._calc_semblance()

    @property
    def semblance(self):
        """ Property returns the copy of `_semblance` attribute to save semblance from occasional changes. """
        return self._semblance.copy()

    def _calc_semblance(self):
        """ Calculation of vertical velocity semblance starts with computing normal moveout for the entire seismogram
        with specified velocity. NMO corrected gather stacked along the offset axis in two ways. The first stack is a
        squared sum of amplitudes named `numerator` while the second one was a sum of squared amplitudes named
        `denominator`. Thus, the resulted semblance values for particular velocity are received as a ratio of these
        stacks in the specified `win_size`. The same algorithm repeats for every velocity point.

        Note
        ----
        1. To maintain the correct units, the velocities are converted to the meter/millisecond.
        """
        velocities_ms = self._velocities / 1000 # from m/s to m/ms
        self._semblance = self._calc_semblance_numba(base_func=self.base_calc_semblance,
                                                     calc_nmo_func=self.base_calc_nmo,
                                                     seismogram=self._seismogram, times=self._times,
                                                     offsets=self._offsets, velocities=velocities_ms,
                                                     sample_rate=self._sample_rate, win_size=self._win_size)

    @staticmethod
    @njit(nogil=True, fastmath=True, parallel=True)
    def _calc_semblance_numba(base_func, calc_nmo_func, seismogram, times, offsets, velocities, sample_rate, win_size):
        """ Parallelized method for calculating vertical velocity semblance. Most of this method uses the class
        attributes already described in the init method, so unique parameters will be described here.

        Parameters
        ----------
        base_func : callable with njit decorator
            Base function for semblance computation.
        calc_nmo_func : callable with njit decorator
            Callable that calculates normal moveout for given seismogram, time, velocity, and offset.

        Returns
        -------
        semblance : 2d np.ndarray
            Array with vertical velocity semblance.
        """
        semblance = np.empty((len(seismogram), len(velocities)))
        for j in prange(len(velocities)):
            semblance[:, j] = base_func(calc_nmo_func=calc_nmo_func, seismogram=seismogram, times=times,
                                        offsets=offsets, velocity=velocities[j], sample_rate=sample_rate,
                                        win_size=win_size, t_min=0, t_max=len(seismogram))
        return semblance

    @use_docs_from(BaseSemblance.plot)
    def plot(self, stacking_velocities=None, **kwargs):
        """ Plot vertical velocity semblance.

        Parameters
        ----------
        stacking_velocities : array-like, optional
            Array with elements in format [[time, velocity], ...]. If given, the law will be plot as a thin light brown
            line above the semblance. Also, if the delay between velocities more than 50 ms, every given point will be
            highlighted with a circle.
        kwargs : dict, optional
            Arguments for :func:`~BaseSemblance.plot` and for :func:`._set_ticks`.
        """
        x_points, y_points = None, None
        # Add a velocity line on semblance.
        if stacking_velocities is not None:
            # Find the coordinates on the graph that correspond to a certain velocity.
            stacking_velocities = np.asarray(stacking_velocities)
            x_points = ((stacking_velocities[:, 1] - self._velocities[0]) /
                        (self._velocities[-1] - self._velocities[0]) * self.semblance.shape[1])
            y_points = stacking_velocities[:, 0] / self._sample_rate
        ticks_range_y = [0, self.semblance.shape[0] * self._sample_rate]
        ticks_range_x = [self._velocities[0], self._velocities[-1]]
        super().plot(self.semblance, ticks_range_x, ticks_range_y, x_points=x_points,
                     y_points=y_points, xlabel='Velocity (m/s)', **kwargs)

    def calc_na_metrics(self, other):
        """" The metric is designed to search for signal leakage in the process of ground-roll attenuation.
        It is based on the assumption that a vertical velocity semblance calculated for the difference between a raw
        and processed gather should not have pronounced energy maxima.

        Parameters
        ----------
        self : Semblance
            Class containing semblance for difference seismogram.
        other : Semblance
            Class containing semblance for raw gather.

        Returns
        -------
        metrics : float
            Metrics value represented how much signal leaked out during the seismogram processing.
        """
        minmax_self = np.max(self.semblance, axis=1) - np.min(self.semblance, axis=1)
        minmax_other = np.max(other.semblance, axis=1) - np.min(other.semblance, axis=1)
        return np.max(minmax_self / (minmax_other + 1e-11))


@use_docs_from(BaseSemblance)
class ResidualSemblance(BaseSemblance):
    """ Residual Semblance is a normalized output-input energy ratio for CDP seismogram along picked stacking velocity.

    The method of computation at a single point completely coincides with the calculation of the :class:`~Semblance`,
    however, the residual semblance is computed in a specified area around the velocity, which allows finding errors
    and update the initially picked stacking velocity. The boundaries in which the calculation is performed for a given
    i-th stacking velocity are determined as `stacking_velocities[i]`*(1 +- `relative_margin`).

    Since the boundaries will be different for each stacking velocity, the residual semblance values are interpolated
    to obtain a rectangular matrix of size (time_length, max(right_boundary - left_boundary)), where
    `left_boundary` and `right_boundary` are the left and the right boundaries for each timestamp respectively.

    Thus, the final semblance is a rectangular matrix, the central values of which indicate the energy ratio at the
    points corresponding to the current stacking velocity. The centerline should contain the maximum energy values for
    every velocity points. If this condition is not met, then it is necessary to correct the stacking velocity.

    Parameters
    ----------
    num_vels : int
        The number of velocities to compute semblance for.

    Attributes
    ----------
    _residual_semblance : 2d np.ndarray
        Array with vertical residual semblance.
    _stacking_velocities : array-like, optional
        Array with elements in format [[time, velocity], ...]. Non-decreasing
        function passing through the maximum energy values on the semblance graph.
    _relative_margin : float, optional, default 0.2
        The relative velocity margin determines the border for a particular stacking velocity value. The boundaries for
        every stacking velocity are `stacking_velocities[i]`*(1 +- `relative_margin`).
    _velocities : array-like
        Arrange of velocity values with the limits for vertical residual semblance computation defined as a
        `numpy.linspace` from `min(stacking_velocities) * (1-relative_margin)` to
        `max(stacking_velocities) * (1+relative_margin)` with `num_vels` elements.
        Measured in meters/seconds.

    Other attributes described in :class:`~BaseSemblance`.
    """
    def __init__(self, seismogram, times, offsets, stacking_velocities, num_vels=140, win_size=25, relative_margin=0.2):
        super().__init__(seismogram, times, offsets, win_size)
        self._residual_semblance = None
        self._stacking_velocities = stacking_velocities
        self._relative_margin = relative_margin
        self._velocities = np.linspace(np.min(self._stacking_velocities[:, 1]) * (1 - self._relative_margin),
                                       np.max(self._stacking_velocities[:, 1]) * (1 + self._relative_margin),
                                       num_vels)

        self._calc_residual_semblance()

    @property
    def residual_semblance(self):
        """ Property returns the copy of `_residual_semblance` attribute to save semblance from occasional changes. """
        return self._residual_semblance.copy()

    def _calc_residual_semblance(self):
        """ Obtaining boundaries based on a given stacking velocity and calculating residual semblance. """
        velocities_ms = self._velocities / 1000 # from m/s to m/ms
        stacking_velocities_ms = self._stacking_velocities.copy()
        stacking_velocities_ms[:, 1] /= 1000 # from m/s to m/ms

        left_bounds, right_bounds = self._calc_velocity_bounds()
        self._residual_semblance = self._calc_res_semblance_numba(base_func=self.base_calc_semblance,
                                                                  calc_nmo_func=self.base_calc_nmo,
                                                                  seismogram=self._seismogram, times=self._times,
                                                                  offsets=self._offsets, velocities=velocities_ms,
                                                                  left_bounds=left_bounds, right_bounds=right_bounds,
                                                                  sample_rate=self._sample_rate,
                                                                  win_size=self._win_size)

    def _calc_velocity_bounds(self):
        """ Calculates the boundaries within which the residual semblance will be considered. To obtain a continuous
        boundary, the stacking velocity values are interpolated.

        Returns
        -------
        left_bounds : 1d array
            Indices of corresponding velocities on left bounds for each time.
        right_bounds : 1d array
            Indices of corresponding velocities on right bounds for each time.
        """
        # Interpolate velocity because it is necessary to know the boundary for each time while the stacking velocity
        # might be set for arbitrary times.
        stacking_times, stacking_vel = zip(*self._stacking_velocities)
        f = interp1d(stacking_times, stacking_vel, fill_value="extrapolate")
        interpolated_velocity = np.clip(f(self._times), self._velocities[0], self._velocities[-1])

        # Define indices of velocities that correspond to velocities on found boundaries.
        left_bounds = (interpolated_velocity * (1 - self._relative_margin)).reshape(-1, 1)
        left_bounds = np.argmin(np.abs(left_bounds - self._velocities), axis=1)
        right_bounds = (interpolated_velocity * (1 + self._relative_margin)).reshape(-1, 1)
        right_bounds = np.argmin(np.abs(right_bounds - self._velocities), axis=1)
        return left_bounds, right_bounds

    @staticmethod
    @njit(nogil=True, fastmath=True, parallel=True)
    def _calc_res_semblance_numba(base_func, calc_nmo_func, seismogram, times, offsets, velocities, left_bounds,
                                  right_bounds, sample_rate, win_size):
        """ Parallelized method for calculating residual semblance. Most of this method uses the class attributes
        already described in the init method, so unique parameters will be described here.

        Parameters
        ----------
        base_func : callable with njit decorator
            Base function for semblance computation.
        calc_nmo_func : callable with njit decorator
            Callable that calculates normal moveout for given seismogram, time, velocity, and offset.
        left_bounds : 1d array
            Indices of corresponding velocities on left bounds for each time.
        right_bounds : 1d array
            Indices of corresponding velocities on right bounds for each time.

        Returns
        -------
        semblance : 2d np.ndarray
            Array with residual semblance.
        """
        semblance = np.zeros((len(seismogram), len(velocities)))
        for i in prange(left_bounds.min(), right_bounds.max() + 1):
            t_min = np.where(right_bounds == i)[0]
            t_min = 0 if len(t_min) == 0 else t_min[0]

            t_max = np.where(left_bounds == i)[0]
            t_max = len(times) - 1 if len(t_max) == 0 else t_max[-1]

            semblance[:, i][t_min: t_max+1] = base_func(calc_nmo_func=calc_nmo_func, seismogram=seismogram,
                                                        times=times, offsets=offsets, velocity=velocities[i],
                                                        sample_rate=sample_rate, win_size=win_size, t_min=t_min,
                                                        t_max=t_max+1)

        semblance_len = (right_bounds - left_bounds).max()
        residual_semblance = np.empty((len(times), semblance_len))
        # # Interpolate semblance to get a rectangular image.
        for i in prange(len(semblance)):
            left_bound, right_bound = left_bounds[i], right_bounds[i]
            cropped_smb = semblance[i][left_bound: right_bound+1]
            residual_semblance[i] = np.interp(np.linspace(0, len(cropped_smb)-1, semblance_len),
                                              np.arange(len(cropped_smb)),
                                              cropped_smb)
        return residual_semblance

    @use_docs_from(BaseSemblance.plot)
    def plot(self, **kwargs):
        """ Plot vertical residual semblance. The graph always has a vertical line in the middle, but if the delay
        between velocities in `self._stacking_velocities` is greater 50 ms, every given point will be highlighted with
        a circle.

        Parameters
        ----------
        kwargs : dict, optional
            Arguments for :func:`~BaseSemblance.plot` and for :func:`._set_ticks`.
        """
        y_points = self._stacking_velocities[:, 0] / self._sample_rate # from ms to ix
        x_points = np.zeros(len(y_points)) + self.residual_semblance.shape[1]/2

        ticks_range_y = [0, self.residual_semblance.shape[0] * self._sample_rate] # from ix to ms
        ticks_range_x = [-self._relative_margin*100, self._relative_margin*100]

        super().plot(self.residual_semblance, ticks_range_x=ticks_range_x, ticks_range_y=ticks_range_y,
                     x_points=x_points, y_points=y_points, xlabel='Relative velocity margin (%)', **kwargs)
