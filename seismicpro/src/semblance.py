""" The file contains classes for velocity analysis. """
# pylint: disable=not-an-iterable
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from matplotlib import colors as mcolors

from .utils import set_ticks
from .decorators import batch_method
from .velocity_model import calculate_stacking_velocity
from .velocity_cube import StackingVelocity


def append_docs_from(method_from):
    """Append the docstring of `method_from` to `method_to`.

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
        message = f'|  For clarity, the docstrings of the `{method_from.__qualname__}` are shown below.  |'
        line = '\n' + '-' * len(message) + '\n'
        support_string = line + message + line
        method_to.__doc__ += support_string + method_from.__doc__
        return method_to
    return decorator


class BaseSemblance:
    """ Base class for velocity analysis.

    Attributes
    ----------
    _gather : array-like
        Data for calculating semblance. The attribute is stored in a transposed form due to performance reasons,
        so that `_gather.shape` is (num_traces, trace_lenght).
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
    def __init__(self, gather, win_size):
        self.gather = gather
        self.gather_data = np.ascontiguousarray(gather.data.T)
        self.win_size = win_size

    @property
    def times(self):
        return self.gather.times  # ms

    @property
    def sample_rate(self):
        return self.gather.sample_rate  # ms

    @property
    def offsets(self):
        return self.gather.offsets  # m

    def get_coords(self, coords_columns="index"):
        return self.gather.get_coords(coords_columns)

    @staticmethod
    @njit(nogil=True, fastmath=True, parallel=True)
    def calc_single_velocity_semblance(nmo_func, gather_data, times, offsets, velocity, sample_rate, win_size,
                                       t_min_ix, t_max_ix):  # pylint: disable=too-many-arguments
        """ Calculate semblance for specified velocity in the preset time window from `t_min` to `t_max`.

        Parameters
        ----------
        nmo_func : njitted callable
            Callable that calculates normal moveout corrected gather for specified time and velocity values
            and range of offsets.
        gather : np.ndarray
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
        t_win_size_min_ix = max(0, t_min_ix - win_size)
        t_win_size_max_ix = min(len(times) - 1, t_max_ix + win_size)

        corrected_gather = np.empty((t_win_size_max_ix - t_win_size_min_ix + 1, gather_data.shape[1]))
        for i in prange(t_win_size_min_ix, t_win_size_max_ix):
            corrected_gather[i - t_win_size_min_ix] = nmo_func(gather_data, times[i], offsets, velocity, sample_rate)

        numerator = np.sum(corrected_gather, axis=1)**2
        denominator = np.sum(corrected_gather**2, axis=1)
        semblance_slice = np.zeros(t_max_ix - t_min_ix)
        for t in prange(t_min_ix, t_max_ix):
            t_rel = t - t_win_size_min_ix
            ix_from = max(0, t_rel - win_size)
            ix_to = min(len(corrected_gather) - 1, t_rel + win_size)
            semblance_slice[t - t_min_ix] = (np.sum(numerator[ix_from : ix_to]) /
                                             (len(offsets) * np.sum(denominator[ix_from : ix_to]) + 1e-6))
        return semblance_slice

    @staticmethod
    @njit(nogil=True, fastmath=True)
    def apply_nmo(gather_data, time, offsets, velocity, sample_rate):
        r""" Default approach for normal moveout computation for single hodograph. Corrected gather calculates
        as following:
        :math:`t_c = \sqrt{t^2 + \frac{l^2}{v^2}}`, where
            t_c - corrected time value.
            t - specified time value.
            l - distance from the source to receiver.
            v - velocity.

        Parameters
        ----------
        gather : np.ndarray
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
        corrected_gather : 1d array
            NMO corrected hodograph.
        """
        corrected_gather = np.zeros(len(offsets))
        corrected_times = (np.sqrt(time**2 + offsets**2/velocity**2) / sample_rate).astype(np.int32)
        for i in range(len(offsets)):
            corrected_time = corrected_times[i]
            if corrected_time < len(gather_data):
                corrected_gather[i] = gather_data[corrected_time, i]
        return corrected_gather

    @staticmethod
    def plot(semblance, ticks_range_x, ticks_range_y, xlabel, title=None,  # pylint: disable=too-many-arguments
             figsize=(15, 12), fontsize=11, grid=None, stacking_times_ix=None, stacking_velocities_ix=None,
             save_to=None, dpi=300, **kwargs):
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
        1. Kwargs passed into the :func:`.set_ticks`.
        """
        # Split the range of semblance amplitudes into 16 levels on a log scale,
        # that will further be used as colormap bins
        max_val = np.max(semblance)
        levels = (np.logspace(0, 1, num=16, base=500) / 500) * max_val
        levels[0] = 0

        # Add level lines and colorize the graph
        fig, ax = plt.subplots(figsize=figsize)
        norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=256)
        x_grid, y_grid = np.meshgrid(np.arange(0, semblance.shape[1]), np.arange(0, semblance.shape[0]))
        ax.contour(x_grid, y_grid, semblance, levels, colors='k', linewidths=.5, alpha=.5)
        img = ax.imshow(semblance, norm=norm, aspect='auto', cmap='seismic')
        fig.colorbar(img, ticks=levels[1::2])

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Time')

        if title is not None:
            ax.set_title(title, fontsize=fontsize)

        # Change markers of stacking velocity points if they are far enough apart
        if stacking_velocities_ix is not None and stacking_times_ix is not None:
            marker = 'o' if np.min(np.diff(np.sort(stacking_times_ix))) > 50 else ''
            plt.plot(stacking_velocities_ix, stacking_times_ix, c='#fafcc2', linewidth=2.5, marker=marker)

        set_ticks(ax, img_shape=semblance.T.shape, ticks_range_x=ticks_range_x, ticks_range_y=ticks_range_y, **kwargs)
        ax.set_ylim(semblance.shape[0], 0)
        if grid:
            ax.grid(c='k')
        if save_to:
            plt.savefig(save_to, bbox_inches='tight', pad_inches=0.1, dpi=dpi)
        plt.show()


@append_docs_from(BaseSemblance)
class Semblance(BaseSemblance):
    r""" Semblance is a normalized output-input energy ratio for CDP gather.

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
    def __init__(self, gather, velocities, win_size=25):
        super().__init__(gather, win_size=win_size)
        self.semblance = None
        self.velocities = velocities  # m/s
        self._calc_semblance()

    def _calc_semblance(self):
        """ Calculation of vertical velocity semblance starts with computing normal moveout for the entire gather
        with specified velocity. NMO corrected gather stacked along the offset axis in two ways. The first stack is a
        squared sum of amplitudes named `numerator` while the second one was a sum of squared amplitudes named
        `denominator`. Thus, the resulted semblance values for particular velocity are received as a ratio of these
        stacks in the specified `win_size`. The same algorithm repeats for every velocity point.

        Note
        ----
        1. To maintain the correct units, the velocities are converted to the meter/millisecond.
        """
        velocities_ms = self.velocities / 1000  # from m/s to m/ms
        self.semblance = self._calc_semblance_numba(semblance_func=self.calc_single_velocity_semblance,
                                                    nmo_func=self.apply_nmo, gather_data=self.gather_data,
                                                    times=self.times, offsets=self.offsets, velocities=velocities_ms,
                                                    sample_rate=self.sample_rate, win_size=self.win_size)

    @staticmethod
    @njit(nogil=True, fastmath=True, parallel=True)
    def _calc_semblance_numba(semblance_func, nmo_func, gather_data, times, offsets, velocities, sample_rate,
                              win_size):
        """ Parallelized method for calculating vertical velocity semblance. Most of this method uses the class
        attributes already described in the init method, so unique parameters will be described here.

        Parameters
        ----------
        semblance_func : callable with njit decorator
            Base function for semblance computation.
        nmo_func : callable with njit decorator
            Callable that calculates normal moveout for given gather, time, velocity, and offset.

        Returns
        -------
        semblance : 2d np.ndarray
            Array with vertical velocity semblance.
        """
        semblance = np.empty((len(gather_data), len(velocities)))
        for j in prange(len(velocities)):
            semblance[:, j] = semblance_func(nmo_func=nmo_func, gather_data=gather_data, times=times, offsets=offsets,
                                             velocity=velocities[j], sample_rate=sample_rate, win_size=win_size,
                                             t_min_ix=0, t_max_ix=len(gather_data))
        return semblance

    @batch_method(target="for", args_to_unpack="stacking_velocity")
    @append_docs_from(BaseSemblance.plot)
    def plot(self, stacking_velocity=None, **kwargs):
        """ Plot vertical velocity semblance.

        Parameters
        ----------
        stacking_velocities : array-like, optional
            Array with elements in format [[time, velocity], ...]. If given, the law will be plot as a thin light brown
            line above the semblance. Also, if the delay between velocities more than 50 ms, every given point will be
            highlighted with a circle.
        kwargs : dict, optional
            Arguments for :func:`~BaseSemblance.plot` and for :func:`.set_ticks`.
        """
        ticks_range_x = [self.velocities[0], self.velocities[-1]]
        ticks_range_y = [self.times[0], self.times[-1]]

        stacking_times_ix, stacking_velocities_ix = None, None
        # Add a stacking velocity line on the plot
        if stacking_velocity is not None:
            stacking_times = stacking_velocity.times if stacking_velocity.times is not None else self.times
            stacking_velocities = stacking_velocity(stacking_times)
            stacking_times_ix = stacking_times / self.sample_rate
            stacking_velocities_ix = ((stacking_velocities - self.velocities[0]) /
                                      (self.velocities[-1] - self.velocities[0]) * self.semblance.shape[1])

        super().plot(self.semblance, ticks_range_x, ticks_range_y, stacking_times_ix=stacking_times_ix,
                     stacking_velocities_ix=stacking_velocities_ix, xlabel='Velocity (m/s)', **kwargs)
        return self

    def calc_na_metrics(self, other):
        """" The metric is designed to search for signal leakage in the process of ground-roll attenuation.
        It is based on the assumption that a vertical velocity semblance calculated for the difference between a raw
        and processed gather should not have pronounced energy maxima.

        Parameters
        ----------
        self : Semblance
            Class containing semblance for difference gather.
        other : Semblance
            Class containing semblance for raw gather.

        Returns
        -------
        metrics : float
            Metrics value represented how much signal leaked out during the gather processing.
        """
        minmax_self = np.max(self.semblance, axis=1) - np.min(self.semblance, axis=1)
        minmax_other = np.max(other.semblance, axis=1) - np.min(other.semblance, axis=1)
        return np.max(minmax_self / (minmax_other + 1e-11))

    @batch_method(target="for", copy_src=False)
    def calculate_stacking_velocity(self, start_velocity_range=(1400, 1800), end_velocity_range=(2500, 5000),
                                    max_acceleration=None, n_times=25, n_velocities=25, coords_columns="index"):
        inline, crossline = self.get_coords(coords_columns)
        times, velocities, _ = calculate_stacking_velocity(self.semblance, self.times, self.velocities,
                                                           start_velocity_range, end_velocity_range, max_acceleration,
                                                           n_times, n_velocities)
        return StackingVelocity(times=times, velocities=velocities, inline=inline, crossline=crossline)


@append_docs_from(BaseSemblance)
class ResidualSemblance(BaseSemblance):
    """ Residual Semblance is a normalized output-input energy ratio for CDP gather along picked stacking velocity.

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
    def __init__(self, gather, stacking_velocity, n_velocities=140, win_size=25, relative_margin=0.2):
        super().__init__(gather, win_size)
        self.residual_semblance = None
        self.stacking_velocity = stacking_velocity
        self.relative_margin = relative_margin
        interpolated_velocities = stacking_velocity(self.times)
        self.velocities = np.linspace(np.min(interpolated_velocities) * (1 - relative_margin),
                                      np.max(interpolated_velocities) * (1 + relative_margin),
                                      n_velocities)
        self._calc_residual_semblance()

    def _calc_residual_semblance(self):
        """ Obtaining boundaries based on a given stacking velocity and calculating residual semblance. """
        velocities_ms = self.velocities / 1000  # from m/s to m/ms
        left_bound_ix, right_bound_ix = self._calc_velocity_bounds()
        self.residual_semblance = self._calc_res_semblance_numba(semblance_func=self.calc_single_velocity_semblance,
                                                                 nmo_func=self.apply_nmo,
                                                                 gather_data=self.gather_data, times=self.times,
                                                                 offsets=self.offsets, velocities=velocities_ms,
                                                                 left_bound_ix=left_bound_ix,
                                                                 right_bound_ix=right_bound_ix,
                                                                 sample_rate=self.sample_rate, win_size=self.win_size)

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
        interpolated_velocities = np.clip(self.stacking_velocity(self.times), self.velocities[0], self.velocities[-1])
        left_bound_values = (interpolated_velocities * (1 - self.relative_margin)).reshape(-1, 1)
        left_bound_ix = np.argmin(np.abs(left_bound_values - self.velocities), axis=1)
        right_bound_values = (interpolated_velocities * (1 + self.relative_margin)).reshape(-1, 1)
        right_bound_ix = np.argmin(np.abs(right_bound_values - self.velocities), axis=1)
        return left_bound_ix, right_bound_ix

    @staticmethod
    @njit(nogil=True, fastmath=True, parallel=True)
    def _calc_res_semblance_numba(semblance_func, nmo_func, gather_data, times, offsets, velocities, left_bound_ix,
                                  right_bound_ix, sample_rate, win_size):
        """ Parallelized method for calculating residual semblance. Most of this method uses the class attributes
        already described in the init method, so unique parameters will be described here.

        Parameters
        ----------
        semblance_func : callable with njit decorator
            Base function for semblance computation.
        nmo_func : callable with njit decorator
            Callable that calculates normal moveout for given gather, time, velocity, and offset.
        left_bounds : 1d array
            Indices of corresponding velocities on left bounds for each time.
        right_bounds : 1d array
            Indices of corresponding velocities on right bounds for each time.

        Returns
        -------
        semblance : 2d np.ndarray
            Array with residual semblance.
        """
        semblance = np.zeros((len(gather_data), len(velocities)))
        for i in prange(left_bound_ix.min(), right_bound_ix.max() + 1):
            t_min_ix = np.where(right_bound_ix == i)[0]
            t_min_ix = 0 if len(t_min_ix) == 0 else t_min_ix[0]

            t_max_ix = np.where(left_bound_ix == i)[0]
            t_max_ix = len(times) - 1 if len(t_max_ix) == 0 else t_max_ix[-1]

            semblance[:, i][t_min_ix : t_max_ix+1] = semblance_func(nmo_func=nmo_func, gather_data=gather_data,
                                                                    times=times, offsets=offsets,
                                                                    velocity=velocities[i], sample_rate=sample_rate,
                                                                    win_size=win_size, t_min_ix=t_min_ix,
                                                                    t_max_ix=t_max_ix+1)

        # Interpolate semblance to get a rectangular image
        semblance_len = (right_bound_ix - left_bound_ix).max()
        residual_semblance = np.empty((len(times), semblance_len))
        for i in prange(len(semblance)):
            cropped_semblance = semblance[i][left_bound_ix[i] : right_bound_ix[i] + 1]
            residual_semblance[i] = np.interp(np.linspace(0, len(cropped_semblance) - 1, semblance_len),
                                              np.arange(len(cropped_semblance)),
                                              cropped_semblance)
        return residual_semblance

    @batch_method(target="for")
    @append_docs_from(BaseSemblance.plot)
    def plot(self, **kwargs):
        """ Plot vertical residual semblance. The graph always has a vertical line in the middle, but if the delay
        between velocities in `self._stacking_velocities` is greater 50 ms, every given point will be highlighted with
        a circle.

        Parameters
        ----------
        kwargs : dict, optional
            Arguments for :func:`~BaseSemblance.plot` and for :func:`.set_ticks`.
        """
        ticks_range_x = [-self.relative_margin * 100, self.relative_margin * 100]
        ticks_range_y = [self.times[0], self.times[-1]]  # from ix to ms

        stacking_times = self.stacking_velocity.times if self.stacking_velocity.times is not None else self.times
        stacking_times_ix = stacking_times / self.sample_rate
        stacking_velocities_ix = np.full_like(stacking_times_ix, self.residual_semblance.shape[1] / 2)

        super().plot(self.residual_semblance, ticks_range_x=ticks_range_x, ticks_range_y=ticks_range_y,
                     stacking_times_ix=stacking_times_ix, stacking_velocities_ix=stacking_velocities_ix,
                     xlabel='Relative velocity margin (%)', **kwargs)
        return self
