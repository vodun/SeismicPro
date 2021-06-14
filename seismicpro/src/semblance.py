"""Implements Semblance and ResidualSemblance classes"""

# pylint: disable=not-an-iterable
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from matplotlib import colors as mcolors

from .utils import set_ticks
from .decorators import batch_method
from .velocity_model import calculate_stacking_velocity
from .velocity_cube import StackingVelocity
from .utils.correction import get_hodograph


class BaseSemblance:
    """Base class for vertical velocity semblance calculation.

    Implements general computation logic and visualisation method.

    Parameters
    ----------
    gather : Gather
        Seismic gather to calculate semblance for.
    win_size : int
        Window size used for semblance calculation. The higher the `win_size` is, the smoother the resulting semblance
        will be but to the detriment of small details. Measured in samples.

    Attributes
    ----------
    gather : Gather
        Seismic gather for which semblance calculation was called.
    gather_data : np.ndarray
        Gather data for semblance calculation. The data is stored in a transposed form, compared to `Gather.data` due
        to performance reasons, so that `gather_data.shape` is (trace_lenght, num_traces).
    win_size : int
        Window size for smoothing the semblance. Measured in samples.
    """
    def __init__(self, gather, win_size):
        self.gather = gather
        self.gather_data = np.ascontiguousarray(gather.data.T)
        self.win_size = win_size  # samples

    @property
    def times(self):
        """np.ndarray of floats: Recording time for each trace value. Measured in milliseconds."""
        return self.gather.times  # ms

    @property
    def sample_rate(self):
        """float: sample rate of seismic traces. Measured in milliseconds."""
        return self.gather.sample_rate  # ms

    @property
    def offsets(self):
        """np.ndarray of floats: The distance between source and receiver for each trace. Measured in meters."""
        return self.gather.offsets  # m

    def get_coords(self, coords_columns="index"):
        """Get spatial coordinates of the semblance.

        The call is redirected to the underlying gather.

        Parameters
        ----------
        coords_columns : None, "index" or 2 element array-like, default "index"
            - If `None`, (`None`, `None`) tuple is returned.
            - If "index", unique underlying gather index value is used to define semblance coordinates
            - If 2 element array-like, `coords_columns` define gather headers to get x and y coordinates from.
            In the last two cases index or column values are supposed to be unique for all traces in the underlying
            gather.

        Returns
        -------
        coords : tuple with 2 elements
            Semblance spatial coordinates.
        """
        return self.gather.get_coords(coords_columns)

    @staticmethod
    @njit(nogil=True, fastmath=True, parallel=True)
    def calc_single_velocity_semblance(nmo_func, gather_data, times, offsets, velocity, sample_rate, win_size,
                                       t_min_ix, t_max_ix):  # pylint: disable=too-many-arguments
        """Calculate semblance for given velocity and time range.

        Parameters
        ----------
        nmo_func : njitted callable
            A callable that calculates normal moveout corrected gather for given time and velocity values and a range
            of offsets.
        gather_data : 2d np.ndarray
            Gather data for semblance calculation with (trace_lenght, num_traces) layout.
        times : 1d np.ndarray
            Recording time for each trace value. Measured in milliseconds.
        offsets : array-like
            The distance between source and receiver for each trace. Measured in meters.
        velocity : array-like
            Seismic wave velocity for semblance computation. Measured in meters/milliseconds.
        sample_rate : float
            Sample rate of seismic traces. Measured in milliseconds.
        win_size : int
            Window size for smoothing the semblance. Measured in samples.
        t_min_ix : int
            Time index in `times` array to start calculating semblance from. Measured in samples.
        t_max_ix : int
            Time index in `times` array to stop calculating semblance at. Measured in samples.

        Returns
        -------
        semblance_slice : 1d np.ndarray
            Calculated semblance values for a specified `velocity` in time range from `t_min_ix` to `t_max_ix`.
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
    def plot(semblance, ticks_range_x, ticks_range_y, xlabel, title=None,  # pylint: disable=too-many-arguments
             figsize=(15, 12), fontsize=11, grid=False, stacking_times_ix=None, stacking_velocities_ix=None,
             save_to=None, dpi=300, **kwargs):
        """Plot vertical velocity semblance and, optionally, stacking velocity.

        Parameters
        ----------
        semblance : 2d np.ndarray
            An array with vertical velocity or residual semblance.
        ticks_range_x : array-like with length 2
            Min and max value of labels on the x-axis.
        ticks_range_y : array-like with length 2
            Min and max value of labels on the y-axis.
        xlabel : str
            The title of the x-axis.
        title : str, optional, default None
            Plot title.
        figsize : array-like with length 2, optional, default (15, 12)
            Output plot size.
        fontsize : int, optional, default 11
            The size of the text on the plot.
        grid : bool, optional, by default False
            Specifies whether to draw a grid on the plot.
        stacking_times_ix : 1d np.ndarray, optional
            Time indices of calculated stacking velocities to show on a plot.
        stacking_velocities_ix : 1d np.ndarray, optional
            Velocity indices of calculated stacking velocities to show on a plot.
        save_to : str, optional, default None
            If given, save the plot to the path specified.
        dpi : int, optional, default 300
            Resolution for the saved figure.
        kwargs : misc, optional
            Additional keyword arguments to :func:`.set_ticks`.
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


class Semblance(BaseSemblance):
    r"""A class for vertical velocity semblance calculation and processing.

    Semblance is a normalized output-input energy ratio for a CDP gather. The higher the values of semblance are, the
    more coherent the signal is along a hyperbolic trajectory over the entire spread length of the gather.

    Semblance instance can be created either directly by passing source gather, velocity range and window size to its
    init or by calling :func:`~Gather.calculate_semblance` method (recommended way).

    The semblance is computed by:
    :math:`S(k, v) = \frac{\sum^{k+N/2}_{i=k-N/2}(\sum^{M-1}_{j=0} f_{j}(i, v))^2}
                          {M \sum^{k+N/2}_{i=k-N/2}\sum^{M-1}_{j=0} f_{j}(i, v)^2}`,
    where:

    S - semblance value for starting time index `k` and velocity `v`,
    M - number of traces in the gather,
    N - window size,
    f_{j}(i, v) - the amplitude value on the `j`-th trace being NMO-corrected for time index `i` and velocity `v`:
    :math:`f_{j}(i, v) = \sqrt{t_0^2 + \frac{l_i^2}{v^2}}`,
    where:

    :math:`t_0` - start time of the hyperbola assosicated with time index `i`,
    :math:`l_j` - offset of the `j`-th trace,
    :math:`v` - velocity value.

    The resulting matrix :math:`S(k, v)` contains vertical velocity semblance values based on hyperbolas with each
    combination of the starting point :math:`k` and velocity :math:`v`. This matrix has shape
    (trace_lenght, n_velocities).

    The algorithm for semblance calculation looks as follows:
    For each velocity from given velocity range:
        1. Calcluate NMO-corrected gather
        2. Calcluate squared sum of amplitudes of the corrected gather along the offset axis
        3. Calcluate sum of squared amplitudes of the corrected gather along the offset axis
        4. For each time calculate sum of values from step 2 and 3 in the window with given `win_size` and divide the
           first obtained value by the second one to get semblance value for given time and velocity

    Parameters
    ----------
    gather : Gather
        Seismic gather to calculate semblance for.
    velocities : 1d np.ndarray
        Range of velocity values for which semblance is calculated. Measured in meters/seconds.
    win_size : int, optional, default 25
        Window size used for semblance calculation. The higher the `win_size` is, the smoother the resulting semblance
        will be but to the detriment of small details. Measured in samples.

    Attributes
    ----------
    gather : Gather
        Seismic gather for which semblance calculation was called.
    gather_data : np.ndarray
        Gather data for semblance calculation. The data is stored in a transposed form, compared to `Gather.data` due
        to performance reasons, so that `gather_data.shape` is (trace_lenght, num_traces).
    velocities : 1d np.ndarray
        Range of velocity values for which semblance is calculated. Measured in meters/seconds.
    win_size : int
        Window size for smoothing the semblance. Measured in samples.
    semblance : 2d np.ndarray
         Array with calculated vertical velocity semblance values.
    """
    def __init__(self, gather, velocities, win_size=25):
        super().__init__(gather, win_size=win_size)
        self.velocities = velocities  # m/s
        velocities_ms = self.velocities / 1000  # from m/s to m/ms
        self.semblance = self._calc_semblance_numba(semblance_func=self.calc_single_velocity_semblance,
                                                    nmo_func=get_hodograph, gather_data=self.gather_data,
                                                    times=self.times, offsets=self.offsets, velocities=velocities_ms,
                                                    sample_rate=self.sample_rate, win_size=self.win_size)

    @staticmethod
    @njit(nogil=True, fastmath=True, parallel=True)
    def _calc_semblance_numba(semblance_func, nmo_func, gather_data, times, offsets, velocities, sample_rate,
                              win_size):
        """Parallelized and njitted method for vertical velocity semblance calculation.

        Parameters
        ----------
        semblance_func : njitted callable
            Base function for semblance calculation for single velocity and a time range.
        nmo_func : njitted callable
            Base function for gather normal moveout correction for given time and velocity.
        other parameters : misc
            Passed directly from class attributes (except for velocities which are converted from m/s to m/ms)

        Returns
        -------
        semblance : 2d np.ndarray
            Array with vertical velocity semblance values.
        """
        semblance = np.empty((len(gather_data), len(velocities)))
        for j in prange(len(velocities)):
            semblance[:, j] = semblance_func(nmo_func=nmo_func, gather_data=gather_data, times=times, offsets=offsets,
                                             velocity=velocities[j], sample_rate=sample_rate, win_size=win_size,
                                             t_min_ix=0, t_max_ix=len(gather_data))
        return semblance

    @batch_method(target="for", args_to_unpack="stacking_velocity")
    def plot(self, stacking_velocity=None, **kwargs):
        """Plot vertical velocity semblance.

        Parameters
        ----------
        stacking_velocity : StackingVelocity, optional
            Stacking velocity to plot if given. If its sample rate is more than 50 ms, every point will be highlighted
            with a circle.
        kwargs : misc, optional
            Additional named arguments for :func:`~BaseSemblance.plot`.

        Returns
        -------
        semblance : Semblance
            Self unchanged.
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

    @batch_method(target="for", args_to_unpack="other")
    def calculate_signal_leakage(self, other):
        """Calculate signal leakage during ground-roll attenuation.

        The metric is based on the assumption that a vertical velocity semblance calculated for the difference between
        raw and processed gathers should not have pronounced energy maxima.

        Parameters
        ----------
        self : Semblance
            Semblance calculated for gather difference.
        other : Semblance
            Semblance for raw gather.

        Returns
        -------
        metric : float
            Signal leakage during gather processing.
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
        return StackingVelocity.from_points(times, velocities, inline, crossline)


class ResidualSemblance(BaseSemblance):
    """A class for residual vertical velocity semblance calculation and processing.

    Residual semblance is a normalized output-input energy ratio for a CDP gather along picked stacking velocity. The
    method of its computation for given time and velocity completely coincides with the calculation of
    :class:`~Semblance`, however, residual semblance is computed in a small area around stacking velocity, thus
    allowing for additional optimizations.

    The boundaries in which calculation is performed for given time `t` are determined as:
    `stacking_velocity(t)` * (1 +- `relative_margin`).
    Since the velocity range length differs for each timestamp, the residual semblance values are interpolated to
    obtain a rectangular matrix of size (trace_lenght, max(right_boundary - left_boundary)), where `left_boundary` and
    `right_boundary` are arrays of left and right boundaries for all timestamps respectively.

    The central vertical line on the residual semblance corresponds to the given stacking velocity and generally should
    pass through local semblance maximas.

    Parameters
    ----------
    gather : Gather
        Seismic gather to calculate semblance for.
    stacking_velocity : StackingVelocity
        Stacking velocity around which residual semblance is calculated.
    n_velocities : int, optional, default 140
        The number of velocities to compute semblance for.
    win_size : int, optional, default 25
        Window size used for semblance calculation. The higher the `win_size` is, the smoother the resulting semblance
        will be but to the detriment of small details. Measured in samples.
    relative_margin : float, optional, default 0.2
        Relative velocity margin, that determines the velocity range for semblance calculation for each time `t` as
        `stacking_velocity(t)` * (1 +- `relative_margin`).

    Attributes
    ----------
    gather : Gather
        Seismic gather for which semblance calculation was called.
    gather_data : np.ndarray
        Gather data for semblance calculation. The data is stored in a transposed form, compared to `Gather.data` due
        to performance reasons, so that `gather_data.shape` is (trace_lenght, num_traces).
    velocities : 1d np.ndarray
        Range of velocity values for which residual semblance is calculated. Measured in meters/seconds.
    win_size : int
        Window size for smoothing the semblance. Measured in samples.
    stacking_velocity : StackingVelocity
        Stacking velocity around which residual semblance is calculated.
    relative_margin : float, optional, default 0.2
         Relative velocity margin, that determines the velocity range for semblance calculation for each timestamp.
    residual_semblance : 2d np.ndarray
         Array with calculated residual vertical velocity semblance values.
    """
    def __init__(self, gather, stacking_velocity, n_velocities=140, win_size=25, relative_margin=0.2):
        super().__init__(gather, win_size)
        self.stacking_velocity = stacking_velocity
        self.relative_margin = relative_margin

        interpolated_velocities = stacking_velocity(self.times)
        self.velocities = np.linspace(np.min(interpolated_velocities) * (1 - relative_margin),
                                      np.max(interpolated_velocities) * (1 + relative_margin),
                                      n_velocities)
        velocities_ms = self.velocities / 1000  # from m/s to m/ms

        left_bound_ix, right_bound_ix = self._calc_velocity_bounds()
        self.residual_semblance = self._calc_res_semblance_numba(semblance_func=self.calc_single_velocity_semblance,
                                                                 nmo_func=get_hodograph, gather_data=self.gather_data,
                                                                 times=self.times, offsets=self.offsets,
                                                                 velocities=velocities_ms,
                                                                 left_bound_ix=left_bound_ix,
                                                                 right_bound_ix=right_bound_ix,
                                                                 sample_rate=self.sample_rate, win_size=self.win_size)

    def _calc_velocity_bounds(self):
        """Calculate velocity boundaries within which residual semblance will be calculated.

        Returns
        -------
        left_bound_ix : 1d array
            Indices of corresponding velocities of left bounds for each time.
        right_bound_ix : 1d array
            Indices of corresponding velocities of right bounds for each time.
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
        """Parallelized and njitted method for residual vertical velocity semblance calculation.

        Parameters
        ----------
        semblance_func : njitted callable
            Base function for semblance calculation for single velocity and a time range.
        nmo_func : njitted callable
            Base function for gather normal moveout correction for given time and velocity.
        left_bound_ix : 1d array
            Indices of corresponding velocities of left bounds for each time.
        right_bound_ix : 1d array
            Indices of corresponding velocities of right bounds for each time.
        other parameters : misc
            Passed directly from class attributes (except for velocities which are converted from m/s to m/ms)

        Returns
        -------
        semblance : 2d np.ndarray
            Array with residual vertical velocity semblance values.
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
    def plot(self, **kwargs):
        """Plot residual vertical velocity semblance. The plot always has a vertical line in the middle, representing
        given stacking velocity.

        Parameters
        ----------
        kwargs : misc, optional
            Additional named arguments for :func:`~BaseSemblance.plot`.

        Returns
        -------
        semblance : ResidualSemblance
            Self unchanged.
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
