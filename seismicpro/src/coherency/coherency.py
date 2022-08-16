"""Implements Coherency and ResidualCoherency classes"""

# pylint: disable=not-an-iterable
import numpy as np
from numba import njit, prange
from numba.typed import List
from scipy.ndimage import median_filter
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt

from .interactive_plot import SemblancePlot
from ..decorators import batch_method, plotter
from ..gather.utils.correction import get_hodograph
from ..stacking_velocity import StackingVelocity, calculate_stacking_velocity
from ..utils import add_colorbar, set_ticks, set_text_formatting
from ..gather.utils import correction


class BaseCoherency:
    """Base class for vertical velocity semblance calculation.
    Implements general computation logic and visualization method.
    Parameters
    ----------
    gather : Gather
        Seismic gather to calculate semblance for.
    win_size : int
        Temporal window size used for semblance calculation. The higher the `win_size` is, the smoother the resulting
        semblance will be but to the detriment of small details. Measured in samples.
    Attributes
    ----------
    gather : Gather
        Seismic gather for which semblance calculation was called.
    gather_data : 2d np.ndarray
        Gather data for semblance calculation. The data is stored in a transposed form, compared to `Gather.data` due
        to performance reasons, so that `gather_data.shape` is (trace_length, num_traces).
    win_size : int
        Temporal window size for smoothing the semblance. Measured in samples.
    """
    def __init__(self, gather, win_size, mode):
        self.gather = gather
        self.gather_data = gather.data
        self.win_size = win_size  # samples

        coherency_dict = {
            "stacked_amplitude": self.stacked_amplitude,
            "S": self.stacked_amplitude,
            "normalized_stacked_amplitude": self.normalized_stacked_amplitude,
            "NS": self.normalized_stacked_amplitude,
            "semblance": self.semblance,
            "NE": self.semblance,
            'CC': self.crosscorrelation
        }

        self.coherency_func = coherency_dict.get(mode)
        if self.coherency_func is None:
            raise ValueError(f"Unknown mode {mode}")

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

    @property
    def coords(self):
        """Coordinates or None: Spatial coordinates of the semblance. Determined by the underlying gather. `None` if
        the gather is indexed by unsupported headers or required coords headers were not loaded or coordinates are
        non-unique for traces of the gather."""
        return self.gather.coords

    def get_time_velocity_by_indices(self, time_ix, velocity_ix):
        """Get time (in milliseconds) and velocity (in kilometers/seconds) by their indices (possibly non-integer) in
        semblance."""
        _ = time_ix, velocity_ix
        raise NotImplementedError

    @staticmethod
    @njit(nogil=True, fastmath={'ninf'}, parallel=True)
    def stacked_amplitude(corrected_gather):
        numerator = np.zeros(corrected_gather.shape[0])
        denominator = np.ones(corrected_gather.shape[0])
        for i in prange(corrected_gather.shape[0]):
            numerator[i] = np.nanmean(corrected_gather[i, :])
        return numerator, denominator

    @staticmethod
    @njit(nogil=True, fastmath={'ninf'}, parallel=True)
    def normalized_stacked_amplitude(corrected_gather):
        numerator = np.zeros(corrected_gather.shape[0])
        denominator = np.zeros(corrected_gather.shape[0])
        for i in prange(corrected_gather.shape[0]):
            numerator[i] = np.abs(np.nansum(corrected_gather[i, :]))
            denominator[i] = np.nansum(np.abs(corrected_gather[i, :]))
        return numerator, denominator

    @staticmethod
    @njit(nogil=True, fastmath={'ninf'}, parallel=True)
    def semblance(corrected_gather):
        numerator = np.zeros(corrected_gather.shape[0])
        denominator = np.zeros(corrected_gather.shape[0])
        for i in prange(corrected_gather.shape[0]):
            numerator[i] = (np.nansum(corrected_gather[i, :]) ** 2) 
            denominator[i] = np.nansum(corrected_gather[i, :] ** 2) * sum(~np.isnan(corrected_gather[i, :]))
        return numerator, denominator


    @staticmethod
    @njit(nogil=True, parallel=True, fastmath=True)
    def crosscorrelation(arr):
        ix = np.concatenate((np.array([0]), np.cumsum(np.arange(arr.shape[0] - 1, 0, -1))))
        res = np.zeros((1 + arr.shape[0] * (arr.shape[0] - 1) // 2, arr.shape[1]))
        for i in prange(arr.shape[0]):
            for j in prange(i + 1, arr.shape[0]):
                res[ix[i] + arr.shape[0] - j] = arr[i] * arr[j]
        numerator = np.ones(arr.shape[0])
        denominator = np.ones(arr.shape[0])
        return numerator, denominator


    # @staticmethod
    # @njit(nogil=True, fastmath={'ninf'}, parallel=True)
    # def crosscorrelation(corrected_gather):
    #     numerator = np.zeros(corrected_gather.shape[0])
    #     denominator = np.ones(corrected_gather.shape[0])
    #     for i in prange(corrected_gather.shape[0]):
    #         s = 0
    #         for lag in range(1, corrected_gather.shape[1]):
    #             s += np.sum(corrected_gather[i, :-lag] * corrected_gather[i, lag:])
    #         numerator[i] = s 
    #     return numerator, denominator

    @staticmethod
    @njit(nogil=True, fastmath={'ninf'}, parallel=True)
    def calc_single_velocity_semblance(nmo_func, coherency_func, gather_data, times, offsets, velocity, sample_rate, win_size,
                                       t_min_ix, t_max_ix):  # pylint: disable=too-many-arguments
        """Calculate semblance for given velocity and time range.
        Parameters
        ----------
        nmo_func : njitted callable
            A callable that calculates normal moveout corrected gather for given time and velocity values and a range
            of offsets.
        gather_data : 2d np.ndarray
            Gather data for semblance calculation with (trace_length, num_traces) layout.
        times : 1d np.ndarray
            Recording time for each trace value. Measured in milliseconds.
        offsets : array-like
            The distance between source and receiver for each trace. Measured in meters.
        velocity : array-like
            Seismic wave velocity for semblance computation. Measured in meters/milliseconds.
        sample_rate : float
            Sample rate of seismic traces. Measured in milliseconds.
        win_size : int
            Temporal window size for smoothing the semblance. Measured in samples.
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

        new_times = times[t_win_size_min_ix: t_win_size_max_ix + 1]
        corrected_gather = correction.apply_nmo(gather_data, times, offsets, np.repeat(velocity, len(times)), sample_rate).T

        numerator, denominator = coherency_func(corrected_gather)
        numerator[np.isnan(numerator)] = 0
        denominator[np.isnan(denominator)] = 0

        semblance_slice = np.zeros(t_max_ix - t_min_ix, dtype=np.float32)
        for t in prange(t_min_ix, t_max_ix):
            t_rel = t - t_win_size_min_ix
            ix_from = max(0, t_rel - win_size)
            ix_to = min(len(corrected_gather) - 1, t_rel + win_size)
            semblance_slice[t - t_min_ix] = (np.sum(numerator[ix_from : ix_to]) / (np.sum(denominator[ix_from : ix_to]) + 1e-6))
        return semblance_slice

    @staticmethod
    def _plot(semblance, title=None, x_label=None, x_ticklabels=None,  # pylint: disable=too-many-arguments
              x_ticker=None, y_ticklabels=None, y_ticker=None, grid=False, stacking_times_ix=None,
              stacking_velocities_ix=None, colorbar=True, ax=None, levels = 10, q=1, **kwargs):
        """Plot vertical velocity semblance and, optionally, stacking velocity.
        Parameters
        ----------
        semblance : 2d np.ndarray
            An array with vertical velocity or residual semblance.
        title : str, optional, defaults to None
            Plot title.
        x_label : str, optional, defaults to None
            The title of the x-axis.
        x_ticklabels : list of str, optional, defaults to None
            An array of labels for the x-axis.
        x_ticker : dict, optional, defaults to None
            Parameters for ticks and ticklabels formatting for the x-axis; see `.utils.set_ticks` for more details.
        y_ticklabels : list of str, optional, defaults to None
            An array of labels for the y-axis.
        y_ticker : dict, optional, defaults to None
            Parameters for ticks and ticklabels formatting for the y-axis; see `.utils.set_ticks` for more details.
        grid : bool, optional, defaults to False
            Specifies whether to draw a grid on the plot.
        stacking_times_ix : 1d np.ndarray, optional
            Time indices of calculated stacking velocities to show on the plot.
        stacking_velocities_ix : 1d np.ndarray, optional
            Velocity indices of calculated stacking velocities to show on the plot.
        colorbar : bool or dict, optional, defaults to True
            Whether to add a colorbar to the right of the semblance plot. If `dict`, defines extra keyword arguments
            for `matplotlib.figure.Figure.colorbar`.
        ax : matplotlib.axes.Axes, optional, defaults to None
            Axes of the figure to plot on.
        kwargs : misc, optional
            Additional common keyword arguments for `x_ticker` and `y_tickers`.
        """
        # Cast text-related parameters to dicts and add text formatting parameters from kwargs to each of them
        (title, x_ticker, y_ticker), kwargs = set_text_formatting(title, x_ticker, y_ticker, **kwargs)

        # Split the range of semblance amplitudes into 16 levels on a log scale,
        # that will further be used as colormap bins
        # max_val = np.max(semblance)
        # levels = (np.logspace(0, 1, num=16, base=500) / 500) * max_val
        # levels[0] = 0

        # Add level lines and colorize the graph
        cmap = plt.get_cmap('seismic')
        norm = mcolors.BoundaryNorm(np.linspace(0, np.quantile(semblance, q), levels), cmap.N)
        x_grid, y_grid = np.meshgrid(np.arange(0, semblance.shape[1]), np.arange(0, semblance.shape[0]))
#        ax.contour(x_grid, y_grid, semblance, levels, colors='k', linewidths=.5, alpha=.5)
        img = ax.imshow(semblance, norm=norm, aspect='auto', cmap=cmap)
        add_colorbar(ax, img, colorbar, y_ticker=y_ticker)
        ax.set_title(**{"label": None, **title})

        # Change markers of stacking velocity points if they are far enough apart
        if stacking_velocities_ix is not None and stacking_times_ix is not None:
            marker = 'o' if np.min(np.diff(np.sort(stacking_times_ix))) > 50 else ''
            ax.plot(stacking_velocities_ix, stacking_times_ix, c='#fafcc2', linewidth=2.5, marker=marker)

        if grid:
            ax.grid(c='k')

        set_ticks(ax, "x", x_label, x_ticklabels, **x_ticker)
        set_ticks(ax, "y", "Time", y_ticklabels, **y_ticker)

    def plot(self, *args, interactive=False, **kwargs):
        """Plot semblance in interactive or non-interactive mode."""
        if not interactive:
            return self._plot(*args, **kwargs)
        return SemblancePlot(self, *args, **kwargs).plot()


class Coherency(BaseCoherency):
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
    N - temporal window size,
    f_{j}(i, v) - the amplitude value on the `j`-th trace being NMO-corrected for time index `i` and velocity `v`. Thus
    the amplitude is taken for the time defined by :math:`t(i, v) = \sqrt{t_0^2 + \frac{l_j^2}{v^2}}`,
    where:
    :math:`t_0` - start time of the hyperbola associated with time index `i`,
    :math:`l_j` - offset of the `j`-th trace,
    :math:`v` - velocity value.
    The resulting matrix :math:`S(k, v)` has shape (trace_length, n_velocities) and contains vertical velocity
    semblance values based on hyperbolas with each combination of the starting point :math:`k` and velocity :math:`v`.
    The algorithm for semblance calculation looks as follows:
    For each velocity from given velocity range:
        1. Calculate NMO-corrected gather.
        2. Calculate squared sum of amplitudes of the corrected gather along the offset axis and its rolling mean in a
           temporal window with given `win_size`.
        3. Calculate sum of squared amplitudes of the corrected gather along the offset axis and its rolling mean in a
           temporal window with given `win_size`.
        4. Divide a value from step 2 by the value from step 3 for each time to get semblance values for selected
           velocity.
    Notes
    -----
    The gather should be sorted by offset.
    Examples
    --------
    Calculate semblance for 200 velocities from 2000 to 6000 m/s and a temporal window size of 8 samples:
    >>> survey = Survey(path, header_index=["INLINE_3D", "CROSSLINE_3D"], header_cols="offset")
    >>> gather = survey.sample_gather().sort(by="offset")
    >>> semblance = gather.calculate_semblance(velocities=np.linspace(2000, 6000, 200), win_size=8)
    Parameters
    ----------
    gather : Gather
        Seismic gather to calculate semblance for.
    velocities : 1d np.ndarray
        Range of velocity values for which semblance is calculated. Measured in meters/seconds.
    win_size : int, optional, defaults to 25
        Temporal window size used for semblance calculation. The higher the `win_size` is, the smoother the resulting
        semblance will be but to the detriment of small details. Measured in samples.
    Attributes
    ----------
    gather : Gather
        Seismic gather for which semblance calculation was called.
    gather_data : 2d np.ndarray
        Gather data for semblance calculation. The data is stored in a transposed form, compared to `Gather.data` due
        to performance reasons, so that `gather_data.shape` is (trace_length, num_traces).
    velocities : 1d np.ndarray
        Range of velocity values for which semblance was calculated. Measured in meters/seconds.
    win_size : int
        Temporal window size for smoothing the semblance. Measured in samples.
    semblance : 2d np.ndarray
        Array with calculated vertical velocity semblance values.
    """
    def __init__(self, gather, velocities, win_size=25, mode='semblance'):
        super().__init__(gather, win_size=win_size, mode=mode)
        self.velocities = velocities  # m/s
        velocities_ms = self.velocities / 1000  # from m/s to m/ms
        self.semblance = self._calc_semblance_numba(semblance_func=self.calc_single_velocity_semblance, coherency_func=self.coherency_func,
                                                    nmo_func=get_hodograph, gather_data=self.gather_data,
                                                    times=self.times, offsets=self.offsets, velocities=velocities_ms,
                                                    sample_rate=self.sample_rate, win_size=self.win_size)

    def get_time_velocity_by_indices(self, time_ix, velocity_ix):
        """Get time (in milliseconds) and velocity (in kilometers/seconds) by their indices (possibly non-integer) in
        semblance."""
        if (time_ix < 0) or (time_ix >= len(self.times)):
            time = None
        else:
            time = np.interp(time_ix, np.arange(len(self.times)), self.times)

        if (velocity_ix < 0) or (velocity_ix >= len(self.velocities)):
            velocity = None
        else:
            velocity = np.interp(velocity_ix, np.arange(len(self.velocities)), self.velocities)
            velocity /= 1000  # from m/s to m/ms

        return time, velocity

    @staticmethod
    @njit(nogil=True, fastmath=True, parallel=True)
    def _calc_semblance_numba(semblance_func, nmo_func, coherency_func, gather_data, times, offsets, velocities, 
                              sample_rate, win_size):
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
        semblance = np.empty((gather_data.shape[1], len(velocities)), dtype=np.float32)
        # TODO: use prange when fixed in numba
        for j in prange(len(velocities)):  # pylint: disable=consider-using-enumerate
            semblance[:, j] = semblance_func(nmo_func=nmo_func, coherency_func=coherency_func, gather_data=gather_data, times=times, offsets=offsets,
                                             velocity=velocities[j], sample_rate=sample_rate, win_size=win_size,
                                             t_min_ix=0, t_max_ix=gather_data.shape[1])
        return semblance

    def _plot(self, stacking_velocity=None, *, title="Semblance", x_ticker=None, y_ticker=None, grid=False,
              colorbar=True, ax=None, **kwargs):
        """Plot vertical velocity semblance."""
        # Add a stacking velocity line on the plot
        stacking_times_ix, stacking_velocities_ix = None, None
        if stacking_velocity is not None:
            stacking_times = stacking_velocity.times if stacking_velocity.times is not None else self.times
            stacking_velocities = stacking_velocity(stacking_times)
            stacking_times_ix = stacking_times / self.sample_rate
            stacking_velocities_ix = ((stacking_velocities - self.velocities[0]) /
                                      (self.velocities[-1] - self.velocities[0]) * self.semblance.shape[1])

        super()._plot(self.semblance, title=title, x_label="Velocity (m/s)", x_ticklabels=self.velocities,
                      x_ticker=x_ticker, y_ticklabels=self.times, y_ticker=y_ticker, ax=ax, grid=grid,
                      stacking_times_ix=stacking_times_ix, stacking_velocities_ix=stacking_velocities_ix,
                      colorbar=colorbar, **kwargs)
        return self

    @plotter(figsize=(10, 9), args_to_unpack="stacking_velocity")
    def plot(self, stacking_velocity=None, *, title="Semblance", interactive=False, **kwargs):
        """Plot vertical velocity semblance.
        Parameters
        ----------
        stacking_velocity : StackingVelocity or str, optional
            Stacking velocity to plot if given. If its sample rate is more than 50 ms, every point will be highlighted
            with a circle.
            May be `str` if plotted in a pipeline: in this case it defines a component with stacking velocities to use.
        title : str, optional, defaults to "Semblance"
            Plot title.
        x_ticker : dict, optional, defaults to None
            Parameters for ticks and ticklabels formatting for the x-axis; see `.utils.set_ticks` for more details.
        y_ticker : dict, optional, defaults to None
            Parameters for ticks and ticklabels formatting for the y-axis; see `.utils.set_ticks` for more details.
        grid : bool, optional, defaults to False
            Specifies whether to draw a grid on the plot.
        colorbar : bool or dict, optional, defaults to True
            Whether to add a colorbar to the right of the semblance plot. If `dict`, defines extra keyword arguments
            for `matplotlib.figure.Figure.colorbar`.
        ax : matplotlib.axes.Axes, optional, defaults to None
            Axes of the figure to plot on.
        kwargs : misc, optional
            Additional common keyword arguments for `x_ticker` and `y_tickers`.
        interactive : bool, optional, defaults to `False`
            Whether to plot semblance in interactive mode. This mode also plots the gather used to calculate the
            semblance. Clicking on semblance highlights the corresponding hodograph on the gather plot and allows
            performing NMO correction of the gather with the selected velocity. Interactive plotting must be performed
            in a JupyterLab environment with the the `%matplotlib widget` magic executed and `ipympl` and `ipywidgets`
            libraries installed.
        sharey : bool, optional, defaults to True, only for interactive mode
            Whether to share y axis of semblance and gather plots.
        gather_plot_kwargs : dict, optional, only for interactive mode
            Additional arguments to pass to `Gather.plot`.
        Returns
        -------
        semblance : Semblance
            Self unchanged.
        """
        return super().plot(stacking_velocity=stacking_velocity, interactive=interactive, title=title, **kwargs)

    @batch_method(target="for", copy_src=False)
    def calculate_stacking_velocity(self, start_velocity_range=(1400, 1800), end_velocity_range=(2500, 5000),
                                    max_acceleration=None, n_times=25, n_velocities=25):
        """Calculate stacking velocity by vertical velocity semblance.
        Notes
        -----
        A detailed description of the proposed algorithm and its implementation can be found in
        :func:`~velocity_model.calculate_stacking_velocity` docs.
        Parameters
        ----------
        start_velocity_range : tuple with 2 elements
            Valid range for stacking velocity for the first timestamp. Both velocities are measured in meters/seconds.
        end_velocity_range : tuple with 2 elements
            Valid range for stacking velocity for the last timestamp. Both velocities are measured in meters/seconds.
        max_acceleration : None or float, defaults to None
            Maximal acceleration allowed for the stacking velocity function. If `None`, equals to
            2 * (mean(end_velocity_range) - mean(start_velocity_range)) / total_time. Measured in meters/seconds^2.
        n_times : int, defaults to 25
            The number of evenly spaced points to split time range into to generate graph edges.
        n_velocities : int, defaults to 25
            The number of evenly spaced points to split velocity range into for each time to generate graph edges.
        Returns
        -------
        stacking_velocity : StackingVelocity
            Calculated stacking velocity.
        Raises
        ------
        ValueError
            If no stacking velocity was found for given parameters.
        """
        times, velocities, _ = calculate_stacking_velocity(self.semblance, self.times, self.velocities,
                                                           start_velocity_range, end_velocity_range, max_acceleration,
                                                           n_times, n_velocities)
        return StackingVelocity(times, velocities, coords=self.coords)


class ResidualCoherency(BaseCoherency):
    """A class for residual vertical velocity semblance calculation and processing.
    Residual semblance is a normalized output-input energy ratio for a CDP gather along picked stacking velocity. The
    method of its computation for given time and velocity completely coincides with the calculation of
    :class:`~Semblance`, however, residual semblance is computed in a small area around given stacking velocity, thus
    allowing for additional optimizations.
    The boundaries in which calculation is performed depend on time `t` and are given by:
    `stacking_velocity(t)` * (1 +- `relative_margin`).
    Since the length of this velocity range varies for different timestamps, the residual semblance values are
    interpolated to obtain a rectangular matrix of size (trace_length, max(right_boundary - left_boundary)), where
    `left_boundary` and `right_boundary` are arrays of left and right boundaries for all timestamps respectively.
    Thus the residual semblance is a function of time and relative velocity margin. Zero margin line corresponds to
    the given stacking velocity and generally should pass through local semblance maxima.
    Residual semblance instance can be created either directly by passing source gather, stacking velocity and other
    arguments to its init or by calling :func:`~Gather.calculate_residual_semblance` method (recommended way).
    Notes
    -----
    The gather should be sorted by offset.
    Examples
    --------
    First let's sample a CDP gather and sort it by offset:
    >>> survey = Survey(path, header_index=["INLINE_3D", "CROSSLINE_3D"], header_cols="offset")
    >>> gather = survey.sample_gather().sort(by="offset")
    Now let's automatically calculate stacking velocity by gather semblance:
    >>> semblance = gather.calculate_semblance(velocities=np.linspace(1400, 5000, 200), win_size=8)
    >>> velocity = semblance.calculate_stacking_velocity()
    Residual semblance for the gather and calculated stacking velocity can be obtained as follows:
    >>> residual = gather.calculate_residual_semblance(velocity, n_velocities=100, win_size=8)
    Parameters
    ----------
    gather : Gather
        Seismic gather to calculate residual semblance for.
    stacking_velocity : StackingVelocity
        Stacking velocity around which residual semblance is calculated.
    n_velocities : int, optional, defaults to 140
        The number of velocities to compute residual semblance for.
    win_size : int, optional, defaults to 25
        Temporal window size used for semblance calculation. The higher the `win_size` is, the smoother the resulting
        semblance will be but to the detriment of small details. Measured in samples.
    relative_margin : float, optional, defaults to 0.2
        Relative velocity margin, that determines the velocity range for semblance calculation for each time `t` as
        `stacking_velocity(t)` * (1 +- `relative_margin`).
    Attributes
    ----------
    gather : Gather
        Seismic gather for which residual semblance calculation was called.
    gather_data : 2d np.ndarray
        Gather data for semblance calculation. The data is stored in a transposed form, compared to `Gather.data` due
        to performance reasons, so that `gather_data.shape` is (trace_length, num_traces).
    velocities : 1d np.ndarray
        Range of velocity values for which residual semblance was calculated. Measured in meters/seconds.
    win_size : int
        Temporal window size for smoothing the semblance. Measured in samples.
    stacking_velocity : StackingVelocity
        Stacking velocity around which residual semblance was calculated.
    relative_margin : float, optional, defaults to 0.2
         Relative velocity margin, that determines the velocity range for semblance calculation for each timestamp.
    residual_semblance : 2d np.ndarray
         Array with calculated residual vertical velocity semblance values.
    """
    def __init__(self, gather, stacking_velocity, n_velocities=140, win_size=25, relative_margin=0.2, mode='semblance'):
        super().__init__(gather, win_size, mode)
        self.stacking_velocity = stacking_velocity
        self.relative_margin = relative_margin

        interpolated_velocities = stacking_velocity(self.times)
        self.velocities = np.linspace(np.min(interpolated_velocities) * (1 - relative_margin),
                                      np.max(interpolated_velocities) * (1 + relative_margin),
                                      n_velocities, dtype=np.float32)
        velocities_ms = self.velocities / 1000  # from m/s to m/ms

        left_bound_ix, right_bound_ix = self._calc_velocity_bounds()
        self.residual_semblance = self._calc_res_semblance_numba(semblance_func=self.calc_single_velocity_semblance, coherency_func=self.coherency_func,
                                                                 nmo_func=get_hodograph, gather_data=self.gather_data,
                                                                 times=self.times, offsets=self.offsets,
                                                                 velocities=velocities_ms,
                                                                 left_bound_ix=left_bound_ix,
                                                                 right_bound_ix=right_bound_ix,
                                                                 sample_rate=self.sample_rate, win_size=self.win_size)

    def get_time_velocity_by_indices(self, time_ix, velocity_ix):
        """Get time (in milliseconds) and velocity (in kilometers/seconds) by their indices (possibly non-integer) in
        residual semblance."""
        if (time_ix < 0) or (time_ix >= len(self.times)):
            return None, None
        time = np.interp(time_ix, np.arange(len(self.times)), self.times)
        center_velocity = self.stacking_velocity(time) / 1000  # from m/s to m/ms

        if (velocity_ix < 0) or (velocity_ix >= self.residual_semblance.shape[1]):
            return time, None
        margin = self.relative_margin * (2 * velocity_ix / (self.residual_semblance.shape[1] - 1) - 1)
        return time, center_velocity * (1 + margin)

    def _calc_velocity_bounds(self):
        """Calculate velocity boundaries for each time within which residual semblance will be calculated.
        Returns
        -------
        left_bound_ix : 1d array
            Indices of corresponding velocities of the left bound for each time.
        right_bound_ix : 1d array
            Indices of corresponding velocities of the right bound for each time.
        """
        interpolated_velocities = np.clip(self.stacking_velocity(self.times), self.velocities[0], self.velocities[-1])
        left_bound_values = (interpolated_velocities * (1 - self.relative_margin)).reshape(-1, 1)
        left_bound_ix = np.argmin(np.abs(left_bound_values - self.velocities), axis=1)
        right_bound_values = (interpolated_velocities * (1 + self.relative_margin)).reshape(-1, 1)
        right_bound_ix = np.argmin(np.abs(right_bound_values - self.velocities), axis=1)
        return left_bound_ix, right_bound_ix

    @staticmethod
    @njit(nogil=True, fastmath=True, parallel=True)
    def _calc_res_semblance_numba(semblance_func, nmo_func, coherency_func, gather_data, times, offsets, velocities, left_bound_ix,
                                  right_bound_ix, sample_rate, win_size):
        """Parallelized and njitted method for residual vertical velocity semblance calculation.
        Parameters
        ----------
        semblance_func : njitted callable
            Base function for semblance calculation for single velocity and a time range.
        nmo_func : njitted callable
            Base function for gather normal moveout correction for given time and velocity.
        left_bound_ix : 1d array
            Indices of corresponding velocities of the left bound for each time.
        right_bound_ix : 1d array
            Indices of corresponding velocities of the right bound for each time.
        other parameters : misc
            Passed directly from class attributes (except for velocities which are converted from m/s to m/ms)
        Returns
        -------
        semblance : 2d np.ndarray
            Array with residual vertical velocity semblance values.
        """
        semblance = np.zeros((len(gather_data), len(velocities)), dtype=np.float32)
        for i in range(left_bound_ix.min(), right_bound_ix.max() + 1):  # TODO: use prange when fixed in numba
            t_min_ix = np.where(right_bound_ix == i)[0]
            t_min_ix = 0 if len(t_min_ix) == 0 else t_min_ix[0]

            t_max_ix = np.where(left_bound_ix == i)[0]
            t_max_ix = len(times) - 1 if len(t_max_ix) == 0 else t_max_ix[-1]

            semblance[:, i][t_min_ix : t_max_ix+1] = semblance_func(nmo_func=nmo_func, coherency_func=coherency_func, gather_data=gather_data,
                                                                    times=times, offsets=offsets,
                                                                    velocity=velocities[i], sample_rate=sample_rate,
                                                                    win_size=win_size, t_min_ix=t_min_ix,
                                                                    t_max_ix=t_max_ix+1)

        # Interpolate semblance to get a rectangular image
        semblance_len = (right_bound_ix - left_bound_ix).max()
        residual_semblance = np.empty((len(times), semblance_len), dtype=np.float32)
        for i in prange(len(semblance)):
            cropped_semblance = semblance[i][left_bound_ix[i] : right_bound_ix[i] + 1]
            residual_semblance[i] = np.interp(np.linspace(0, len(cropped_semblance) - 1, semblance_len),
                                              np.arange(len(cropped_semblance)),
                                              cropped_semblance)
        return residual_semblance

    @batch_method(target="for", copy_src=False)
    def correct_stacking_velocity(self, kernel_size=1):
        """ Correct stacking velocity the way it follows the maximum coherency path.
        Parameters
        ----------
        kernel_size : int
            Median filter kernel size. Must be positive odd interger.
        Returns
        -------
            : StackingVelocity
            Corrected stacking velocity.
        """
        ind = np.argmax(self.residual_semblance, 1)
        center_ind = self.residual_semblance.shape[1] / 2
        delta = (ind - center_ind) / center_ind
        corrected_velocity = self.stacking_velocity(self.times) * (1 + delta * self.relative_margin)
        if kernel_size is not 1:
            corrected_velocity = median_filter(corrected_velocity, kernel_size)
        return StackingVelocity.from_points(self.times, corrected_velocity,
                                            self.stacking_velocity.inline, self.stacking_velocity.crossline)

    def _plot(self, *, title="Residual semblance", x_ticker=None, y_ticker=None, grid=False, colorbar=True, ax=None,
              **kwargs):
        """Plot residual vertical velocity semblance."""
        x_ticklabels = np.linspace(-self.relative_margin, self.relative_margin, self.residual_semblance.shape[1]) * 100

        stacking_times = self.stacking_velocity.times if self.stacking_velocity.times is not None else self.times
        stacking_times_ix = stacking_times / self.sample_rate
        stacking_velocities_ix = np.full_like(stacking_times_ix, self.residual_semblance.shape[1] / 2)

        super()._plot(self.residual_semblance, title=title, x_label="Relative velocity margin (%)",
                      x_ticklabels=x_ticklabels, x_ticker=x_ticker, y_ticklabels=self.times, y_ticker=y_ticker, ax=ax,
                      grid=grid, stacking_times_ix=stacking_times_ix, stacking_velocities_ix=stacking_velocities_ix,
                      colorbar=colorbar, **kwargs)
        return self

    @plotter(figsize=(10, 9))
    def plot(self, *, title="Residual semblance", interactive=False, **kwargs):
        """Plot residual vertical velocity semblance. The plot always has a vertical line in the middle, representing
        the stacking velocity it was calculated for.
        Parameters
        ----------
        title : str, optional, defaults to "Residual semblance"
            Plot title.
        x_ticker : dict, optional, defaults to None
            Parameters for ticks and ticklabels formatting for the x-axis; see `.utils.set_ticks` for more details.
        y_ticker : dict, optional, defaults to None
            Parameters for ticks and ticklabels formatting for the y-axis; see `.utils.set_ticks` for more details.
        grid : bool, optional, defaults to False
            Specifies whether to draw a grid on the plot.
        colorbar : bool or dict, optional, defaults to True
            Whether to add a colorbar to the right of the residual semblance plot. If `dict`, defines extra keyword
            arguments for `matplotlib.figure.Figure.colorbar`.
        ax : matplotlib.axes.Axes, optional, defaults to None
            Axes of the figure to plot on.
        kwargs : misc, optional
            Additional common keyword arguments for `x_ticker` and `y_tickers`.
        interactive : bool, optional, defaults to `False`
            Whether to plot residual semblance in interactive mode. This mode also plots the gather used to calculate
            the residual semblance. Clicking on residual semblance highlights the corresponding hodograph on the gather
            plot and allows performing NMO correction of the gather with the selected velocity. Interactive plotting
            must be performed in a JupyterLab environment with the the `%matplotlib widget` magic executed and `ipympl`
            and `ipywidgets` libraries installed.
        sharey : bool, optional, defaults to True, only for interactive mode
            Whether to share y axis of residual semblance and gather plots.
        gather_plot_kwargs : dict, optional, only for interactive mode
            Additional arguments to pass to `Gather.plot`.
        Returns
        -------
        semblance : ResidualSemblance
            Self unchanged.
        """
        return super().plot(interactive=interactive, title=title, **kwargs)
