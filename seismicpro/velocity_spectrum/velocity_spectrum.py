"""Implements VerticalVelocitySpectrum and ResidualVelocitySpectrum classes."""

# pylint: disable=not-an-iterable, too-many-arguments
import math

import numpy as np
from numba import njit, prange
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt


from .utils import coherency_funcs
from .interactive_plot import VelocitySpectrumPlot
from ..decorators import batch_method, plotter
from ..stacking_velocity import StackingVelocity, calculate_stacking_velocity
from ..utils import add_colorbar, set_ticks, set_text_formatting
from ..gather.utils import correction
from ..const import DEFAULT_STACKING_VELOCITY


COHERENCY_FUNCS = {
    "stacked_amplitude": coherency_funcs.stacked_amplitude,
    "S": coherency_funcs.stacked_amplitude,
    "normalized_stacked_amplitude": coherency_funcs.normalized_stacked_amplitude,
    "NS": coherency_funcs.normalized_stacked_amplitude,
    "semblance": coherency_funcs.semblance,
    "NE": coherency_funcs.semblance,
    'crosscorrelation': coherency_funcs.crosscorrelation,
    'CC': coherency_funcs.crosscorrelation,
    'ENCC': coherency_funcs.energy_normalized_crosscorrelation,
    'energy_normalized_crosscorrelation': coherency_funcs.energy_normalized_crosscorrelation
}


class BaseVelocitySpectrum:
    """Base class for vertical velocity spectrum calculation.
    Implements general computation logic and visualization method.

    Parameters
    ----------
    gather : Gather
        Seismic gather to calculate velocity spectrum for.
    window_size : float
        Temporal window size used for velocity spectrum calculation. The higher the `window_size` is, the smoother
        the resulting velocity spectrum will be but to the detriment of small details. Measured in milliseconds.
    mode: str, defaults to `semblance`
        The coherency measure. See the `COHERENCY_FUNCS` for avaliable options.
    max_stretch_factor : float, defaults to np.inf
        Max allowable factor for the muter that attenuates the effect of waveform stretching after nmo correction.
        This mute is applied after nmo correction for each provided velocity and before coherency calculation.
        The lower the value, the stronger the mute. In case np.inf(default) no mute is applied. 
        Reasonably good value is 0.65.

    Attributes
    ----------
    gather : Gather
        Seismic gather for which velocity spectrum calculation was called.
    half_win_size_samples : int
        Half of the temporal window size for smoothing the velocity spectrum. Measured in samples.
    coherency_func : callable
        The function that estimates the coherency measure for hodograph.
    max_stretch_factor: float
        Max allowable factor for stretch muter.
    """

    def __init__(self, gather, window_size, mode='semblance', max_stretch_factor=np.inf):
        self.gather = gather.copy()
        self.half_win_size_samples = math.ceil((window_size / gather.sample_rate / 2)
        self.max_stretch_factor = max_stretch_factor

        self.coherency_func = COHERENCY_FUNCS.get(mode)
        if self.coherency_func is None:
            raise ValueError(f"Unknown mode {mode}, avaliable modes are {COHERENCY_FUNCS.keys()}")

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
        """Coordinates or None: Spatial coordinates of the velocity_spectrum. Determined by the underlying gather.
        `None` if the gather is indexed by unsupported headers or required coords headers were not loaded
        or coordinates are non-unique for traces of the gather."""
        return self.gather.coords

    def get_time_velocity_by_indices(self, time_ix, velocity_ix):
        """Get time (in milliseconds) and velocity (in kilometers/seconds) by their indices (possibly non-integer) in
        velocity spectrum."""
        _ = time_ix, velocity_ix
        raise NotImplementedError

    @staticmethod
    @njit(nogil=True, fastmath=True, parallel=True)
    def calc_single_velocity_spectrum(coherency_func, gather_data, times, offsets, velocity, sample_rate,
                                       half_win_size_samples, t_min_ix, t_max_ix, max_stretch_factor=np.inf):
        """Calculate velocity spectrum for given velocity and time range.

        Parameters
        ----------
        coherency_func: njitted callable
            The function that estimates hodograph coherency.
        gather_data : 2d np.ndarray
            Gather data for velocity spectrum calculation.
        times : 1d np.ndarray
            Recording time for each trace value. Measured in milliseconds.
        offsets : array-like
            The distance between source and receiver for each trace. Measured in meters.
        velocity : array-like
            Seismic wave velocity for velocity spectrum computation. Measured in meters/milliseconds.
        sample_rate : float
            Sample rate of seismic traces. Measured in milliseconds.
        half_win_size_samples : int
            Half of the temporal size for smoothing the velocity spectrum. Measured in samples.
        t_min_ix : int
            Time index in `times` array to start calculating velocity spectrum from. Measured in samples.
        t_max_ix : int
            Time index in `times` array to stop calculating velocity spectrum at. Measured in samples.
        max_stretch_factor : float, defaults to np.inf
            Max allowable factor for the muter that attenuates the effect of waveform stretching after nmo correction.
            The lower the value, the stronger the mute. In case np.inf(default) no mute is applied. 
            Reasonably good value is 0.65.

        Returns
        -------
        velocity_spectrum_slice : 1d np.ndarray
            Calculated velocity spectrum values for a specified `velocity` in time range from `t_min_ix` to `t_max_ix`.
        """
        t_win_size_min_ix = max(0, t_min_ix - half_win_size_samples)
        t_win_size_max_ix = min(len(times) - 1, t_max_ix + half_win_size_samples)

        corrected_gather_data = correction.apply_nmo(gather_data, times[t_win_size_min_ix: t_win_size_max_ix + 1],
                                                     offsets, velocity, sample_rate, mute_crossover=False,
                                                     max_stretch_factor=max_stretch_factor)

        numerator, denominator = coherency_func(corrected_gather_data)

        velocity_spectrum_slice = np.empty(t_max_ix - t_min_ix, dtype=np.float32)
        for t in prange(t_min_ix, t_max_ix):
            t_rel = t - t_win_size_min_ix
            ix_from = max(0, t_rel - half_win_size_samples)
            ix_to = min(corrected_gather_data.shape[1] - 1, t_rel + half_win_size_samples)
            velocity_spectrum_slice[t - t_min_ix] = np.sum(numerator[ix_from : ix_to]) / \
                                                    (np.sum(denominator[ix_from : ix_to]) + 1e-8)
        return velocity_spectrum_slice

    def _plot(self, title=None, x_label=None, x_ticklabels=None,
              x_ticker=None, y_ticklabels=None, y_ticker=None, grid=False, stacking_times_ix=None,
              stacking_velocities_ix=None, colorbar=True, clip_threshold_quantile=0.99, n_levels=10, ax=None, **kwargs):
        """Plot vertical velocity spectrum and, optionally, stacking velocity.

        Parameters
        ----------
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
            Whether to add a colorbar to the right of the velocity spectrum plot. 
            If `dict`, defines extra keyword arguments for `matplotlib.figure.Figure.colorbar`.
        clip_threshold_quantile : float, optional, defaults to 0.99
            Clip the velocity spectrum values by given quantile.
        n_levels: int, optional, defaluts to 10
            The number of levels on the colorbar.
        ax : matplotlib.axes.Axes, optional, defaults to None
            Axes of the figure to plot on.
        kwargs : misc, optional
            Additional common keyword arguments for `x_ticker` and `y_tickers`.
        """
        # Cast text-related parameters to dicts and add text formatting parameters from kwargs to each of them
        (title, x_ticker, y_ticker), kwargs = set_text_formatting(title, x_ticker, y_ticker, **kwargs)

        cmap = plt.get_cmap('seismic')
        level_values = np.linspace(0, np.quantile(self.velocity_spectrum, clip_threshold_quantile), n_levels)
        norm = mcolors.BoundaryNorm(level_values, cmap.N, clip=True)
        img = ax.imshow(self.velocity_spectrum, norm=norm, aspect='auto', cmap=cmap)
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
        """Plot velocity spectrum in interactive or non-interactive mode."""
        if not interactive:
            return self._plot(*args, **kwargs)
        return VelocitySpectrumPlot(self, *args, **kwargs).plot()


class VerticalVelocitySpectrum(BaseVelocitySpectrum):
    r"""A class for vertical velocity spectrum calculation and processing.

    Velocity spectrum is a measure for hodograph coherency. The higher the values of velocity spectrum are, the
    more coherent the signal is along a hyperbolic trajectory over the entire spread length of the gather.

    Velocity spectrum instance can be created either directly by passing source gather(and optional parameters like 
    velocity range, window size, coherency measure and factor for stretch mute) to its init
    or by calling :func:`~Gather.calculate_vertical_velocity_spectrum` method (recommended way).

    The velocity spectrum is computed by:
    :math:`VS(k, v) = \frac{\sum^{k+N/2}_{i=k-N/2} numerator(i, v)}
                           {\sum^{k+N/2}_{i=k-N/2} denominator(i, v)},
    where:

     - VS - velocity spectrum value for starting time index `k` and velocity `v`,
     - N - temporal window size,
     - numerator(i, v) - numerator of the coherency measure,
     - denominator(i, v) - denominator of the coherency measure,

    For different coherency measures the numerator and denominator calculated as follows:

    - Stacked Amplitude:
        numerator(i, v) = sum^{M-1}_{j=0} f_{j}(i, v)^2
        denominator(i, v) = 1

    - Energy Normalized Crosscorrelation:
        numerator(i, v) = (sum^{M-1}_{j=0} f_{j}(i, v))^2 - sum^{M-1}_{j=0} f_{j}(i, v)^2
        denominator(i, v) = (M - 1) * (sum^{M-1}_{j=0} f_{j}(i, v)^2)        

    - Semblance:
        numerator(i, v) = (sum^{M-1}_{j=0} f_{j}(i, v))^2
        denominator(i, v) = M * sum^{M-1}_{j=0} f_{j}(i, v)^2
    where:

    f_{j}(i, v) - the amplitude value on the `j`-th trace being NMO-corrected for time index `i` and velocity `v`. Thus
    the amplitude is taken for the time defined by :math:`t(i, v) = \sqrt{t_0^2 + \frac{l_j^2}{v^2}}`,
    where:

    :math:`t_0` - start time of the hyperbola associated with time index `i`,
    :math:`l_j` - offset of the `j`-th trace,
    :math:`v` - velocity value.

    See the COHERENCY_FUNCS for the full list available coherency measures

    The resulting matrix :math:`VS(k, v)` has shape (trace_length, n_velocities) and contains vertical velocity
    spectrum values based on hyperbolas with each combination of the starting point :math:`k` and velocity :math:`v`.

    The algorithm for velocity spectrum calculation looks as follows:
    For each velocity from given velocity range:
        1. Calculate NMO-corrected gather.
        2. Estimate numerator and denominator for given coherency measure for each timestamp.
        3. Get the velocity spectrum values as the division of rolling sums in a 
        temporal windows of numerator and denominator.

    Examples
    --------
    Calculate velocity spectrum for 200 velocities from 2000 to 6000 m/s and a temporal window size of 16 ms:
    >>> survey = Survey(path, header_index=["INLINE_3D", "CROSSLINE_3D"], header_cols="offset")
    >>> gather = survey.sample_gather().sort(by="offset")
    >>> velocity_spectrum = gather.calculate_vertical_velocity_spectrum(velocities=np.linspace(2000, 6000, 200),
                                                                        window_size=16)

    Parameters
    ----------
    gather : Gather
        Seismic gather to calculate velocity spectrum for.
    velocities : 1d np.ndarray, optional, defaults to None
        Range of velocity values for which velocity spectrum is calculated. Measured in meters/seconds.
        If not provided, velocities evenly sampled from  `const.DEFAULT_STACKING_VELOCITY` with step 100 m/s.
    window_size : int, optional, defaults to 50
        Temporal window size used for velocity spectrum calculation. The higher the `window_size` is, the smoother
        the resulting velocity spectrum will be but to the detriment of small details. Measured in miliseconds.
    mode: str, optional, defaults to 'semblance'
        The measure for estimating hodograph coherency. 
        The available options are: 
            `semblance` or `NE`,
            `stacked_amplitude` or `S`,
            `normalized_stacked_amplitude` or `NS`,
            `crosscorrelation` or `CC`,
            `energy_normalized_crosscorrelation` or `ENCC`
    max_stretch_factor : float, defaults to np.inf
        Max allowable factor for the muter that attenuates the effect of waveform stretching after nmo correction.
        This mute is applied after nmo correction for each provided velocity and before coherency calculation.
        The lower the value, the stronger the mute. In case np.inf(default) no mute is applied. 
        Reasonably good value is 0.65.

    Attributes
    ----------
    gather : Gather
        Seismic gather for which velocity spectrum calculation was called.
    velocities : 1d np.ndarray
        Range of velocity values for which vertical velocity spectrum was calculated. Measured in meters/seconds.
    half_win_size_samples : int
        Half of the temporal window size for smoothing the vertical velocity spectrum. Measured in samples.
    velocity_spectrum : 2d np.ndarray
        Array with calculated vertical velocity spectrum values.
    max_stretch_factor: float
        Max allowable factor for stretch muter.
    """
    def __init__(self, gather, velocities=None, window_size=50, mode='semblance', max_stretch_factor=np.inf):
        super().__init__(gather, window_size, mode, max_stretch_factor)
        if velocities is not None:
            self.velocities = velocities  # m/s
        else:
            self.velocities = np.arange(DEFAULT_STACKING_VELOCITY(gather.times[0]) * 0.8,
                                        DEFAULT_STACKING_VELOCITY(gather.times[-1]) * 1.2,
                                        100)
        velocities_ms = self.velocities / 1000  # from m/s to m/ms
        self.velocity_spectrum = self._calc_spectrum_numba(
                                                spectrum_func=self.calc_single_velocity_spectrum,
                                                coherency_func=self.coherency_func, gather_data=self.gather.data,
                                                times=self.times, offsets=self.offsets, velocities=velocities_ms,
                                                sample_rate=self.sample_rate, max_stretch_factor=max_stretch_factor,
                                                half_win_size_samples=self.half_win_size_samples)

    def get_time_velocity_by_indices(self, time_ix, velocity_ix):
        """Get time (in milliseconds) and velocity (in kilometers/seconds) by their indices (possibly non-integer) in
        velocity spectrum."""
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
    def _calc_spectrum_numba(spectrum_func, coherency_func, gather_data, times, offsets, velocities,
                            sample_rate, half_win_size_samples, max_stretch_factor):
        """Parallelized and njitted method for vertical velocity spectrum calculation.

        Parameters
        ----------
        spectrum_func : njitted callable
            Base function for velocity spectrum calculation for single velocity and a time range.
        other parameters : misc
            Passed directly from class attributes (except for velocities which are converted from m/s to m/ms)

        Returns
        -------
        velocity_spectrum : 2d np.ndarray
            Array with vertical velocity spectrum values.
        """
        velocity_spectrum = np.empty((gather_data.shape[1], len(velocities)), dtype=np.float32)
        for j in prange(len(velocities)):  # pylint: disable=consider-using-enumerate
            velocity_spectrum[:, j] = spectrum_func(coherency_func=coherency_func, gather_data=gather_data,
                                                    times=times, offsets=offsets, velocity=velocities[j],
                                                    half_win_size_samples=half_win_size_samples,
                                                    t_min_ix=0, t_max_ix=gather_data.shape[1],
                                                    sample_rate=sample_rate, max_stretch_factor=max_stretch_factor)
        return velocity_spectrum

    def _plot(self, stacking_velocity=None, *, title=None, x_ticker=None, y_ticker=None, grid=False, colorbar=True,
              ax=None, **kwargs):
        """Plot vertical velocity spectrum."""
        # Add a stacking velocity line on the plot
        stacking_times_ix, stacking_velocities_ix = None, None
        if stacking_velocity is not None:
            stacking_times = stacking_velocity.times[stacking_velocity.times <= self.times[-1]]
            stacking_velocities = stacking_velocity(stacking_times)
            stacking_times_ix = stacking_times / self.sample_rate
            stacking_velocities_ix = ((stacking_velocities - self.velocities[0]) /
                                      (self.velocities[-1] - self.velocities[0]) * self.velocity_spectrum.shape[1])

        super()._plot(title=title, x_label="Velocity, m/s", x_ticklabels=self.velocities,
                      x_ticker=x_ticker, y_ticklabels=self.times, y_ticker=y_ticker, ax=ax, grid=grid,
                      stacking_times_ix=stacking_times_ix, stacking_velocities_ix=stacking_velocities_ix,
                      colorbar=colorbar, **kwargs)
        return self

    @plotter(figsize=(10, 9), args_to_unpack="stacking_velocity")
    def plot(self, stacking_velocity=None, *, title=None, interactive=False, **kwargs):
        """Plot vertical velocity spectrum.

        Parameters
        ----------
        stacking_velocity : StackingVelocity or str, optional
            Stacking velocity to plot if given. If its sample rate is more than 50 ms, every point will be highlighted
            with a circle.
            May be `str` if plotted in a pipeline: in this case it defines a component with stacking velocities to use.
        title : str, optional
            Plot title. If not provided, equals to stacked lines "Vertical Velocity Spectrum" and coherency func name.
        x_ticker : dict, optional, defaults to None
            Parameters for ticks and ticklabels formatting for the x-axis; see `.utils.set_ticks` for more details.
        y_ticker : dict, optional, defaults to None
            Parameters for ticks and ticklabels formatting for the y-axis; see `.utils.set_ticks` for more details.
        grid : bool, optional, defaults to False
            Specifies whether to draw a grid on the plot.
        colorbar : bool or dict, optional, defaults to True
            Whether to add a colorbar to the right of the velocity spectrum plot.
            If `dict`, defines extra keyword arguments for `matplotlib.figure.Figure.colorbar`.
        clip_threshold_quantile : float, optional, defaults to 0.99
            Clip the velocity spectrum values by given quantile.
        n_levels: int, optional, defaluts to 10
            The number of levels on the colorbar.
        ax : matplotlib.axes.Axes, optional, defaults to None
            Axes of the figure to plot on.
        kwargs : misc, optional
            Additional common keyword arguments for `x_ticker` and `y_tickers`.
        interactive : bool, optional, defaults to `False`
            Whether to plot velocity spectrum in interactive mode. This mode also plots the gather used to calculate the
            velocity spectrum. Clicking on velocity spectrum highlights the corresponding hodograph on the gather plot
            and allows performing NMO correction of the gather with the selected velocity. 
            Interactive plotting must be performed in a JupyterLab environment with the `%matplotlib widget` 
            magic executed and `ipympl` and `ipywidgets` libraries installed.
        sharey : bool, optional, defaults to True, only for interactive mode
            Whether to share y axis of velocity spectrum and gather plots.
        gather_plot_kwargs : dict, optional, only for interactive mode
            Additional arguments to pass to `Gather.plot`.

        Returns
        -------
        velocity_spectrum : VerticalVelocitySpectrum
            Self unchanged.
        """
        if title is None:
            title = f"Vertical Velocity Spectrum \n Coherency func: {self.coherency_func.__name__}"
        return super().plot(stacking_velocity=stacking_velocity, interactive=interactive, title=title, **kwargs)

    @batch_method(target="for", copy_src=False)
    def calculate_stacking_velocity(self, start_velocity_range=(1400, 1800), end_velocity_range=(2500, 5000),
                                    max_acceleration=None, n_times=25, n_velocities=25):
        """Calculate stacking velocity by vertical velocity spectrum.

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
        times, velocities, _ = calculate_stacking_velocity(self.velocity_spectrum, self.times, self.velocities,
                                                           start_velocity_range, end_velocity_range, max_acceleration,
                                                           n_times, n_velocities)
        return StackingVelocity(times, velocities, coords=self.coords)


class ResidualVelocitySpectrum(BaseVelocitySpectrum):
    """A class for residual vertical velocity spectrum calculation and processing.

    Residual velocity spectrum is hodograph coherency measure for a CDP gather along picked stacking velocity. The
    method of its computation for given time and velocity completely coincides with the calculation of
    :class:`~VerticalVelocitySpectrum`, however, residual velocity spectrum is computed in a small area around given 
    stacking velocity, thus allowing for additional optimizations.

    The boundaries in which calculation is performed depend on time `t` and are given by:
    `stacking_velocity(t)` * (1 +- `relative_margin`).

    Since the length of this velocity range varies for different timestamps, the residual velocity spectrum values are
    interpolated to obtain a rectangular matrix of size (trace_length, max(right_boundary - left_boundary)), where
    `left_boundary` and `right_boundary` are arrays of left and right boundaries for all timestamps respectively.

    Thus the residual velocity spectrum is a function of time and relative velocity margin. Zero margin line 
    corresponds to the given stacking velocity and generally should pass through local velocity spectrum maxima.

    Residual velocity spectrum instance can be created either directly by passing gather, stacking velocity and other
    arguments to its init or by calling :func:`~Gather.calculate_residual_velocity_spectrum` method (recommended way).

    Examples
    --------
    First let's sample a CDP gather and sort it by offset:
    >>> survey = Survey(path, header_index=["INLINE_3D", "CROSSLINE_3D"], header_cols="offset")
    >>> gather = survey.sample_gather().sort(by="offset")

    Now let's automatically calculate stacking velocity by gather velocity spectrum:
    >>> velocity_spectrum = gather.calculate_vertical_velocity_spectrum()
    >>> velocity = velocity_spectrum.calculate_stacking_velocity()

    Residual velocity spectrum for the gather and calculated stacking velocity can be obtained as follows:
    >>> residual = gather.calculate_residual_velocity_spectrum(velocity, n_velocities=100)

    Parameters
    ----------
    gather : Gather
        Seismic gather to calculate residual velocity spectrum for.
    stacking_velocity : StackingVelocity
        Stacking velocity around which residual velocity spectrum is calculated.
    n_velocities : int, optional, defaults to 140
        The number of velocities to compute residual velocity spectrum for.
    window_size : int, optional, defaults to 50
        Temporal window size used for velocity spectrum calculation. The higher the `window_size` is, the smoother
        the resulting velocity spectrum will be but to the detriment of small details. Measured in miliseconds.
    relative_margin : float, optional, defaults to 0.2
        Relative velocity margin, that determines the velocity range for velocity spectrum calculation
        for each time `t` as `stacking_velocity(t)` * (1 +- `relative_margin`).
    mode: str, optional, defaults to 'semblance'
        The measure for estimating hodograph coherency. 
        The available options are: 
            `semblance`, 
            `stacked_amplitude`,
            `normalized_stacked_amplitude`,
            `crosscorrelation`
            `energy_normalized_crosscorrelation`
    max_stretch_factor : float, defaults to np.inf
        Max allowable factor for the muter that attenuates the effect of waveform stretching after nmo correction.
        This mute is applied after nmo correction for each provided velocity and before coherency calculation.
        The lower the value, the stronger the mute. In case np.inf(default) no mute is applied. 
        Reasonably good value is 0.65

    Attributes
    ----------
    gather : Gather
        Seismic gather for which residual velocity spectrum calculation was called.
    velocities : 1d np.ndarray
        Range of velocity values for which residual velocity spectrum was calculated. Measured in meters/seconds.
    half_win_size_samples : int
        Half of the temporal window size for smoothing the velocity spectrum. Measured in samples.
    stacking_velocity : StackingVelocity
        Stacking velocity around which residual velocity spectrum was calculated.
    relative_margin : float, optional, defaults to 0.2
         Relative velocity margin, that determines the velocity range for velocity spectrum calculation for each time.
    velocity_spectrum : 2d np.ndarray
         Array with calculated residual vertical velocity velocity_spectrum values.
    max_stretch_factor: float
        Max allowable factor for stretch muter.
    """
    def __init__(self, gather, stacking_velocity, n_velocities=140, window_size=50, relative_margin=0.2,
                mode='semblance', max_stretch_factor=np.inf):
        super().__init__(gather, window_size, mode, max_stretch_factor)
        self.stacking_velocity = stacking_velocity
        self.relative_margin = relative_margin

        interpolated_velocities = stacking_velocity(self.times)
        self.velocities = np.linspace(np.min(interpolated_velocities) * (1 - relative_margin),
                                      np.max(interpolated_velocities) * (1 + relative_margin),
                                      n_velocities, dtype=np.float32)
        velocities_ms = self.velocities / 1000  # from m/s to m/ms

        left_bound_ix, right_bound_ix = self._calc_velocity_bounds()
        self.velocity_spectrum = self._calc_res_velocity_spectrum_numba(
                                                self.calc_single_velocity_spectrum, coherency_func=self.coherency_func,
                                                gather_data=self.gather.data, times=self.times,
                                                offsets=self.offsets, velocities=velocities_ms,
                                                left_bound_ix=left_bound_ix, right_bound_ix=right_bound_ix,
                                                half_win_size_samples=self.half_win_size_samples,
                                                sample_rate=self.sample_rate, max_stretch_factor=max_stretch_factor)

    def get_time_velocity_by_indices(self, time_ix, velocity_ix):
        """Get time (in milliseconds) and velocity (in kilometers/seconds) by their indices (possibly non-integer) in
        residual velocity spectrum."""
        if (time_ix < 0) or (time_ix >= len(self.times)):
            return None, None
        time = np.interp(time_ix, np.arange(len(self.times)), self.times)
        center_velocity = self.stacking_velocity(time) / 1000  # from m/s to m/ms

        if (velocity_ix < 0) or (velocity_ix >= self.velocity_spectrum.shape[1]):
            return time, None
        margin = self.relative_margin * (2 * velocity_ix / (self.velocity_spectrum.shape[1] - 1) - 1)
        return time, center_velocity * (1 + margin)

    def _calc_velocity_bounds(self):
        """Calculate velocity boundaries for each time within which residual velocity spectrum will be calculated.

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
    def _calc_res_velocity_spectrum_numba(spectrum_func, coherency_func, gather_data, times, offsets, velocities,
                                          left_bound_ix, right_bound_ix, sample_rate, half_win_size_samples,
                                          max_stretch_factor):
        """Parallelized and njitted method for residual vertical velocity spectrum calculation.

        Parameters
        ----------
        spectrum_func : njitted callable
            Base function for velocity spectrum calculation for single velocity and a time range.
        coherency_func : njitted callable
            Function for estimating hodograph coherency.
        left_bound_ix : 1d array
            Indices of corresponding velocities of the left bound for each time.
        right_bound_ix : 1d array
            Indices of corresponding velocities of the right bound for each time.
        other parameters : misc
            Passed directly from class attributes (except for velocities which are converted from m/s to m/ms)

        Returns
        -------
        velocity_spectrum : 2d np.ndarray
            Array with residual vertical velocity spectrum values.
        """
        velocity_spectrum = np.zeros((gather_data.shape[1], len(velocities)), dtype=np.float32)
        for i in prange(left_bound_ix.min(), right_bound_ix.max() + 1):
            t_min_ix = np.where(right_bound_ix == i)[0]
            t_min_ix = 0 if len(t_min_ix) == 0 else t_min_ix[0]

            t_max_ix = np.where(left_bound_ix == i)[0]
            t_max_ix = len(times) - 1 if len(t_max_ix) == 0 else t_max_ix[-1]

            velocity_spectrum[t_min_ix : t_max_ix+1, i] = spectrum_func(
                                                                coherency_func=coherency_func,
                                                                gather_data=gather_data, times=times, offsets=offsets,
                                                                velocity=velocities[i], sample_rate=sample_rate,
                                                                half_win_size_samples=half_win_size_samples,
                                                                t_min_ix=t_min_ix, t_max_ix=t_max_ix+1,
                                                                max_stretch_factor=max_stretch_factor)

        # Interpolate velocity spectrum to get a rectangular image
        residual_velocity_spectrum_len = (right_bound_ix - left_bound_ix).max()
        residual_velocity_spectrum = np.empty((len(times), residual_velocity_spectrum_len), dtype=np.float32)
        for i in prange(len(residual_velocity_spectrum)):
            cropped_velocity_spectrum = velocity_spectrum[i, left_bound_ix[i] : right_bound_ix[i] + 1]
            x = np.linspace(0, len(cropped_velocity_spectrum) - 1, residual_velocity_spectrum_len)
            residual_velocity_spectrum[i] = np.interp(x, np.arange(len(cropped_velocity_spectrum)),
                                                      cropped_velocity_spectrum)
        return residual_velocity_spectrum

    def _plot(self, *, title=None, x_ticker=None, y_ticker=None, grid=False, colorbar=True, ax=None, **kwargs):
        """Plot residual vertical velocity spectrum."""
        x_ticklabels = np.linspace(-self.relative_margin, self.relative_margin, self.velocity_spectrum.shape[1]) * 100

        stacking_times = self.stacking_velocity.times[self.stacking_velocity.times <= self.times[-1]]
        stacking_times_ix = stacking_times / self.sample_rate
        stacking_velocities_ix = np.full_like(stacking_times_ix, self.velocity_spectrum.shape[1] / 2)

        super()._plot(title=title, x_label="Relative velocity margin, %",
                      x_ticklabels=x_ticklabels, x_ticker=x_ticker, y_ticklabels=self.times, y_ticker=y_ticker, ax=ax,
                      grid=grid, stacking_times_ix=stacking_times_ix, stacking_velocities_ix=stacking_velocities_ix,
                      colorbar=colorbar, **kwargs)
        return self

    @plotter(figsize=(10, 9))
    def plot(self, *, title=None, interactive=False, **kwargs):
        """Plot residual vertical velocity spectrum. The plot always has a vertical line in the middle, representing
        the stacking velocity it was calculated for.

        Parameters
        ----------
        title : str, optional
            Plot title. If not provided, equals to stacked lines "Residual Velocity Spectrum" and coherency func name.
        x_ticker : dict, optional, defaults to None
            Parameters for ticks and ticklabels formatting for the x-axis; see `.utils.set_ticks` for more details.
        y_ticker : dict, optional, defaults to None
            Parameters for ticks and ticklabels formatting for the y-axis; see `.utils.set_ticks` for more details.
        grid : bool, optional, defaults to False
            Specifies whether to draw a grid on the plot.
        colorbar : bool or dict, optional, defaults to True
            Whether to add a colorbar to the right of the residual velocity spectrum plot.
            If `dict`, defines extra keyword arguments for `matplotlib.figure.Figure.colorbar`.
        clip_threshold_quantile : float, optional, defaults to 0.99
            Clip the residual velocity spectrum values by given quantile.
        n_levels: int, optional, defaluts to 10
            The number of levels on the colorbar.
        ax : matplotlib.axes.Axes, optional, defaults to None
            Axes of the figure to plot on.
        kwargs : misc, optional
            Additional common keyword arguments for `x_ticker` and `y_tickers`.
        interactive : bool, optional, defaults to `False`
            Whether to plot residual velocity spectrum in interactive mode. This mode also plots the gather used to
            calculate the residual velocity spectrum. Clicking on residual velocity spectrum highlights the
            corresponding hodograph on the gather plot and allows performing NMO correction of the gather with
            the selected velocity. Interactive plotting must be performed in a JupyterLab environment
            with the `%matplotlib widget` magic executed and `ipympl` and `ipywidgets` libraries installed.
        sharey : bool, optional, defaults to True, only for interactive mode
            Whether to share y axis of residual velocity spectrum and gather plots.
        gather_plot_kwargs : dict, optional, only for interactive mode
            Additional arguments to pass to `Gather.plot`.

        Returns
        -------
        velocity_spectrum : ResidualVelocitySpectrum
            Self unchanged.
        """
        if title is None:
            title = f"Residual Velocity Spectrum \n Coherency func: {self.coherency_func.__name__}"
        return super().plot(interactive=interactive, title=title, **kwargs)
