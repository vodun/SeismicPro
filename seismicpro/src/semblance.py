""" File contains classes for velocity analysis."""
# pylint: disable=not-an-iterable
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from scipy.interpolate import interp1d
from matplotlib import colors as mcolors

from .plot_utils import _set_ticks


class BaseSemblance:
    """ Base class for velocity analysis. """
    def __init__(self, seismogram, times, offsets, velocities, window):
        self._seismogram = np.ascontiguousarray(seismogram.T)
        self._times = times # ms
        self._offsets = offsets # m
        self._velocities = velocities # m/s
        self._samples_step = self._times[1] - self._times[0]
        self._window = window

    @staticmethod
    @njit(nogil=True, fastmath=True, parallel=True)
    def base_calc_semblance(seismogram, semblance, ix, times, offsets, velocity, samples_step, # pylint: disable=too-many-arguments
                            window, calc_nmo_func, nmo, t_min, t_max, t_window_min, t_window_max):
        """ !! """
        for i in prange(t_window_min, t_window_max):
            nmo[i - t_window_min] = calc_nmo_func(seismogram, times[i], offsets, velocity, samples_step)

        numerator = np.sum(nmo, axis=1)**2
        denominator = np.sum(nmo**2, axis=1)
        for t in prange(t_min, t_max):
            t_rel = t - t_window_min
            ix_from = max(0, t_rel - window)
            ix_to = min(len(nmo) - 1, t_rel + window)
            semblance[t, ix] = (np.sum(numerator[ix_from : ix_to]) /
                                (len(offsets) * np.sum(denominator[ix_from : ix_to])
                                 + 1e-6))

    @staticmethod
    @njit(nogil=True, fastmath=True)
    def base_calc_nmo(seismogram, time, offsets, velocity, samples_step):
        """ correct time"""
        corrected_seismogram = np.zeros(len(offsets))
        corrected_times = (np.sqrt(time**2 + offsets**2/velocity**2) / samples_step).astype(np.int32)
        for i in range(len(offsets)):
            corrected_time = corrected_times[i]
            if corrected_time < len(seismogram):
                corrected_seismogram[i] = seismogram[corrected_time, i]
        return corrected_seismogram

    # TODO: sort args, add docs, delete func form plot_utils
    def plot(self, semblance, ticks_range_x, ticks_range_y, xlabel='', figsize=(15, 12), title='', index='', fontsize=11,
             x_points=None, y_points=None,  grid=None, save_to=None, dpi=300, **kwargs):
        """ !! """
        # Split range of semblance on specific levels. Probably the levels are gonna scared
        # unprepared person but i found the result based on this levels the most attractive.
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

        if title or index:
            ax.set_title('{} {}'.format(title, index), fontsize=fontsize)

        # Change marker of velocity points if they are set at distance from each other.
        # This avoid dots in every point, if velocity law is set for every time.
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

class Semblance(BaseSemblance):
    """!!"""
    def __init__(self, seismogram, times, offsets, velocities, window=25):
        super().__init__(seismogram=seismogram, times=times, offsets=offsets,
                         velocities=velocities, window=window)
        self._semblance = None
        self._calc_semblance()

    @property
    def semblance(self):
        """!!"""
        return self._semblance.copy()

    def _calc_semblance(self):
        """ semblance """
        velocities_ms = self._velocities / 1000 # from m/s to m/ms
        self._semblance = self.calc_numba_semblance(base_func=self.base_calc_semblance,
                                                    calc_nmo_func=self.base_calc_nmo,
                                                    seismogram=self._seismogram, times=self._times,
                                                    offsets=self._offsets, velocities=velocities_ms,
                                                    samples_step=self._samples_step, window=self._window)

    @staticmethod
    @njit(nogil=True, fastmath=True, parallel=True)
    def calc_numba_semblance(base_func, calc_nmo_func, seismogram, times, offsets, velocities, samples_step, window):
        """ calculate semblance """
        semblance = np.empty((len(seismogram), len(velocities)))
        for j in prange(len(velocities)):
            nmo = np.empty_like(seismogram)
            base_func(seismogram=seismogram, semblance=semblance, ix=j, times=times, offsets=offsets,
                      velocity=velocities[j], samples_step=samples_step, window=window, calc_nmo_func=calc_nmo_func,
                      nmo=nmo, t_min=0, t_max=len(nmo), t_window_min=0, t_window_max=len(times))
        return semblance

    def plot(self, stacking_velocity=None, **kwargs):
        """ !! """
        x_points, y_points = None, None
        # Add a velocity line on semblance.
        if stacking_velocity is not None:
            # Find the coordinates on the graph that correspond to a certain velocity.
            stacking_velocity = np.asarray(stacking_velocity)
            x_points = ((stacking_velocity[:, 1] - self._velocities[0]) /
                        (self._velocities[-1] - self._velocities[0]) * self.semblance.shape[1])
            y_points = stacking_velocity[:, 0] / self._samples_step
        ticks_range_y = [0, self.semblance.shape[0] * self._samples_step]
        ticks_range_x = [self._velocities[0], self._velocities[-1]]
        super().plot(self.semblance, ticks_range_x, ticks_range_y, x_points=x_points, y_points=y_points, xlabel='Velocity (m/s)', **kwargs)

    def calc_minmax_metrics(self, other):
        """" other is a raw semblance here, while self is a diff. """
        minmax_self = np.max(np.max(self.semblance, axis=1) - np.min(self.semblance, axis=1))
        minmax_other = np.max(np.max(other.semblance, axis=1) - np.min(other.semblance, axis=1))
        return minmax_self / (minmax_other + 1e-11)


class ResidualSemblance(BaseSemblance):
    """!!"""
    def __init__(self, seismogram, times, offsets, velocities, stacking_velocity, window=25, deviation=0.2):
        super().__init__(seismogram, times, offsets, velocities, window)
        self._residual_semblance = None
        self._stacking_velocity = stacking_velocity
        self._deviation = deviation

        self._calc_residual_semblance()

    @property
    def residual_semblance(self):
        """ !! """
        return self._residual_semblance.copy()

    def _calc_residual_semblance(self):
        """ semblance """
        velocities_ms = self._velocities / 1000 # from m/s to m/ms
        stacking_velocity_ms = self._stacking_velocity.copy()
        stacking_velocity_ms[:, 1] /= 1000 # from m/s to m/ms

        left_bounds, right_bounds = self._calc_velocity_bounds()
        self._residual_semblance = self._calc_numba_res_semblance(base_func=self.base_calc_semblance,
                                                        calc_nmo_func=self.base_calc_nmo,
                                                        seismogram=self._seismogram, times=self._times,
                                                        offsets=self._offsets, velocities=velocities_ms,
                                                        left_bounds=left_bounds, right_bounds=right_bounds,
                                                        samples_step=self._samples_step, window=self._window)

    def _calc_velocity_bounds(self):
        """ some """
        stacking_times, stacking_velocities = zip(*self._stacking_velocity)
        f = interp1d(stacking_times, stacking_velocities, fill_value="extrapolate")
        interpolated_velocity = np.clip(f(self._times), self._velocities[0], self._velocities[-1])
        left_bound = (interpolated_velocity * (1 - self._deviation)).reshape(-1, 1)
        left_bounds = np.argmin(np.abs(left_bound - self._velocities), axis=1)
        right_bound = (interpolated_velocity * (1 + self._deviation)).reshape(-1, 1)
        right_bounds = np.argmin(np.abs(right_bound - self._velocities), axis=1)
        return left_bounds, right_bounds

    @staticmethod
    @njit(nogil=True, fastmath=True, parallel=True)
    def _calc_numba_res_semblance(base_func, calc_nmo_func, seismogram, times, offsets, velocities, left_bounds,
                                  right_bounds, samples_step, window):
        """some docs"""
        semblance = np.zeros((len(seismogram), len(velocities)))
        for i in prange(left_bounds.min(), right_bounds.max() + 1):
            t_low = np.where(right_bounds == i)[0]
            t_low = 0 if len(t_low) == 0 else t_low[0]
            t_low_window = max(0, t_low - window)

            t_up = np.where(left_bounds == i)[0]
            t_up = len(times) - 1 if len(t_up) == 0 else t_up[-1]
            t_up_window = min(len(times) - 1, t_up + window)

            nmo = np.empty((t_up_window - t_low_window + 1, len(offsets)))
            base_func(seismogram=seismogram, semblance=semblance, ix=i, times=times, offsets=offsets,
                      velocity=velocities[i], samples_step=samples_step, window=window, calc_nmo_func=calc_nmo_func,
                      nmo=nmo, t_min=t_low, t_max=t_up+1, t_window_min=t_low_window, t_window_max=t_up_window+1)

        semblance_len = (right_bounds - left_bounds).max()
        residual_semblance = np.zeros((len(times), semblance_len))
        # Interpolate resulted semblance to get a rectangular image.
        for i in prange(len(semblance)):
            ixs = np.where(semblance[i])[0]
            cropped_smb = semblance[i][ixs[0]: ixs[-1]+1]
            residual_semblance[i] = np.interp(np.linspace(0, len(cropped_smb)-1, semblance_len),
                                            np.arange(len(cropped_smb)), cropped_smb)
        return residual_semblance

    def plot(self, **kwargs):
        """ !! """
        y_points = np.arange(len(self.residual_semblance))
        x_points = np.zeros(len(y_points)) + self.residual_semblance.shape[1]/2

        ticks_range_y = [0, self.residual_semblance.shape[0] * self._samples_step]
        ticks_range_x = [-self._deviation*100, self._deviation*100]

        super().plot(self.residual_semblance, ticks_range_x, ticks_range_y, x_points=x_points, y_points=y_points, xlabel='Velocity deviation (%)', **kwargs)
