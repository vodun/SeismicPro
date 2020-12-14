""" Class for QC of deconvolution procedures """

import numpy as np
from numba import njit, float32

from .utils import plot_metrics, calc_range
from .base import BaseQC

from seismiqb import SeismicGeometry, GeometryMetrics


class DeconQC(BaseQC):
    """ Metrics and plotters for deconvolution QC. """

    def __init__(self):
        self._path_before = None
        self._path_after = None
        self._gmb = None
        self._gma = None
        self._acf_data = None
        self._fx_data = None
        self._fig = None
        self._sr = None
        self._plot_dict = None

    @property
    def data(self):
        """ .!! """
        if self._acf_data is not None and self._fx_data is not None:
            return np.concatenate((self._acf_data, self._fx_data), axis=-1)
        return None

    def load(self, path_before, path_after):
        """ .!! """
        if path_before != self._path_before:
            self._path_before = path_before
            geom_b = SeismicGeometry(path_before, index=SeismicGeometry.INDEX_POST, collect_stats=False, spatial=False,
                                     headers=SeismicGeometry.HEADERS_PRE_FULL + SeismicGeometry.HEADERS_POST_FULL)
            self._gmb = GeometryMetrics(geom_b)
            self._sr = geom_b.sample_rate
        if path_after != self._path_after:
            self._path_after = path_after
            geom_a = SeismicGeometry(path_after, index=SeismicGeometry.INDEX_POST, collect_stats=False, spatial=False,
                                     headers=SeismicGeometry.HEADERS_PRE_FULL + SeismicGeometry.HEADERS_POST_FULL)
            self._gma = GeometryMetrics(geom_a)

    def plot(self, path_before, path_after, heights, fwindows, **kwargs):
        """ .!! """
        self.load(path_before, path_after)
        self.calc_data(heights, fwindows, **kwargs)
        self._fig = plot_metrics(self.data, **self._plot_dict, xlabel='', ylabel='')

    def calc_data(self, heights, fwindows, **kwargs):
        """ .!! """
        self._acf_data, acf_plot_params = self.run_acf(heights)
        self._fx_data, fx_plot_params = self.run_fx(heights, fwindows, **kwargs)
        names = ['titles', 'cmaps', 'vmins', 'vmaxs']
        vals = [np.concatenate((p1, p2)) for p1, p2 in zip(acf_plot_params, fx_plot_params)]
        self._plot_dict = dict(zip(names, vals))

    def save_plot(self, path):
        """ .!! """
        self._fig.savefig(path, bbox_inches='tight', pad_inches=0)

    def run_acf(self, heights):
        """ Plot a map of ACF deconvolution QC and save it. """

        metric_b = self._gmb.evaluate('tracewise', func=calc_ac_params, l=4, agg=lambda x: x,
                                      num_shifts=100, heights=heights, plot=False)
        metric_a = self._gma.evaluate('tracewise', func=calc_ac_params, l=4, agg=lambda x: x,
                                      num_shifts=100, heights=heights, plot=False)
        diff = metric_a - metric_b

        titles = ['Width', 'Minima', 'Maxima', 'Energy']
        cmaps = ['seismic', 'seismic_r', 'seismic', 'seismic']
        vmins, vmaxs = calc_range(diff)
        plot_dict = (titles, cmaps, vmins, vmaxs)
        return diff, plot_dict

    def run_fx(self, heights, fwindows, kernel=(5, 5), block_size=(1000, 1000)):
        """  Plot a map of FX deconvolution QC and save it. """

        fwindows = tuple(zip(fwindows[::2], fwindows[1::2]))
        freqs = np.fft.rfftfreq(heights[1] - heights[0], d=self._sr / 1000)
        freq_windows = tuple([np.argwhere(np.diff((low <= freqs) & (freqs < high)))[:, 0] if low > 0
                              else np.array((0, np.argwhere(np.diff((low <= freqs) & (freqs < high)))[:, 0][0]))
                              for low, high in fwindows])
        n_plots = len(fwindows)

        def prep_func(arr):
            return np.abs(np.fft.rfft(arr, axis=-1))

        fx_b = self._gmb.evaluate('blockwise', func=calc_fx_corr, l=4, agg=lambda x: x, kernel=kernel,
                                  block_size=block_size, heights=heights, freq_windows=freq_windows,
                                  prep_func=prep_func, plot=False)
        fx_a = self._gma.evaluate('blockwise', func=calc_fx_corr, l=4, agg=lambda x: x, kernel=kernel,
                                  block_size=block_size, heights=heights, freq_windows=freq_windows,
                                  prep_func=prep_func, plot=False)
        diff = fx_a - fx_b

        titles = ['{}-{} Hz'.format(*fw) for fw in fwindows]
        cmaps = ['seismic_r'] * n_plots
        vmins, vmaxs = calc_range(diff)
        plot_dict = (titles, cmaps, vmins, vmaxs)
        return diff, plot_dict


class SliceDeconQC(BaseQC):
    """ .!! """

    def __init__(self):
        self._path_before = None
        self._path_after = None
        self._geometries = None
        self._acf_data = None
        self._fx_data = None
        self._fig = None
        self._sr = None
        self._plot_dict = None

    @property
    def data(self):
        """ .!! """
        if self._acf_data is not None and self._fx_data is not None:
            return np.concatenate((self._acf_data, self._fx_data), axis=-1)
        return None

    def load(self, path_before, path_after):
        """ .!! """
        if path_before != self._path_before:
            self._path_before = path_before
            geom_b = SeismicGeometry(path_before, index=SeismicGeometry.INDEX_POST, collect_stats=False, spatial=False,
                                     headers=SeismicGeometry.HEADERS_PRE_FULL + SeismicGeometry.HEADERS_POST_FULL)
            self._sr = geom_b.sample_rate
        if path_after != self._path_after:
            self._path_after = path_after
            geom_a = SeismicGeometry(path_after, index=SeismicGeometry.INDEX_POST, collect_stats=False, spatial=False,
                                     headers=SeismicGeometry.HEADERS_PRE_FULL + SeismicGeometry.HEADERS_POST_FULL)
        self._geometries = [geom_b, geom_a]

    def plot(self, path_before, path_after, heights, fwindows, **kwargs):
        """ .!! """
        self.load(path_before, path_after)
        self.calc_data(heights, fwindows, **kwargs)
        self._fig = plot_metrics(self.data, **self._plot_dict)

    def calc_data(self, heights, axis, loc, max_freq):
        """ .!! """
        traces = np.stack([g.load_slide(loc=loc, axis=axis)[:, slice(*heights)].T for g in self._geometries], axis=-1)
        traces = np.asfortranarray(traces)

        # FX calculations
        freq = np.fft.rfftfreq(traces.shape[0], d=self._sr / 1000)
        fxs = np.abs(np.fft.rfft(traces, axis=0))[freq < max_freq, ...]
        _, vmaxs = calc_range(fxs)
        self._fx_data = fxs / vmaxs
        # extent = [0, self._fx_data.shape[1], max_freq, 0]

        # ACF calculations
        self._acf_data = np.apply_along_axis(calc_acf, 0, traces)

        # Plotting params
        self._plot_dict = dict(
            titles = ['ACF before', 'ACF after', 'FX before', 'FX after'],
            cmaps = ['seismic']*4,
            vmins = [-1e-10]*2 + [-1]*2,
            vmaxs = [1]*4,
            ylabel = ['Shift, samples']*2 + ['Frequency, Hz']*2,
            xlabel = ['']*4)

######################## Support functions ########################

@njit
def calc_root(arr, i):
    """Calculate root of a linear equasion.

    Find root of a linear equasion that approximates
    autocorrelation (AC) near zero.

    Parameters
    ----------
    arr : ndarray of shape (N,)
        Array with values of the AC.
    i : int
        Index of the AC value that preceeds change
        of the sign of AC.

    Returns
    -------
    root : float
        Point where AC changes sign.
    """
    k = arr[i+1] - arr[i]
    b = arr[i] - k*i
    root = - (b / k)
    return root

@njit
def _calc_ac_params(trace, num_shifts=-1):
    """Calculate parameters of trace autocorrelation function.

    Parameters
    ----------
    trace : ndarray of size (N,)
        Seismic trace.
    num_shifts: int, optional
        Default is -1.
        Minimal number of shifts for ACF. If non-positive,
        shifts until reaches second zero of the ACF derivative.
        If provided, shifts until num_shifts.

    Returns
    -------
    (k0, amps) : tuple
        k0 - float value; coordinate of first sign change of AC.
        amps - ndarray of shape (2,); amplitudes of first
        minima and second maxima of AC.
    """
    k0 = float32(0) # Shift at which AC changes sign first time
    trace_length = len(trace)
    ac = np.zeros(trace_length, dtype=np.float32) # AC values
    der = np.zeros(trace_length-1, dtype=np.float32) # AC derivative
    der_sign_changes = 0 # Number of times AC derivative changes sign
    amps = np.zeros(2, dtype=np.float32) # Container for amplitudes

    ac[0] = np.dot(trace, trace) # Calculate AC for k=0 out of loop
    k = 1 # AC shift
    while k < trace_length:
        ac[k] = np.dot(trace[k:], trace[:-k])

        if k0 == 0 and np.sign(ac[k]) != np.sign(ac[k-1]):
            k0 = calc_root(ac, k-1)

        der[k-1] = ac[k] - ac[k-1]

        if k > 1:
            if np.sign(der[k-1]) != np.sign(der[k-2]):
                der_sign_changes += 1
        k += 1

        if der_sign_changes >= 2 and k >= num_shifts:
            break
    amps[0] = np.min(ac[int(np.ceil(k0)):])
    amps[1] = np.max(ac[int(np.ceil(k0)):])
    amps = amps / ac[0]
    energy = np.sum((ac / ac[0])**2)
    return k0, amps[0], amps[1], energy

def calc_ac_params(trace, heights=None, num_shifts=-1):
    """ Calculate relative difference in AC parameters between
    two traces.

    Parameters
    ----------

    Returns
    -------
    (k0_diff, amps_diff) : tuple
        k0_diff - float; relative difference in AC shift for first
        sign change.
        amps_diff - ndarray of shape (2,); relative chage in amplitudes
        of first minima and second maxima of AC.
    """
    heights = slice(None) if heights is None else slice(*heights)
    trace = trace[heights]

    return _calc_ac_params(trace, num_shifts=num_shifts)

@njit
def calc_acf(trace, shifts=50):
    """Calculate ACF.

    Parameters
    ----------

    Returns
    -------
    """
    ac = np.zeros(shifts, dtype=np.float32) # AC values
    ac[0] = np.dot(trace, trace) # Calculate AC for k=0 out of loop
    k = 1 # AC shift
    while k < shifts:
        ac[k] = np.dot(trace[k:], trace[:-k])
        k += 1
    return ac / ac[0]

@njit
def calc_fx_corr(fx, freq_windows):
    """ .!! """
    res = np.full(len(freq_windows), np.nan)
    mid_fx = fx.shape[0] // 2 + 1
    for i, fwindow in enumerate(freq_windows):
        cc = np.corrcoef(fx[:, fwindow[0]:fwindow[1]])[mid_fx]
        denom = np.sum(~np.isnan(cc))
        if denom > 1:
            res[i] = (np.nansum(cc) - 1) / (denom - 1)
    return res
