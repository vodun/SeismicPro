""" Class for QC based on stacked data correlations """

import numpy as np
from numba import njit

from .utils import calc_range, plot_metrics, plot_slice
from .base import BaseQC

# pylint: disable=wrong-import-order
# sys.path.append('..')
from seismiqb import SeismicGeometry, GeometryMetrics


class StackCorrQC(BaseQC):
    """ Metrics based on stack data lateral correlations. """

    def __init__(self):
        """ .!! """
        self._path_before = None
        self._path_after = None
        self._data = None
        self._gm = None
        self._fig = None
        self._plot_dict = None

    @property
    def data(self):
        """ .!! """
        return self._data

    def load(self, path_before, path_after):
        """ .!! """
        if path_before != self._path_before:
            self._path_before = path_before
            geom_b = SeismicGeometry(path_before, index=SeismicGeometry.INDEX_POST, collect_stats=False, spatial=False,
                                     headers=SeismicGeometry.HEADERS_PRE_FULL + SeismicGeometry.HEADERS_POST_FULL)
        if path_after != self._path_after:
            self._path_after = path_after
            geom_a = SeismicGeometry(path_after, index=SeismicGeometry.INDEX_POST, collect_stats=False, spatial=False,
                                     headers=SeismicGeometry.HEADERS_PRE_FULL + SeismicGeometry.HEADERS_POST_FULL)
            self._gm = GeometryMetrics((geom_b, geom_a))

    def calc_data(self, heights, kernel=(5, 5), block_size=(1000, 1000)):
        """ .!! """
        self._data = self._gm.evaluate('blockwise', func=calc_corr, l=2, kernel=kernel,
                                       block_size=block_size, heights=heights, plot=False)

        vmins, vmaxs = calc_range(self._data)
        self._plot_dict = dict(
            titles=['Difference of correlations', 'Correlation of difference'],
            vmins=vmins,
            vmaxs=vmaxs,
            cmaps=['RdYlGn', 'RdYlGn'],
        )

    def plot(self, path_before, path_after, heights, **kwargs):
        """ .!! """
        self.load(path_before, path_after)
        self.calc_data(heights, **kwargs)
        self._fig = plot_metrics(self.data, **self._plot_dict)
        
    def save_plot(self, path):
        """ .!! """
        self._fig.savefig(path, bbox_inches='tight', pad_inches=0)


######################## Support functions ########################

@njit
def avg_corr(traces, middle_trace):
    """ .!! """
    cc = np.corrcoef(traces)[middle_trace]
    denom = np.sum(~np.isnan(cc))
    if denom > 1:
        return (np.nansum(cc) - 1) / (denom - 1)
    return np.nan

@njit
def calc_corr(tb, ta):
    """ .!! """
    res = np.full(2, np.nan)
    mid = tb.shape[0] // 2 + 1

    res[0] = avg_corr(ta, mid) - avg_corr(tb, mid)

    diff = ta - tb
    res[1] = avg_corr(diff, mid)
    return res
