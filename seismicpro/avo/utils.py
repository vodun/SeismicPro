import numpy as np
from numba import njit, prange


@njit(nogil=True, parallel=True)
def aggregate_by_bins_numba(stats, offsets, bin_bounds, limits):
    res = np.empty((len(limits), len(bin_bounds)), dtype=np.float32)
    for num in prange(len(limits)):
        start_ix, end_ix = limits[num]
        bin_ixs = np.searchsorted(bin_bounds, offsets[start_ix: end_ix])
        for ix in prange(len(bin_bounds)):
            bin_stats = stats[start_ix: end_ix][bin_ixs==ix+1]
            res[num, ix] = np.nanmean(bin_stats) if len(bin_stats) > 0 else 0
    return res
