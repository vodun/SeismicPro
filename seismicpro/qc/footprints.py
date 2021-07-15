import os
import numpy as np
import scipy as sp

from matplotlib import pyplot as plt
from matplotlib import colors

from seismiqb import SeismicGeometry

class BaseQC():
    """Base class for metrics calculation and plotting. """
    def __init__(self):
        self._figs = {}
        self._paths = None

    @property
    def figures(self):
        """ List available figures """
        return list(self._figs.keys())

    def plot(self, **kwargs):
        """ Plot calculated metrics. """
        raise NotImplementedError

    def load(self, *args):
        """ Load data from files.
        Loading is skipped if the new file list matches what is already loaded """

        paths = tuple(map(os.path.abspath, args))

        if self._paths == paths:
            return

        self._paths = paths
        self._figs = {}

        self._load(*paths)

    def _load(self, *paths):
        """ controller-dependent loading """
        raise NotImplementedError

    def process(self, *args, **kwargs):
        """ Do everything """
        print('start load')
        self.load(*args)
        print('start plot')
        self.plot(**kwargs)

    def save_plots(self, **kwargs):
        """ save figures in controllers to provided paths """
        for name in kwargs:
            os.makedirs(os.path.dirname(kwargs[name]), exist_ok=True)
            self._figs[name].savefig(kwargs[name], bbox_inches='tight', pad_inches=0)


class FootprintsSlicesQC(BaseQC):
    """ inspect depth slices sums and their projections to iline ad xline axes """

    IMSIZE=5

    def __init__(self):
        super().__init__()
        self.geoms = None

    def _load(self, *paths):
        print(paths)
        self.geoms = [SeismicGeometry(path, collect_stats=True, spatial=False) for path in paths]
        print(self.geoms)

    def plot(self, ilim=None, xlim=None, dlim=None, num_workers=25, figsize=None):
        """ visualize projections sums and their FFTs """

        dlim = _get_dlim_samples(dlim, self.geoms[0])
        print('got dlim')
        num = len(self.geoms)
        if figsize is None:
            figsize=(len(self.geoms)*self.IMSIZE, self.IMSIZE*2)

        fig0, ax0 = plt.subplots(num, 2, figsize=figsize)
        if num == 1:
            ax0 = ax0[:, np.newaxis]

        fig1, ax1 = plt.subplots(4, 1, figsize=figsize)

        descriptions = []
        for i, geom in enumerate(self.geoms):
            print(descriptions)
            descriptions.append(_plot_geom(geom, ilim, xlim, dlim, ax0.T[i], ax1, num_workers))
        print('plotted geom')
        unique_desc = set(tuple(zip(*descriptions))[1])

        if len(unique_desc) == 1:
            suptitle = unique_desc.pop()
        else:
            suptitle = '\n'.join(f"{name}: {desc}" for name, desc in descriptions)
        fig0.suptitle(suptitle)

        for ax in ax1:
            ax.legend()

        ax1[0].set_title('iline sums')
        ax1[1].set_title('ilines sums fft')
        ax1[1].set_yscale('log')
        ax1[1].axvline(1/300, color="black", linestyle=":", alpha=0.7)

        ax1[2].set_title('xline sums')
        ax1[3].set_title('xlines sums fft')
        ax1[3].set_yscale('log')
        ax1[3].axvline(1/300, color="black", linestyle=":", alpha=0.7)

        self._figs['depth_sums'] = fig0
        self._figs['proj_sums'] = fig1


def _get_dlim_samples(dlim, geom, single_slice=False):
    if isinstance(dlim, int):
        dlim = int(dlim//geom.sample_rate)
    elif len(dlim) == 1:
        dlim = int(dlim[0]//geom.sample_rate)
    elif len(dlim) == 2 and not single_slice:
        dlim = (int(dlim[0]//geom.sample_rate), int(dlim[1]//geom.sample_rate + 1))
    else:
        fmt_opts = ("", "") if single_slice else ("s", " or a pair of integers")
        msg = "depth{} should be a single integer{}".format(*fmt_opts)
        raise ValueError(msg)
    return dlim


def _plot_projection(geomsum, zt, name, axs, axis_proj):
    sum0 = np.sum(geomsum, axis=axis_proj)/np.sum(zt, axis=axis_proj)
    axs[2*axis_proj].plot(sum0, label=name)
    kk = np.abs(np.fft.rfft(sum0))
    wns = np.fft.rfftfreq(sum0.shape[0], d=25)
    axs[2*axis_proj+1].plot(wns, kk, label=name)


def _plot_geom(geom, ilim, xlim, dlim, ax0, ax1, num_workers=25):
    ilen, xlen, dlen = geom.cube_shape
    name = geom.name

    ilim_c = make_slice(ilim, ilen)
    xlim_c = make_slice(xlim, xlen)
    dlim_c = make_slice(dlim, dlen)

    zt = 1 - geom.zero_traces[ilim_c, xlim_c]

    cube = geom[ilim_c, xlim_c, dlim_c]

    geomsum = np.sum(np.abs(cube), axis=2) if dlim_c.stop - dlim_c.start > 1 else cube.squeeze()

    ax0[0].imshow(geomsum, cmap='gray')
    ax0[0].set_title(f'{name} sums spatial domain')

    with sp.fft.set_workers(num_workers):
        fft_slices = np.abs(sp.fft.fft2(cube, axes=(0, 1))).sum(axis=2)

    fft_slices = sp.fft.fftshift(fft_slices)

    ax0[1].imshow(fft_slices, norm=colors.LogNorm())
    ax0[1].set_title(f'{name} sums KK domain')

    _plot_projection(geomsum, zt, name, ax1, axis_proj=0)
    _plot_projection(geomsum, zt, name, ax1, axis_proj=1)

    return name, _slices_desc(ilim_c, xlim_c, dlim_c)


def _slices_desc(ilim_c, xlim_c, dlim_c):
    if isinstance(dlim_c, int):
        dlim_c = dlim_c * 2
    else:
        dlim_c = (dlim_c.start * 2, (dlim_c.stop - 1) * 2) if dlim_c.stop - dlim_c.start > 1 else dlim_c.start * 2
    return f"depth: {dlim_c} ms, ilines: {(ilim_c.start, ilim_c.stop- 1)}, xlines: {(xlim_c.start, xlim_c.stop-1)}"


def artifact_map(arr, mean=None, std=None, kernel_size=4):
    """ ML artifact detection quality map """

    if mean is None:
        mean = np.mean(arr, axis=(0,1))
    if std is None:
        std = np.std(arr, axis=(0,1))
    arr = (arr - mean) / std

    kernel = np.tile([[-1, 1], [1, -1]], (kernel_size//2, kernel_size//2))

    if arr.ndim == 3:
        kernel = np.expand_dims(kernel, -1)

    res = np.abs(sp.signal.fftconvolve(arr, kernel, mode='same', axes=(0,1)))

    res[:kernel_size - 1, :] = 0
    res[1 - kernel_size:, :] = 0
    res[:, :kernel_size - 1] = 0
    res[:, 1 - kernel_size:] = 0

    return res

def make_slice(lims, ref):
    """ make slice object using provided limits

    Parameters
    ----------
    lims : None, int, tuple, or slice
        arguments for slice creation,
        if int, slice(lims, lims+1) is created
    ref : int
        maximum value for silce's `stop` parameter

    Returns
    -------
    slice
        new slice object
    """
    if not isinstance(lims, slice):
        if isinstance(lims, tuple):
            lims = slice(*lims)
        elif isinstance(lims, int):
            lims = slice(lims, lims+1)
        else:
            lims = slice(lims)

    if lims.stop is None or lims.stop > ref:
        lims = slice(lims.start, ref, lims.step)

    if lims.start is None:
        lims = slice(0, lims.stop, lims.step)

    return lims