#!/usr/bin/env python3
""" Calculate deconvolution metrics and plot maps. """

import os

import warnings
import argparse

from .decon import DeconQC #calc_acf, calc_ac_params, calc_fx_corr
from .utils import parse_tuple#, plot_metrics, calc_range,

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)


def run_decon_qc():
    """ .!! """
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--path_before', type=str, help="Path to SEGY file before deconvolution.",
                        required=True)
    parser.add_argument('-a', '--path_after', type=str, help="Path to SEGY file after deconvolution.",
                        required=True)
    parser.add_argument('-he', '--heights', type=parse_tuple, default='0,10000',
                        help="Time window in 'from,to' format (in milliseconds). Default is '0,10000'.")
    parser.add_argument('-f', '--fwindows', type=parse_tuple, default='0,10,10,20,20,30,30,40',
                        help="Frequency windows for correlation calculation in 'from,to,from,to,...' format." + \
                             "Default is '0,10,10,20,20,30,30,40'.")
    parser.add_argument('-k', '--kernel', type=parse_tuple, default='5,5',
                        help="Lateral window for correlation  calculation, in 'ilines,xlines' format." + \
                             "Default is '5,5'.")
    parser.add_argument('-bs', '--block_size', type=parse_tuple, default='1000,1000',
                        help="Size of a block to read from segy file in 'ilines,xlines' format." + \
                             "Reduce it if OOM error occurs. Default is '1000,1000'.")

    args = parser.parse_args()

    path_before = args.path_before
    path_after = args.path_after
    heights = args.heights
    fwindows = args.fwindows
    kernel = args.kernel
    block_size = args.block_size

    # geometries = [SeismicGeometry(path, headers=SeismicGeometry.HEADERS_PRE_FULL + SeismicGeometry.HEADERS_POST_FULL,
    #                               index=SeismicGeometry.INDEX_POST,
    #                               collect_stats=False, spatial=False) for path in [path_before, path_after]]

    # name = os.path.splitext(os.path.split(path_after)[1])[0] + '_{}ms_{}ms'.format(*heights)
    # heights = (np.array(heights) / geometries[0].sample_rate).astype(int)

    # if interactive:
    #     run_interactive_slice_plots(geometries, heights, name, max_freq)
    # else:
    #     run_acf_check(geometries, heights, name)
    #     run_fx_check(geometries, heights, kernel, block_size, fwindows, name)

    qc = DeconQC()
    qc.plot(path_before, path_after, heights=heights, fwindows=fwindows, kernel=kernel, block_size=block_size)
    save_path = 'decon_qc_{}_{}.jpg'.format(os.path.splitext(os.path.basename(path_before)),
                                            os.path.splitext(os.path.basename(path_after)))
    qc.save_plot(save_path)


# def run_acf_check(geometries, heights, name):
#     """ Plot a map of ACF deconvolution QC and save it. """
#     gmb = GeometryMetrics(geometries[0])
#     gma = GeometryMetrics(geometries[1])

#     metric_b = gmb.evaluate('tracewise', func=calc_ac_params, l=4, agg=lambda x: x,
#                             num_shifts=100, heights=heights, plot=False)
#     metric_a = gma.evaluate('tracewise', func=calc_ac_params, l=4, agg=lambda x: x,
#                             num_shifts=100, heights=heights, plot=False)
#     diff = metric_a - metric_b

#     titles = ('Width', 'Minima', 'Maxima', 'Energy')
#     cmaps = ['seismic']*4
#     cmaps[1] = 'seismic_r'
#     vmins, vmaxs = calc_range(diff)
#     _ = plot_metrics(diff, titles, vmins, vmaxs, cmaps, cols=2, title=name, savefig=name+'_diff_acf.jpg')

#     vmins, vmaxs = calc_range(metric_b)
#     _ = plot_metrics(metric_b, titles, vmins, vmaxs, cmaps, cols=2, title=name, savefig=name+'_before_acf.jpg')
#     vmins, vmaxs = calc_range(metric_a)
#     _ = plot_metrics(metric_a, titles, vmins, vmaxs, cmaps, cols=2, title=name, savefig=name+'_after_acf.jpg')


# def run_fx_check(geometries, heights, kernel, block_size, fwindows, name):
#     """  Plot a map of FX deconvolution QC and save it. """
#     gmb = GeometryMetrics(geometries[0])
#     gma = GeometryMetrics(geometries[1])

#     fwindows = tuple(zip(fwindows[::2], fwindows[1::2]))
#     freqs = np.fft.rfftfreq(heights[1] - heights[0], d=geometries[0].sample_rate / 1000)
#     freq_windows = tuple([np.argwhere(np.diff((low <= freqs) & (freqs < high)))[:, 0] if low > 0
#                           else np.array((0, np.argwhere(np.diff((low <= freqs) & (freqs < high)))[:, 0][0]))
#                           for low, high in fwindows])
#     n_plots = len(fwindows)

#     def prep_func(arr):
#         return np.abs(np.fft.rfft(arr, axis=-1))

#     fx_b = gmb.evaluate('blockwise', func=calc_fx_corr, l=4, agg=lambda x: x, kernel=kernel, block_size=block_size,
#                         heights=heights, freq_windows=freq_windows, prep_func=prep_func, plot=False)
#     fx_a = gma.evaluate('blockwise', func=calc_fx_corr, l=4, agg=lambda x: x, kernel=kernel, block_size=block_size,
#                         heights=heights, freq_windows=freq_windows, prep_func=prep_func, plot=False)

#     diff = fx_a - fx_b

#     titles = ['{}-{} Hz'.format(*fw) for fw in fwindows]
#     cmaps = ['seismic_r'] * n_plots
#     vmins, vmaxs = calc_range(diff)
#     _ = plot_metrics(diff, titles, vmins, vmaxs, cmaps, cols=2, title=name, savefig=name+'_diff_fx.jpg')

#     vmins, vmaxs = calc_range(fx_b)
#     _ = plot_metrics(fx_b, titles, vmins, vmaxs, cmaps, cols=2, title=name, savefig=name+'_before_fx.jpg')
#     vmins, vmaxs = calc_range(fx_a)
#     _ = plot_metrics(fx_a, titles, vmins, vmaxs, cmaps, cols=2, title=name, savefig=name+'_after_fx.jpg')


if __name__ == "__main__":
    run_decon_qc()
