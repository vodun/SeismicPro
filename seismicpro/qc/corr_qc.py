#!/usr/bin/env python3
""" Calculate correlation metrics and plot maps. """

import os

import warnings
import argparse

from seismicpro.qc.correlation import StackCorrQC
from seismicpro.qc.utils import parse_tuple

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)


def run_corr_qc():
    """ .!! """
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--path_before', type=str, help="Path to SEGY file before.",
                        required=True)
    parser.add_argument('-a', '--path_after', type=str, help="Path to SEGY file after.",
                        required=True)
    parser.add_argument('-he', '--heights', type=parse_tuple, default='0,10000',
                        help="Time window in 'from,to' format (in milliseconds). Default is '0,10000'.")
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
    kernel = args.kernel
    block_size = args.block_size

    qc = StackCorrQC()
    qc.plot(path_before, path_after, heights=heights, kernel=kernel, block_size=block_size)
    save_path = 'corr_qc_{}_{}.jpg'.format(os.path.splitext(os.path.basename(path_before)[0]),
                                            os.path.splitext(os.path.basename(path_after))[0])
    qc.save_plot(save_path)


if __name__ == "__main__":
    run_corr_qc()
