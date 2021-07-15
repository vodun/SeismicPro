#!/usr/bin/env python3
""" Script for validating footprints removal """

import argparse
import os

from seismicpro.qc.footprints import FootprintsSlicesQC

def process():
    """ parse script arguments and pass to a FootprintsSlicesQC controller """
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--paths', nargs="+", help="Path to data. Multiple files should be separated by space. \
                                                         (ex:file1.sgy file2.sgy)", required=True)
    parser.add_argument('-d', '--depths', nargs="+", type=int, help="depths, ms", default=[0, 100000])
    parser.add_argument('-s', '--save_to', type=str, help="Directory to save plots.",
                        default="footprints_slices_qc")

    args = parser.parse_args()

    controller = FootprintsSlicesQC()
    controller.process(*args.paths, dlim=args.depths)
    controller.save_plots(depth_sums=os.path.join(args.save_to, 'depth_sums.png'),
                            proj_sums=os.path.join(args.save_to, 'proj_sums.png'))

if __name__ == "__main__":
    process()
