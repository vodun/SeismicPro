""" Perform f-k filtering  """

import os
import argparse

from .fk_utils import get_device, process_kk_notch


def parse_args():
    """ parse script arguments and run `process`"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to input SEGY file.")
    parser.add_argument('-k', '--kk_params', type=int, nargs=4, metavar="V",
                        help="KK-filtering parameters. " +
                        "4 integers, representing F_offset, F_delta, K_offset, K_delta. " +
                        "")
    parser.add_argument('-n', '--notch_params', type=int, nargs=4, metavar="P",
                        help="K-notch-filtering parameters. " +
                        "4 integers, representing X_offset, X_radius, Y_offset, Y_radius." +
                        "")
    parser.add_argument('-o', '--output', type=str, help="Path to output. " +
                        "If omitted, is created by adding '_filtered' postfix to input file name")
    parser.add_argument('--cpu_only', action="store_false", help=" use GPU if available (default:Do not use GPU)")

    args = parser.parse_args()

    inp_path = args.input
    save_to = args.output
    kk_params = args.kk_params
    notch_params = args.notch_params

    if kk_params is None and notch_params is None:
        print("At least one of kk_params or notch_params should be provided!")

    if save_to is None:
        postfix = '_filtered'
        if kk_params:
            postfix += '_kk_' + '_'.join([str(i) for i in kk_params])
        if notch_params:
            postfix += '_notch_' + '_'.join([str(i) for i in notch_params])
        save_to = postfix.join((os.path.splitext(inp_path)[0], '.sgy'))
        print("Output path not provided by user. The result will be written to {}".format(save_to))
    device = get_device(args.cpu_only)

    process_kk_notch(inp_path, save_to, kk_params, notch_params, device)


if __name__ == "__main__":
    parse_args()
