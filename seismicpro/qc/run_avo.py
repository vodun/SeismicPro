""" The file is represent script to draw Amplitude vs Offset graph for given data. There are few examples how to
use this script.

1. Draw Amplitude vs Offset graph for given data in window from 1890 to 1910 and class size 30:
>>> python3 run_avo.py -p path_to_data.segy -c 30 -w 1890 1910

2. Draw Amplitude vs Offset graph for given data use horizon in a window with the distance from the horizon value
up by 10 and down by 20, class size 30:
>>> python3 run_avo.py -p path_to_data.segy -c 30 -H path_to_horizon -w 10 20

3. Draw standard deviation of Amplitude vs Offset for given data in a window from 1890 to 1910 and class size 30:
>>> python3 run_avo.py -p path_to_data.segy -c 30 -w 1890 1910 -t std

4. Standard devation of Amplitude vs Offset also works with multiple files:
>>> python3 run_avo.py -p path_to_data_1.segy path_to_data_2.segy ... path_to_data_N.segy -t std ...

For thorothfull description of other aguments of the script use:
>>> python3 run_avo.py -h
"""
# pylint: disable=wrong-import-position
import sys

import argparse

from seismicpro.batchflow import B
from seismicpro import FieldIndex, CustomIndex
from .avo_dataset import AvoDataset
from .utils import avo_plot, std_plot


def str2bool(value):
    """ Allows to use the following values as bools:
    'yes'/'no', 'true'/'false', 't'/'f', 'y'/'n', '1'/'0'.
    """
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    """ Parse argumets from given command."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', nargs="+", help="Path to data.",
                        required=True)
    parser.add_argument('-c', '--class_size', type=str, help="Lenght of one class or lenght of each class if \
                                                              iterable. If `offset`, each class will contain traces \
                                                              from the same offset.",
                        required=True)
    parser.add_argument('-t', '--type', type=str, help="Type of output plot. Should be either `avo` or `std`. If avo \
                                                        is given, the graph will contain the calculated mean square \
                                                        values of amplitudes for each seismogram (in blue) and the\
                                                        average intra-class value of amplitudes (in red). \
                                                        If `std`, for each class, its average value and standard \
                                                        deviation will be calculated. If multiple files are given, \
                                                        `std` type will show all files on one plot, `avo` will create\
                                                        plot for each file.",
                        required=False, default='avo', choices=['avo', 'std'])
    parser.add_argument('-a', '--align', type=str2bool, default=False, help="Aligns the graph to a single average\
                                                                             value. Works only with `std` type.",
                        required=False)
    parser.add_argument('-s', '--save', type=str, help="Path to save resulted plot.",
                        required=False, default=None)

    parser.add_argument('-H', '--horizon_path', type=str, help="Path to horizon.",
                        required=False)
    parser.add_argument('-w', '--window', nargs="+", help="Window size. Waits for two numbers separated by a space \
                                                           without any additional symbols (ex: 1000 1010). \
                                                           If `horizon_path` is given, first number is a distance from \
                                                           horizon to left border of the window, and second number \
                                                           is the distance for the right border.",
                        required=False)
    parser.add_argument('-m', '--method', type=str, help="Should be either `rms` or `abs`.",
                        required=False, default='rms', choices=['abs', 'rms'])
    parser.add_argument('-st', '--stats', type=str, help="Add statistic to plot. There are only two avalible \
                                                          statistics: std - the average value of the standard \
                                                          deviation within each class. corr - the Pearson correlation\
                                                          coefficient between approximation and original average class\
                                                          amplitudes. both - use both of statistics.",
                        required=False, default='None', choices=['std', 'corr', 'both', 'None'])

    #Arguments for plot AVO.
    parser.add_argument('-pa', '--amp_size', type=int, help="Size of ampltudes points on AVO plot.",
                        required=False, default=3)
    parser.add_argument('-pr', '--avg_size', type=int, help="Size of RMS dots on AVO plot.",
                        required=False, default=50)
    parser.add_argument('-pf', '--figsize', nargs="+", help="Size of AVO plot. Should be one or two numbers "\
                                                            " without any additional symbopythls.",
                        required=False, default=(15, 7))
    parser.add_argument('-pt', '--title', type=str, help="Title for AVO plot.",
                        required=False, default=None)
    parser.add_argument('-pra', '--is_approx', type=bool, help="If true, RMS will be shown as an approximation or "\
                                                               "real RMS values.",
                        required=False, default=False)
    parser.add_argument('-pad', '--approx_degree', type=int, help="Degree of function for approximation.",
                        required=False, default=3)
    parser.add_argument('-pd', '--dpi', type=int, help="Output dpi.",
                        required=False, default=100)
    args = parser.parse_args()
    to_int = lambda a: list(map(int, a)) if a is not None else a

    class_size = int(args.class_size) if args.class_size != 'offset' else args.class_size
    figsize = to_int(args.figsize)
    window = to_int(args.window)
    stats = args.stats if args.stats != 'both' else ['std', 'corr']

    if args.horizon_path is not None:
        horizon_window = window
        window = None
    else:
        horizon_window = None

    plot_kwargs = dict(amp_size=args.amp_size,
                       avg_size=args.avg_size,
                       figsize=figsize,
                       title=args.title,
                       is_approx=args.is_approx,
                       approx_degree=args.approx_degree,
                       stats=stats,
                       dpi=args.dpi,
                       save_img=args.save)

    return run_avo(paths=args.path,
                   class_size=class_size,
                   method=args.method,
                   window=window,
                   horizon_window=horizon_window,
                   horizon_path=args.horizon_path,
                   plot_type=args.type,
                   align=args.align,
                   **plot_kwargs)


def run_avo(paths, class_size, method, window, horizon_window, horizon_path, plot_type, align, **kwargs): # pylint: disable=too-many-arguments, too-many-locals
    """ Calculate Amplitude vs Offset graph based on specified parameters."""
    extra_headers = ['offset', 'CDP', 'INLINE_3D', 'CROSSLINE_3D']

    # Define names of dataset variables for save AVO distribution(s).
    names = [path.split('/')[-1].split('.')[0] for path in paths]
    container_names = [name+'_class' for name in names]

    results = []
    for path, name, container in zip(paths, names, container_names):
        index = CustomIndex(FieldIndex(name=name, extra_headers=extra_headers, path=path), index_name='CDP')
        if len(index.indices) == 0:
            raise ValueError('Given file either empty or have incomparable indices.')

        dataset = AvoDataset(index)
        batch_size = 10 if len(index.indices) > 10 else len(index.indices)

        hor_pipeline = (dataset.p
                        .load(fmt='segy', components=name)
                        .sort_traces(src=name, dst=name, sort_by='offset')
                        .find_avo_distribution(B(), src=name, class_size=class_size, window=window,
                                               horizon_path=horizon_path, horizon_window=horizon_window,
                                               method=method, container_name=container)
                        .run_later(batch_size=batch_size, n_epochs=1, shuffle=False,
                                   drop_last=False, bar=True)
                       )
        hor_pipeline.run()
        results.append(getattr(dataset, container))

    save_img_path = kwargs.pop('save_img', None)
    title = kwargs.pop('title', None)

    if plot_type == 'avo':
        for res, name in zip(results, names):
            once_title = 'AVO for {}'.format(name) if title is None else title
            save_img = 'avo_{}.png'.format(name) if save_img_path is None else save_img_path

            avo_plot(data=res,
                     class_size=class_size,
                     save_img=save_img,
                     title=once_title,
                     **kwargs)

    else:
        title = 'AVO for ' + ', '.join(names) if title is None else title
        save_img = 'std_avo.png' if save_img_path is None else save_img_path
        std_plot(avo_results=results,
                 class_size=class_size,
                 names=names,
                 align_mean=align,
                 save_img=save_img,
                 title=title,
                 **kwargs)


if __name__ == "__main__":
    sys.exit(parse_args())
