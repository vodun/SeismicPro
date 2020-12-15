""" file contains metrics based on semblance for ground-roll attenuation. """
import os
import sys
import argparse

import numpy as np

PATH = os.path.realpath(__file__)
PATH = PATH.split('/')[:-2]
PATH = '/'.join(PATH)

sys.path.insert(0, os.path.join(PATH))

from seismicpro.batchflow import B, V, inbatch_parallel, action, Pipeline
from seismicpro import FieldIndex, CustomIndex, SeismicBatch, SeismicDataset, MetricsMap


class SimpleMetrics:
    @staticmethod
    @inbatch_parallel(init="_init_component", target="threads")
    def calculate_minmax(batch, index, src, raw_semb, dst):
        """some docs"""
        pos = batch.index.get_pos(index)
        diff_semblance = getattr(batch, src)[pos]
        raw_semblance = getattr(batch, raw_semb)[pos]
        minmax_raw = np.max(np.max(raw_semblance, axis=1) - np.min(raw_semblance, axis=1))
        minmax_diff = np.max(np.max(diff_semblance, axis=1) - np.min(diff_semblance, axis=1))
        getattr(batch, dst)[pos] = minmax_diff / (minmax_raw + 1e-11)
        return batch

def parse_args():
    """ Parse argumets. """
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', nargs="+", help="Path to data.", required=True)
    parser.add_argument('-b', '--bin_size', nargs="+", help=" ",
                        required=True)
    parser.add_argument('-vr', '--velocity_range', nargs='+', help="Path to data. Waits for two numbers separated by a space \
                                                        without any additional symbols (ex: 1000 1010).", required=True)
    parser.add_argument('-vn', '--velocity_number', type=int, help=" ", required=True)
    parser.add_argument('-a', '--agg_func', type=str, help=" ", required=False,
                        default='mean', choices= ['std', 'max', 'min', 'mean', 'median'])
    parser.add_argument('-w', '--window', type=int, help="Window size. ",
                    required=False, default=25)
    # plotter args
    parser.add_argument('-pf', '--figsize', nargs="+", help="Size of metrics map. Should be two numbers "\
                                                            " without any additional symbopythls.",
                        required=False, default=(20, 7))
    parser.add_argument('-pt', '--title', type=str, help="Title for metrics map.",
                        required=False, default=None)
    parser.add_argument('-pd', '--pad', type=bool, help="Whether to add padding or not",
                        required=False, default=None)
    parser.add_argument('-s', '--save', type=str, help="Path to save resulted plot. By default it will be saved to \
                                                        the current directory with name 'metrics_map'.",
                        required=False, default='metrics_map.png')
    parser.add_argument('-d', '--dpi', type=int, help=" ",
                    required=False, default=300)

    args = parser.parse_args()
    to_int = lambda a: list(map(int, a)) if a is not None else a
    # import pdb;pdb.set_trace()
    bin_size = to_int(args.bin_size)
    velocity_range = to_int(args.velocity_range)
    figsize = to_int(args.figsize)

    if len(bin_size) == 1:
        bin_size = bin_size[0]
    velocities = np.linspace(*velocity_range, args.velocity_number)

    plot_kwargs = dict(figsize=figsize,
                       title=args.title,
                       pad=args.pad,
                       dpi=args.dpi,
                       save_to=args.save)

    return construct_metrics(paths=args.path,
                   bin_size=bin_size,
                   agg_func=args.agg_func,
                   velocities=velocities,
                   window=args.window,
                   **plot_kwargs)


def construct_metrics(paths, bin_size, agg_func, velocities, window, **plot_kwargs):
    path_in, path_out = paths
    extra_headers=['offset', 'SourceX',  'SourceY']
    components = ('raw', 'out')
    index = (FieldIndex(name='raw', extra_headers=extra_headers, path=path_in)
            .merge(FieldIndex(name='out', extra_headers=extra_headers, path=path_out)))
    dataset = SeismicDataset(index)

    pipeline = (dataset.p
                .add_namespace(SimpleMetrics)
                .init_variable('metrics')

                .load(fmt='segy', components=components)
                .sort_traces(src=components, dst=components, sort_by='offset')

                .calculate_vertical_velocity_semblance(src='raw', dst='raw_semblance',
                                                       velocities=velocities,  window=window)

                .call(lambda batch: batch.raw - batch.out, save_to=B('out'))
                .calculate_vertical_velocity_semblance(src='out', dst='diff_out_semblance',
                                                       velocities=velocities,  window=window)

                .calculate_minmax(B(), src='diff_out_semblance', raw_semb='raw_semblance', dst='metrics')

                .gather_metrics(MetricsMap,
                                coords=B('index').get_df()[["SourceX", "SourceY"]].drop_duplicates().values,
                                metrics=B('metrics'),
                                save_to=V('metrics', mode='a'))

                .run_later(1, shuffle=True, n_epochs=1, drop_last=False, bar=True)
    )
    pipeline.run()

    metrics = pipeline.v('metrics')
    _ = metrics.evaluate('construct_map', metrics_name='metrics',
                         bin_size=bin_size, agg_func=agg_func, agg=None,
                         **plot_kwargs)

if __name__ == "__main__":
    sys.exit(parse_args())
