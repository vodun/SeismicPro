"""
Script for searching artefacts by calculating
difference between raw and denoised max_abs amplitude and divided by median(abs(raw))
"""

import sys
import argparse

import numpy as np

from seismicpro.batchflow import V, B, action, inbatch_parallel
from seismicpro.src import FieldIndex, SeismicDataset, SeismicBatch
from seismicpro.src.utils import make_index
from seismicpro.src.seismic_metrics import MetricsMap


class MaxAbsBatch(SeismicBatch):
    """ Batch class with `get_max_abs` action"""

    @action
    @inbatch_parallel(init='_init_component')
    def get_max_abs(self, index, dst):
        """
        Compute difference between raw and lift max_abs divided by median(abs(raw))

        Parameters
        ----------
        index:
            current item's index, populated by `_init_component`
        dst: str
            name of the component to put result to
        """

        pos = self.get_pos(None, 'raw', index)
        raw = getattr(self, 'raw')[pos]
        lift = getattr(self, 'lift')[pos]

        getattr(self, dst)[pos] = (np.max(np.abs(raw)) - np.max(np.abs(lift))) / \
                    (np.median(np.abs(raw)) + np.finfo(np.float32).eps)
        return self


def run_max_abs_all():
    """ parse script arguments and run `max_abs_all`"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--path_raw', type=str, help="Path to raw SEGY file.",
                        required=True)
    parser.add_argument('-l', '--path_lift', type=str, help="Path to denoised SEGY file.",
                        required=True)
    parser.add_argument('-o', '--path_out', type=str, help="Path to file with results", default='res.csv')

    args = parser.parse_args()
    path_raw = args.path_raw
    path_lift = args.path_lift
    save_to = args.path_out

    max_abs_all(path_raw, path_lift, save_to)


def max_abs_all(raw_path, lift_path, out_path):
    """
    Compute difference between between raw and lift max_abs divided by median(abs(raw)) for each FieldID
    and dump results to file in .csv format

    Parameters
    ----------
    raw_path: str
        path to raw SGY file
    lift_path: str
        path to denoised SGY file
    out_path: str
        path to results

    """
    index = make_index({'raw': raw_path, 'lift': lift_path}, index_type=FieldIndex, extra_headers=['offset'])
    test_set = SeismicDataset(index, batch_class=MaxAbsBatch)

    test_pipeline = (test_set.p
                     .init_variable('res_all', default=list())
                     .init_variable('metrics')
                     .load(components='raw', fmt='segy')
                     .load(components='lift', fmt='segy')
                     .get_max_abs(dst='res_all')
                     .gather_metrics(MetricsMap, metrics=B('res_all'),
                                     coords=B('index').get_df()[['SourceX', 'SourceY']].values,
                                     save_to=V('metrics', mode='a'))
                     .update(V('res_all', mode='a'), B('res_all'))
                    )

    test_pipeline = test_pipeline.run(batch_size=min(10, len(index)),
                                      n_epochs=1, drop_last=False, shuffle=False)

    res_all = np.hstack(test_pipeline.get_variable('res_all'))

    idx = np.argsort(res_all, kind='stable')

    with open(out_path, 'w') as outf:
        outf.write("FieldID,Val\n")
        for idx, val in zip(index.indices[idx], res_all[idx]):
            outf.write("{},{}\n".format(idx, val))

    metrics = test_pipeline.v('metrics')
    metrics.evaluate('map', bin_size=10, figsize=(10, 7), save_dir='artifacts.jpg', pad=True, max_value=20)



if __name__ == "__main__":
    run_max_abs_all()
