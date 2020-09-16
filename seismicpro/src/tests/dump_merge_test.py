""" Test for dumping data to sgy files and merging them """

import os

import pytest
import numpy as np


from seismicpro.batchflow import V, B, L, I
from seismicpro.src import SeismicDataset, FieldIndex, TraceIndex, merge_segy_files

PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../datasets/demo_data/teapot_dome_10.sgy')
# 10 field records with shapes (912, 2049) or (966, 2049)


def compare_files(path_1, path_2, compare_all):
    """ Checks that the content of a given SGY-files is a subset of (or is equal to)
    the content of the other file

    Parameters
    ----------
    path_1 : str
        path to the larger file
    path_2 : str
        path to the smaller file
    compare_all : bool
        whether to chek exact equality
    """

    index1 = TraceIndex(name='f1', path=path_1)
    index2 = TraceIndex(name='f2', path=path_2)
    index = index1.merge(index2)

    # all traces from the smaller file are in the larger one
    assert len(index2) == len(index)

    if compare_all:
        # both files have same traces
        assert len(index1) == len(index2)

    index = FieldIndex(index)

    ppl = (
        SeismicDataset(index).p
        .load(components='f1', fmt='segy', tslice=slice(2000))
        .load(components='f2', fmt='segy', tslice=slice(2000))
        .sort_traces(src=('f1', 'f2'), sort_by='TraceNumber')
        .add_components('res')
        .init_variable('res', default=[])
        .apply_parallel(lambda arrs: np.allclose(*arrs), src=('f1', 'f2'), dst='res')
        .update(V('res', 'a'), B('res'))
    )

    ppl.run(batch_size=1, n_epochs=1, drop_last=False, shuffle=False, bar=False)

    res = np.stack(ppl.get_variable('res'))

    assert np.all(res)


@pytest.mark.parametrize('index_type', [FieldIndex, TraceIndex])
def test_dump_split_merge(index_type, tmp_path):
    """
    Dump content item-wise for one iteration. Merge dumped files.
    Check that all traces from the merged file are in the input file
    """

    index = index_type(path=PATH, name='raw')

    ppl = (
        SeismicDataset(index).p
        .load(components='raw', fmt='segy', tslice=slice(2000))
        .dump(src='raw', path=tmp_path, fmt='sgy', split=True)
        )

    ppl.next_batch(4)

    merged_path = os.path.join(tmp_path, "out.sgy")
    merge_segy_files(path=os.path.join(tmp_path, "*.sgy"),
                     output_path=merged_path, bar=False)

    compare_files(PATH, merged_path, compare_all=False)


@pytest.mark.parametrize('index_type, batch_size', [(FieldIndex, 4), (TraceIndex, 1000)])
def test_dump_nosplit_merge(index_type, batch_size, tmp_path):
    """
    Dump content batch-wise for the whole dataset. Merge dumped files.
    Check that merged and input files contents equal
    """

    index = index_type(path=PATH, name='raw')

    ppl = (
        SeismicDataset(index).p
        .load(components='raw', fmt='segy', tslice=slice(2000))
        .dump(src='raw', path=L(lambda x: os.path.join(tmp_path, str(x) + '.sgy'))(I()),
              fmt='sgy', split=False)
        )

    ppl.run(batch_size=batch_size, n_epochs=1, drop_last=False, shuffle=False, bar=False)

    merged_path = os.path.join(tmp_path, "out.sgy")
    merge_segy_files(path=os.path.join(tmp_path, "*.sgy"),
                     output_path=merged_path, bar=False)

    compare_files(PATH, merged_path, compare_all=True)
