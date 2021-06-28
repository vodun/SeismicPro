""" Test for dumping data to sgy files and merging them """
import os

import glob
import pytest
import numpy as np


from seismicpro.batchflow import V, B#, L, I
from seismicpro.src import Survey, SeismicDataset, aggregate_segys


PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../datasets/demo_data/teapot_dome_10.sgy')
# 10 field records with shapes (912, 2049) or (966, 2049)


def create_single_gather(index, header_index, header_cols=None, name=None):
    survey = Survey(PATH, header_index=header_index, header_cols=header_cols, name=name)
    return survey.get_gather(index)


def compare_gathers(expected_gather, suspected_gather):
    expected_headers = expected_gather.headers.reset_index()
    expected_headers.drop(columns="TRACE_SEQUENCE_FILE", inplace=True)
    suspected_headers = suspected_gather.headers.reset_index()
    suspected_headers.drop(columns="TRACE_SEQUENCE_FILE", inplace=True)

    if len(expected_headers) > 0 and len(suspected_headers) > 0:
        assert expected_headers.equals(suspected_headers), "Headers before and after dump don't match"
    assert suspected_gather.shape == expected_gather.shape, "Shapes of the data before and after dump don't match"
    assert np.allclose(suspected_gather.data, expected_gather.data), "Amplitudes before and after dump don't match"
    assert np.allclose(suspected_gather.samples, expected_gather.samples), "Samples before and after dump don't match"
    assert suspected_gather.sample_rate == expected_gather.sample_rate, "Sample rate before and after dump don't match"


@pytest.mark.parametrize('name', ['some_name', None])
@pytest.mark.parametrize('copy_header', [False, True])
@pytest.mark.parametrize('header_index', ['FieldRecord', 'TRACE_SEQUENCE_FILE'])
@pytest.mark.parametrize('header_cols', [None, 'TraceNumber', 'all'])
@pytest.mark.parametrize('dump_index', [175, 182])
def test_dump_single_gather(tmp_path, name, copy_header, header_index, header_cols, dump_index):
    expected_gather = create_single_gather(dump_index, header_index=header_index, header_cols=header_cols, name=name)
    expected_gather.dump(path=tmp_path, name=name, copy_header=copy_header)

    files = glob.glob(os.path.join(tmp_path, '*'))
    assert len(files) == 1, "Dump creates more than one file"

    dumped_survey = Survey(files[0], header_index=header_index, header_cols=header_cols)
    ix = 1 if header_index == 'TRACE_SEQUENCE_FILE' else dump_index
    dumped_gather = dumped_survey.get_gather(index=ix)
    compare_gathers(expected_gather, dumped_gather)

    if copy_header:
        full_exp_headers = Survey(PATH, header_index=header_index, header_cols='all').headers
        full_exp_headers = full_exp_headers.loc[dump_index:dump_index].reset_index()
        full_dump_headers = Survey(files[0], header_index=header_index, header_cols='all').headers
        full_dump_headers = full_dump_headers.reset_index()
        full_exp_headers.drop(columns="TRACE_SEQUENCE_FILE", inplace=True)
        full_dump_headers.drop(columns="TRACE_SEQUENCE_FILE", inplace=True)
        assert full_exp_headers.equals(full_dump_headers), "Copy_header don't save all columns during the dump"


def test_dump_single_gather_with_empty_path():
    gather = create_single_gather(175, 'FieldRecord')
    with pytest.raises(FileNotFoundError):
        gather.dump(path='')


def test_dump_single_gather_with_empty_name(tmp_path):
    gather = create_single_gather(175, 'FieldRecord')
    with pytest.raises(ValueError):
        gather.dump(path=tmp_path, name='')


@pytest.mark.parametrize('mode', ['one_folder', 'split'])
@pytest.mark.parametrize('indices', [[175], [175, 182], 'all'])
def test_aggregate_segys(tmp_path, mode, indices):
    expected_survey = Survey(PATH, header_index='FieldRecord', header_cols='all', name='raw')
    indices = expected_survey.headers.index.drop_duplicates() if indices == 'all' else indices

    if mode == 'split':
        paths = ['folder/'*i for i in range(len(indices))]
    else:
        paths = [''] * len(indices)
    for num, (ix, path) in enumerate(zip(indices, paths)):
        g = expected_survey.get_gather(ix)
        g.dump(os.path.join(tmp_path, path), name=f'{num}_{ix}', copy_header=True)

    aggregate_segys(os.path.join(tmp_path, './**/*.sgy'), os.path.join(tmp_path, 'aggr.sgy'), recursive=True)

    dumped_survey = Survey(os.path.join(tmp_path, 'aggr.sgy'), header_index='FieldRecord', header_cols='all')
    assert np.allclose(expected_survey.samples, dumped_survey.samples),"Samples don't match"
    assert np.allclose(expected_survey.sample_rate, dumped_survey.sample_rate), "Sample rate doesn't match"
    assert np.allclose(expected_survey.samples_length, dumped_survey.samples_length), "length of samples doesn't match"

    #TODO: optimize
    expected_survey_headers = expected_survey.headers.loc[indices].sort_values(['FieldRecord', 'TraceNumber']).reset_index(drop=True)
    dumped_survey_headers = dumped_survey.headers.sort_values(['FieldRecord', 'TraceNumber']).reset_index(drop=True)
    expected_survey_headers.drop(columns="TRACE_SEQUENCE_FILE", inplace=True)
    dumped_survey_headers.drop(columns="TRACE_SEQUENCE_FILE", inplace=True)
    assert len(expected_survey_headers) == len(dumped_survey_headers), "Length of surveys' headers don't match"
    assert expected_survey_headers.equals(dumped_survey_headers), "The headers don't match"

    for ix in indices:
        expected_gather = expected_survey.get_gather(ix)
        expected_gather.sort(by='TraceNumber')
        dumped_gather = dumped_survey.get_gather(ix)
        dumped_gather.sort(by='TraceNumber')
        compare_gathers(expected_gather, dumped_gather)
