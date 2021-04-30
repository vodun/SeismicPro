""" Test for dumping data to sgy files and merging them """

import os

import pytest
import numpy as np


from seismicpro.batchflow import V, B, L, I
from seismicpro.src import Survey, SeismicDataset, aggregate_segys

PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../datasets/demo_data/teapot_dome_10.sgy')
# 10 field records with shapes (912, 2049) or (966, 2049)

def check_dumped_files(full_survey, index, tmp_path):
    for path in os.listdir(tmp_path):
        dumped_survey = Survey(os.path.join(tmp_path, path), header_index=index, header_cols='all')
        dumped_index = dumped_survey.headers.index.unique()
        assert len(dumped_index) == 1
        dumped_index = dumped_index[0]

        # Check headers
        expected_header = full_survey.headers.loc[dumped_index]
        expected_header = expected_header.drop(columns="TRACE_SEQUENCE_FILE")

        dumped_header = dumped_survey.headers.drop(columns="TRACE_SEQUENCE_FILE")
        assert expected_header.equals(dumped_header)

        # Check data
        expected_data = full_survey.get_gather(index=dumped_index).data
        dumped_data = dumped_survey.get_gather(index=dumped_index).data
        assert np.allclose(expected_data, dumped_data)


@pytest.mark.parametrize('index', ['FieldRecord'])#, ['FieldRecord', 'TraceNumber']])
@pytest.mark.parametrize('name', ['some_file', None])
def test_gather_dump(index, name, tmp_path):
    # TODO: Check that dump will copy all headers, even when header_cols is not 'all'.
    survey = Survey(PATH, header_index=index, header_cols='all')
    # dump every gather
    for ix in survey.headers.index.unique():
        gather = survey.get_gather(index=ix)
        #TODO: add test for copy_header
        gather.dump(path=tmp_path, name=name, copy_header=False)
    check_dumped_files(full_survey=survey, index=index, tmp_path=tmp_path)

@pytest.mark.parametrize('index', ['FieldRecord'])#, ['FieldRecord', 'TraceNumber']])
def test_batch_dump(index, tmp_path):
    survey = Survey(PATH, header_index=index, header_cols='all', name='raw')
    dataset = SeismicDataset(surveys=survey)
    pipeline = (dataset.p
        .load(src='raw')
        .dump(src='raw', path=tmp_path, copy_header=False)
    )
    pipeline.run(1)
    check_dumped_files(full_survey=survey, index=index, tmp_path=tmp_path)

@pytest.mark.parametrize('index', ['FieldRecord'])#, ['FieldRecord', 'TraceNumber']])
def test_aggregate_segys(index, tmp_path):
    survey = Survey(PATH, header_index=index, header_cols='all', name='raw')
    dataset = SeismicDataset(surveys=survey)
    pipeline = (dataset.p
        .load(src='raw')
        .dump(src='raw', path=tmp_path, copy_header=False)
    )
    pipeline.run(1)
    merged_path = os.path.join(tmp_path, "out.sgy")
    aggregate_segys(in_paths=os.path.join(tmp_path, "*.sgy"), out_path=merged_path, bar=False)

    dumped_survey = Survey(path=merged_path, header_index=index, header_cols='all', name='raw')
    dumped_headers = dumped_survey.headers.drop(columns="TRACE_SEQUENCE_FILE")
    dumped_headers = dumped_headers.sort_values(by='offset')
    survey_headers = survey.headers.drop(columns="TRACE_SEQUENCE_FILE")
    survey_headers = survey_headers.sort_values(by='offset')
    assert dumped_headers.equals(survey_headers)

    for ix in survey.headers.index.unique():
        gather = survey.get_gather(index=ix)
        dumped_gather = dumped_survey.get_gather(index=ix)
        assert np.allclose(gather.data, dumped_gather.data)
