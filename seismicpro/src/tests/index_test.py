"""Implementation of tests for seismic index"""

import pytest

from seismicpro import Survey, SeismicIndex


def test_index_creation_from_survey(segy_path):
    """test_index_creation_from_survey"""
    survey = Survey(segy_path, header_index='FieldRecord')
    _ = SeismicIndex(surveys=survey)

def test_index_creation_from_index(segy_path):
    """test_index_creation_from_index"""
    survey = Survey(segy_path, header_index='FieldRecord')
    index = SeismicIndex(surveys=survey)
    _ = SeismicIndex(index=index)

@pytest.mark.parametrize('name1,name2', [pytest.param('name', 'name', marks=pytest.mark.xfail), ['name1', 'name2']])
@pytest.mark.parametrize('header_index1,header_index2', [pytest.param('INLINE_3D', 'offset', marks=pytest.mark.xfail),
                                                        ['FieldRecord', 'FieldRecord']])
def test_index_with_two_surveys_merge(segy_path, header_index1, header_index2, name1, name2):
    """test_index_with_two_surveys_merge"""
    survey_one = Survey(segy_path, header_index=header_index1, name=name1)
    survey_two = Survey(segy_path, header_index=header_index2, name=name2)
    _ = SeismicIndex(surveys=[survey_one, survey_two], mode='m')

@pytest.mark.parametrize('name1,name2', [pytest.param('name1', 'name2', marks=pytest.mark.xfail), ['name', 'name']])
@pytest.mark.parametrize('header_index1,header_index2', [pytest.param('INLINE_3D', 'offset', marks=pytest.mark.xfail),
                                                        ['FieldRecord', 'FieldRecord']])
def test_index_with_two_surveys_concat(segy_path, header_index1, header_index2, name1, name2):
    """test_index_with_two_surveys_concat"""
    survey_one = Survey(segy_path, header_index=header_index1, name=name1)
    survey_two = Survey(segy_path, header_index=header_index2, name=name2)
    _ = SeismicIndex(surveys=[survey_one, survey_two], mode='c')

def test_index_with_multiple_merge_concats(segy_path):
    """test_index_with_multiple_merge_concats"""
    s1_before = Survey(segy_path, header_index='FieldRecord', name="before")
    s2_before = Survey(segy_path, header_index='FieldRecord', name="before")

    s1_after = Survey(segy_path, header_index='FieldRecord', name="after")
    s2_after = Survey(segy_path, header_index='FieldRecord', name="after")

    index_before = SeismicIndex(surveys=[s1_before, s2_before], mode="c")
    index_after = SeismicIndex(surveys=[s1_after, s2_after], mode="c")
    _ = SeismicIndex(surveys=[index_before, index_after], mode="m")

def test_index_split(segy_path):
    """test_index_split"""
    survey = Survey(segy_path, header_index='FieldRecord')
    index = SeismicIndex(surveys=survey)
    index.split()
