"""Implementation of tests for survey"""

# pylint: disable=redefined-outer-name
import pytest
import numpy as np

from seismicpro import Survey
# TODO:
# 1. Add first break file for test


@pytest.fixture(scope='module')
def survey(segy_path):
    """Create gather"""
    survey = Survey(segy_path, header_index=['INLINE_3D', 'CROSSLINE_3D'], header_cols=['offset', 'FieldRecord'],
                    collect_stats=True)
    return survey

@pytest.mark.parametrize('header_cols', [None, 'offset', 'all'])
@pytest.mark.parametrize('name', [None, 'raw'])
@pytest.mark.parametrize('limits', [None, (0, 1000)])
@pytest.mark.parametrize('collect_stats', [True, False])
def test_survey_creation(segy_path, header_cols, name, limits, collect_stats):
    """test_survey_creation"""
    _ = Survey(segy_path, header_index='FieldRecord', header_cols=header_cols, name=name, limits=limits,
                collect_stats=collect_stats)

def test_survey_get_quantile(survey):
    """test_survey_get_quantile"""
    survey.get_quantile(0.01)

def test_survey_get_gather(survey):
    """test_survey_get_gather"""
    _ = survey.get_gather(index=survey.headers.index[0])

def test_survey_copy(survey):
    """test_survey_copy"""
    _ = survey.copy()

def test_survey_filter(survey):
    """test_survey_filter"""
    survey.filter(lambda offset: offset < 1500, cols="offset", inplace=False)

def test_survey_apply(survey):
    """test_survey_apply"""
    survey.apply(lambda offset: np.abs(offset), cols="offset", inplace=False)

def test_survey_reindex(survey):
    """test_survey_reindex"""
    _ = survey.reindex('TRACE_SEQUENCE_FILE', inplace=False)

@pytest.mark.parametrize('limits', [1000, (1, 1000), (1, 1000, 2), slice(0, 1000, 2), None])
def test_survey_set_limits(survey, limits):
    """test_survey_set_limits"""
    _ = survey.set_limits(limits)

def test_survey_generate_supergathers(survey):
    """test_survey_generate_supergathers"""
    _ = survey.generate_supergathers(inplace=False)
