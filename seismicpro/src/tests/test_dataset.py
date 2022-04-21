"""Implementation of tests for seismic index"""

# pylint: disable=redefined-outer-name
import pytest

from seismicpro import Survey, SeismicIndex, SeismicDataset

@pytest.fixture
def survey(segy_path):
    """survey"""
    return Survey(segy_path, header_index='FieldRecord')

def test_dataset_creation_from_survey(survey):
    """test_index_creation_from_survey"""
    _ = SeismicDataset(survey)

def test_dataset_creation_from_index(survey):
    """test_dataset_creation_from_index"""
    index = SeismicIndex(survey)
    _ = SeismicDataset(index)

def test_dataset_split(survey):
    """test_dataset_split"""
    dataset = SeismicDataset(survey)
    dataset.split()

def test_dataset_collect_stats(survey):
    """test_dataset_collect_stats"""
    dataset = SeismicDataset(survey)
    dataset.collect_stats()
