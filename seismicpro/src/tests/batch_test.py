"""Implementation of tests for seismic index"""

# pylint: disable=redefined-outer-name
import pytest

from seismicpro import Survey, SeismicDataset


@pytest.fixture(scope='module')
def dataset(segy_path):
    """dataset"""
    survey = Survey(segy_path, header_index='FieldRecord', name='raw')
    return SeismicDataset(surveys=survey)

def test_batch_load(dataset):
    """test_batch_load"""
    batch = dataset.next_batch(1)
    batch.load(src='raw')

def test_batch_load_combined(segy_path):
    """test_batch_load_combined"""
    survey = Survey(segy_path, header_index='TRACE_SEQUENCE_FILE', name='raw')
    dataset = SeismicDataset(surveys=survey)
    batch = dataset.next_batch(1000)
    batch = batch.load(src='raw', combined=True)
    assert len(batch.raw) == 1
    assert len(batch.raw[0].data) == 1000

def test_batch_make_model_inputs(dataset):
    """test_batch_make_model_inputs"""
    #TODO: change it to pipeline!
    dataset = dataset.copy()
    batch = dataset.next_batch(1, shuffle=False)
    batch.load(src='raw').make_model_inputs(src=[batch.raw[0].data], dst='inputs', mode='c', axis=0,
                                            expand_dims_axis=1)
    assert batch.inputs.shape == (126, 1, 1500)

def test_batch_make_model_outputs(dataset):
    """test_batch_make_model_outputs"""
    #TODO: change it to pipeline!
    dataset = dataset.copy()
    batch = dataset.next_batch(2, shuffle=False)
    batch.load(src='raw').make_model_inputs(src=[batch.raw[0].data, batch.raw[1].data], dst='inputs', mode='c', axis=0,
                                            expand_dims_axis=1)
    batch.split_model_outputs(src='inputs', dst='outputs', shapes=[126, 198])
    assert batch.outputs[0].shape == (126, 1, 1500)
    assert batch.outputs[1].shape == (198, 1, 1500)
