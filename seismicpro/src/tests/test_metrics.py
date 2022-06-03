""" Constant subsequences metrics for QC """

import pytest
import numpy as np

from seismicpro.src.metrics.qc_metric import MaxClipsLenMetric, ConstLenMetric


class DummyGather:
    """Dummy object with `data` attribute to emulate gathers in metrics """
    def __init__(self, data):
        self.data = np.asarray(data)


CLIPS_PARAMS = [
    ([1, 2, 2, 3, 3, 4], [1, 0, 0, 0, 0, 1]),
    ([1, 2, 2, 3, 3, 3], [1, 0, 0, 3, 3, 3]),
    ([1, -2, 3, 3, 3, 1], [0, 1, 3, 3, 3, 0]),
    (np.asarray([[1, 2, 2, 0]]), np.asarray([[0, 2, 2, 1]])),
    (np.asarray([[1, 2, 2, 0], [1, 1, 2, 3]]), np.asarray([[0, 2, 2, 1], [2, 2, 0, 1]])),
]
@pytest.mark.parametrize("arr,expected", CLIPS_PARAMS)
def test_clips(arr, expected):
    """Test MaxClipsLenMetric"""
    assert np.allclose(MaxClipsLenMetric.get_res(DummyGather(arr)).astype(int), np.asarray(expected))


CLIPLEN_IND_PARAMS = [
    ([1, 2, 2, 3, 3, 4], [0, 2, 2, 2, 2, 0]),
    ([1, 2, 2, 3, 3, 3], [0, 2, 2, 3, 3, 3]),
    ([1, -2, 3, 3, 3, 1], [0, 0, 3, 3, 3, 0]),
    (np.asarray([[1, 2, 2, 0]]), np.asarray([[0, 2, 2, 0]])),
    (np.asarray([[1, 2, 2, 0], [1, 1, 2, 3]]), np.asarray([[0, 2, 2, 0], [2, 2, 0, 0]])),
]
@pytest.mark.parametrize("arr,expected", CLIPLEN_IND_PARAMS)
def test_const_subseq(arr, expected):
    """Test ConstLenMetric"""
    assert np.allclose(ConstLenMetric.get_res(DummyGather(arr)).astype(int), np.asarray(expected))
