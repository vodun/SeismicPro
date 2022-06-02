""" Constant subsequences for QC """

import pytest
import numpy as np

from seismicpro.src.metrics.utils import get_const_indicator, get_constlen_indicator


from seismicpro.src.metrics.qc_metric import MaxClipsLenMetric


CLIPLEN_IND_PARAMS = [
    ([1, 2, 2, 3, 3, 4], [1, 0, 0, 0, 0, 1]),
    ([1, -2, 3, 3, 3, 1], [0, 1, 3, 3, 3, 0]),
    (np.asarray([[1, 2, 2, 0]]), np.asarray([[0, 2, 2, 1],])),
    (np.asarray([[1, 2, 2, 0], [1, 1, 2, 3]]), np.asarray([[0, 2, 2, 1], [2, 2, 0, 1]])),
]
@pytest.mark.parametrize("arr,expected", CLIPLEN_IND_PARAMS)
def test_MaxClipsLenMetric(arr, expected):
    """Test constant subsequence length indicator"""
    assert np.allclose(MaxClipsLenMetric.get_res(np.asarray(arr)).astype(int), np.asarray(expected))



CLIP_IND_PARAMS = [
    ([1, 2, 2, 3, 3, 4], None, [0, 0, 1, 0, 1, 0]),
    ([1, 2, 2, 3, 3, 4], 3, [0, 0, 0, 1, 1, 0]),
    ([1, 2, 2, 3, 3, 4], 4, [0, 0, 0, 0, 0, 1]),
    (np.asarray([[1, 2, 2, 3]]), None, np.asarray([[0, 0, 1, 0]])),
    (np.asarray([[1, 2, 2, 3], [1, 1, 2, 3]]), None, np.asarray([[0, 0, 1, 0], [0, 1, 0, 0]])),
]
@pytest.mark.parametrize("arr,cmpval,expected", CLIP_IND_PARAMS)
def test_get_const_indicator(arr, cmpval, expected):
    """Test constant subsequence indicator"""
    assert np.allclose(get_const_indicator(np.asarray(arr), cmpval).astype(int), np.asarray(expected))

CLIPLEN_IND_PARAMS = [
    ([1, 2, 2, 3, 3, 4], None, [0, 0, 2, 0, 2, 0]),
    ([1, 2, 2, 3, 3, 4], 3, [0, 0, 0, 2, 2, 0]),
    ([1, 2, 3, 3, 3, 4], None, [0, 0, 0, 3, 3, 0]),
    (np.asarray([[1, 2, 2, 3]]), None, np.asarray([[0, 0, 2, 0],])),
    (np.asarray([[1, 2, 2, 3], [1, 1, 2, 3]]), None, np.asarray([[0, 0, 2, 0], [0, 2, 0, 0]])),
]
@pytest.mark.parametrize("arr,cmpval,expected", CLIPLEN_IND_PARAMS)
def test_get_constlen_indicator(arr, cmpval, expected):
    """Test constant subsequence length indicator"""
    assert np.allclose(get_constlen_indicator(np.asarray(arr), cmpval).astype(int), np.asarray(expected))
