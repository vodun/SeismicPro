"""Test indexer classes"""

import pytest
import numpy as np
import pandas as pd

from seismicpro.src.utils.indexer import create_indexer, TraceIndexer, GatherIndexer


@pytest.mark.parametrize("index", [
    pd.Index([2, 1, 3]),  # Unique, non-monotonic index
    pd.Index([4, 2, 1, 3, 5, 5]),  # Non-unique, non-monotonic index
    pd.MultiIndex.from_tuples([(1, 2), (3, 4), (0, 0)]),  # Unique, non-monotonic multiindex
    pd.MultiIndex.from_tuples([(1, 2), (3, 4), (0, 0), (1, 2)]),  # Non-unique, non-monotonic index
])
def test_create_indexer_fails(index):
    """Test whether indexer instantiation fails."""
    with pytest.raises(ValueError):
        create_indexer(index)


@pytest.mark.parametrize("index, is_unique, query_index, query_pos", [
    # Trace indexer creation
    [pd.Index([10]), True, [10], [0]],  # Single-element index
    [pd.Index([1, 2]), True, [1], [0]],  # Unique monotonically increasing index
    [pd.Index([1, 2, 3]), True, [1, 3], [0, 2]],  # Unique monotonically increasing index, several requests
    [pd.Index([2, 5, 7]), True, [7, 2, 5], [2, 0, 1]],  # Unique monotonically increasing index, reversed requests
    [pd.Index([2, 1]), True, [1, 2], [1, 0]],  # Unique monotonically decreasing index, reversed requests
    [pd.Index([9, 5, 4]), True, [], []],  # Unique monotonically decreasing index, empty request
    [pd.MultiIndex.from_tuples([(1, 2), (3, 4)]), True, [(1, 2)], [0]], # Unique monotonically increasing multiindex
    [pd.MultiIndex.from_tuples([(10, 10), (6, 3)]), True, [(6, 3)], [1]],  # Unique monotonically decreasing multiindex

    # Gather indexer creation
    [pd.Index([0, 0]), False, [0], [0, 1]],  # Single duplicated element
    [pd.Index([0, 0, 1]), False, [1], [2]],  # Monotonically increasing index
    [pd.Index([5, 2, 1, 1]), False, [1, 2], [2, 3, 1]],  # Monotonically decreasing index
    [pd.MultiIndex.from_tuples([(3, 3), (3, 3)]), False, [(3, 3)], [0, 1]],  # Single duplicated element
    [pd.MultiIndex.from_tuples([(3, 3), (3, 3), (5, 5)]), False, [(5, 5)], [2]],  # Monotonically increasing index
    [pd.MultiIndex.from_tuples([(2, 4), (2, 4), (1, 5)]), False, [(2, 4)], [0, 1]],  # Monotonically decreasing index
])
def test_create_indexer(index, is_unique, query_index, query_pos):
    """Test whether the correct type of indexer is created and correct positions are returned."""
    indexer = create_indexer(index)
    index_type = TraceIndexer if is_unique else GatherIndexer
    assert isinstance(indexer, index_type)
    assert np.array_equal(indexer.get_loc(query_index), query_pos)
