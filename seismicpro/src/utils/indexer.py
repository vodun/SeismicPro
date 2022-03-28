"""Implements classes to speed up `DataFrame` indexing.

Multiple `DataFrame`s in `SeismicPro`, such as headers of surveys, have sorted but non-unique index. In this case
`pandas` uses binary search with `O(logN)` complexity while performing lookups. Indexing can be significantly
accelerated by utilizing a `dict` which maps each index value to a range of its positions resulting in `O(1)` lookups.
"""

from itertools import chain

import numpy as np


class BaseIndexer:
    """Base indexer class."""
    def __init__(self, index):
        self.index = index
        self.unique_indices = None

    def get_loc(self, index):
        """Get locations of `index` values in the source index."""
        _ = index
        raise NotImplementedError


class TraceIndexer(BaseIndexer):
    """Construct an indexer for unique monotonic `index`. Should not be instantiated directly, use `create_indexer`
    function instead.
    """
    def __init__(self, index):
        super().__init__(index)
        self.unique_indices = index
        _ = self.get_loc(index[:1])  # Warmup of `get_loc`: the first call is way slower than the following ones

    def get_loc(self, index):
        """Get locations of `index` values in the source index.

        Parameters
        ----------
        index : array-like
            Indices of traces to get locations for.

        Returns
        -------
        locations : np.ndarray
            Locations of the requested indices.
        """
        return self.index.get_indexer(index)


class GatherIndexer(BaseIndexer):
    """Construct an indexer for monotonic, but non-unique `index`. Should not be instantiated directly, use
    `create_indexer` function instead.
    """
    def __init__(self, index):
        super().__init__(index)
        unique_indices_pos = np.where(~index.duplicated())[0]
        ix_start = unique_indices_pos
        ix_end = chain(unique_indices_pos[1:], [len(index)])
        self.unique_indices = index[unique_indices_pos]
        self.index_to_headers_pos = {ix: range(*args) for ix, *args in zip(self.unique_indices, ix_start, ix_end)}

    def get_loc(self, index):
        """Get locations of `index` values in the source index.

        Parameters
        ----------
        index : array-like
            Indices of gathers to get locations for.

        Returns
        -------
        locations : list
            Locations of the requested gathers.
        """
        return list(chain.from_iterable(self.index_to_headers_pos[item] for item in index))


def create_indexer(index):
    """Construct an appropriate indexer for the passed `index`:
    * If `index` is monotonic and unique, default `pandas` indexer is used,
    * If `index` is monotonic but non-unique, an extra mapping from each `index` value to a range of its positions is
      constructed to speed up lookups from `O(logN)` to `O(1)`,
    * If `index` is non-monotonic, an error is raised.

    The returned object implements `get_loc` method that returns locations of given indices in the `index`.

    Parameters
    ----------
    index : pd.Index
        An index to construct an indexer for.

    Returns
    -------
    indexer : BaseIndexer
        The constructed indexer.
    """
    if not index.is_monotonic:
        raise ValueError("Indexer can be created only for monotonic indices")
    if index.is_unique:
        return TraceIndexer(index)
    return GatherIndexer(index)
