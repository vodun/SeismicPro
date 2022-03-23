from itertools import chain

import numpy as np


def unique_indices_sorted(arr):
    """Return indices of the first occurrences of the unique values in a sorted array."""
    mask = np.empty(len(arr), dtype=np.bool_)
    np.any(arr[1:] != arr[:-1], axis=1, out=mask[1:])
    mask[0] = True
    return np.where(mask)[0]


class BaseIndexer:
    def __init__(self, index):
        self.index = index
        self.unique_indices = None

    def get_loc(self, index):
        _ = index
        raise NotImplementedError


class TraceIndexer(BaseIndexer):
    def __init__(self, index):
        super().__init__(index)
        self.unique_indices = index

    def get_loc(self, index):
        return self.index.get_indexer(index)


class GatherIndexer(BaseIndexer):
    def __init__(self, index):
        super().__init__(index)
        unique_indices_pos = unique_indices_sorted(index.to_frame().values)
        ix_start = unique_indices_pos
        ix_end = chain(unique_indices_pos[1:], [len(index)])

        self.unique_indices = index[unique_indices_pos]
        self.index_to_headers_pos = {ix: range(*args) for ix, *args in zip(self.unique_indices, ix_start, ix_end)}

    def get_loc(self, index):
        return list(chain.from_iterable(self.index_to_headers_pos[item] for item in index))


def create_indexer(index):
    if not index.is_monotonic:
        raise ValueError("Indexer can be created only for monotonic indices")
    if index.is_unique:
        return TraceIndexer(index)
    return GatherIndexer(index)
