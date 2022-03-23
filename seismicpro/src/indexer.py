from itertools import chain

import numpy as np


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
        _ = self.get_loc(index[:1])  # Warmup call to `get_loc`: the first call is way slower than the following ones

    def get_loc(self, index):
        return self.index.get_indexer(index)


class GatherIndexer(BaseIndexer):
    def __init__(self, index):
        super().__init__(index)
        unique_indices_pos = np.where(~index.duplicated())[0]
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
