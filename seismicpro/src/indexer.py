from itertools import chain

import numpy as np


def unique_indices_sorted(arr):
    """Return indices of the first occurrences of the unique values in a sorted array."""
    mask = np.empty(len(arr), dtype=np.bool_)
    np.any(arr[1:] != arr[:-1], axis=1, out=mask[1:])
    mask[0] = True
    return np.where(mask)[0]


class TraceIndexer:
    def __init__(self, index):
        self.index = index
        self.indices = index

    def get_loc(self, index):
        return self.index.get_indexer(index)


class GatherIndexer:
    def __init__(self, index, unique_indices_pos=None):
        if unique_indices_pos is None:
            unique_indices_pos = unique_indices_sorted(index.to_frame().values)
        start_pos = unique_indices_pos
        end_pos = chain(unique_indices_pos[1:], [len(index)])

        self.indices = index[unique_indices_pos]
        self.index_to_headers_pos = {ix: range(start, end) for ix, start, end in zip(self.indices, start_pos, end_pos)}

    def get_loc(self, index):
        return list(chain.from_iterable(self.index_to_headers_pos[item] for item in index))


def create_indexer(headers):
    unique_indices_pos = unique_indices_sorted(headers.index.to_frame().values)
    if len(unique_indices_pos) == len(headers):
        return TraceIndexer(headers.index)
    return GatherIndexer(headers.index, unique_indices_pos)
