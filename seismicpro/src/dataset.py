from .batch import SeismicBatch
from .index import SeismicIndex
from ..batchflow import Dataset


class SeismicDataset(Dataset):
    def __init__(self, index=None, batch_class=SeismicBatch, preloaded=None, copy=True, **kwargs):
        if index is None:
            index = SeismicIndex(**kwargs)
        super().__init__(index, batch_class=batch_class, preloaded=preloaded, copy=copy, **kwargs)

    def create_subset(self, index):
        if isinstance(index, SeismicIndex) and isinstance(self.index, SeismicIndex):
            if not index.indices.isin(self.indices).all():
                raise IndexError("Some indices from given index are not present in current SeismicIndex")
            return type(self).from_dataset(self, self.index.create_subset(index))
        return super().create_subset(index)
