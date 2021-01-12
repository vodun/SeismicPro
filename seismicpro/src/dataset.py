from ..batchflow import Dataset
from .batch import SeismicBatch
from .index import SeismicIndex


class SeismicDataset(Dataset):
    def __init__(self, index=None, batch_class=SeismicBatch, preloaded=None, copy=True, **kwargs):
        if index is None:
            index = SeismicIndex(**kwargs)
        super().__init__(index, batch_class=batch_class, preloaded=preloaded, copy=copy, **kwargs)
