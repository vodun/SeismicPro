import numpy as np

from .gather import Gather
from .semblance import Semblance, ResidualSemblance
from .decorators import create_batch_methods, apply_to_each_component
from .utils import to_list
from ..batchflow import Batch, action, DatasetIndex


@create_batch_methods(Gather, Semblance, ResidualSemblance)
class SeismicBatch(Batch):
    @property
    def wrapped_indices(self):
        if isinstance(self.indices, np.ndarray):
            return self.indices.tolist()
        return [[index] for index in self.indices.values.tolist()]

    def _init_component(self, *args, dst, **kwargs):
        """Create and preallocate a new attribute with the name ``dst`` if it
        does not exist and return batch indices."""
        _ = args, kwargs
        dst = to_list(dst)
        for comp in dst:
            if self.components is None or comp not in self.components:
                self.add_components(comp, init=self.array_of_nones)
        return self.wrapped_indices

    @action
    def load(self, src=None, fmt=None, components=None, combine=False, **kwargs):
        if fmt.lower() in ["sgy", "segy"]:
            if combine:
                if components is None:
                    components = src
                return self._load_combined_gather(src=src, dst=components, **kwargs)
            return self._load_gather(src=src, dst=components, **kwargs)
        return super().load(src=src, fmt=fmt, components=components, **kwargs)

    @apply_to_each_component(target="for", fetch_method_target=False)
    def _load_gather(self, index, src, dst, **kwargs):
        pos = self.index.get_pos(index)
        getattr(self, dst)[pos] = self.index.get_gather(survey_name=src, index=index, **kwargs)

    def _load_combined_gather(self, src, dst, **kwargs):
        list_src, list_dst = to_list(src), to_list(dst)
        index_len = len(np.unique(np.array(self.indices.tolist())[:, 0]))
        new_batch = type(self)(DatasetIndex(np.arange(index_len)))
        if self.components is not None:
            for component in self.components:
                new_batch.add_components(component, getattr(self, component))

        for src, dst in zip(list_src, list_dst):
            gather = self.index.get_combined_gather(survey_name=src, indices=self.indices, **kwargs)
            new_batch.add_components(dst, init=np.array(gather + [None])[:-1])
        return new_batch
