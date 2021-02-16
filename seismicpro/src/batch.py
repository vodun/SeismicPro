from .gather import Gather
from .decorators import add_batch_methods, apply_to_each_component
from .utils import to_list
from ..batchflow import Batch, action, inbatch_parallel


@add_batch_methods(Gather)
class SeismicBatch(Batch):
    @property
    def wrapped_indices(self):
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
    def load(self, src=None, fmt=None, components=None, **kwargs):
        if fmt.lower() in ["sgy", "segy"]:
            return self._load_gather(src=src, dst=components, **kwargs)
        return super().load(src=src, fmt=fmt, components=components, **kwargs)

    @apply_to_each_component(target="threads", check_src_type=False)
    def _load_gather(self, index, src, dst, **kwargs):
        pos = self.index.get_pos(index)
        getattr(self, dst)[pos] = self.index.get_gather(survey_name=src, index=index, **kwargs)
