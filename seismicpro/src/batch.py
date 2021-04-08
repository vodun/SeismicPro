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
    def load(self, src=None, fmt="sgy", components=None, combined=False, **kwargs):
        if isinstance(fmt, str) and fmt.lower() in {"sgy", "segy"}:
            if not combined:
                return self._load_gather(src=src, dst=components, **kwargs)
            unique_files = self.indices.unique(level=0)
            combined_batch = type(self)(DatasetIndex(unique_files), dataset=self.dataset, pipeline=self.pipeline)
            return combined_batch._load_combined_gather(src=src, dst=components, parent_index=self.index, **kwargs)
        return super().load(src=src, fmt=fmt, components=components, **kwargs)

    @apply_to_each_component(target="for", fetch_method_target=False)
    def _load_combined_gather(self, index, src, dst, parent_index, **kwargs):
        pos = self.index.get_pos(index)
        survey_index = parent_index.indices.to_frame().loc[index].index
        getattr(self, dst)[pos] = parent_index.get_gather(survey_name=src, concat_id=index,
                                                          survey_index=survey_index, **kwargs)

    @apply_to_each_component(target="for", fetch_method_target=False)
    def _load_gather(self, index, src, dst, **kwargs):
        pos = self.index.get_pos(index)
        concat_id, *survey_index = index
        survey_index = survey_index[0] if len(survey_index) == 1 else tuple(survey_index)
        getattr(self, dst)[pos] = self.index.get_gather(survey_name=src, concat_id=concat_id,
                                                        survey_index=survey_index, **kwargs)

    @action
    def update_cube(self, cube, src):
        for model in getattr(self, src):
            cube.update(model)
        return self
