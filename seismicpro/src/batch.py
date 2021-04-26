import numpy as np

from .gather import Gather
from .semblance import Semblance, ResidualSemblance
from .decorators import create_batch_methods, apply_to_each_component
from .utils import to_list
from ..batchflow import Batch, action, DatasetIndex, NamedExpression


@create_batch_methods(Gather, Semblance, ResidualSemblance)
class SeismicBatch(Batch):
    @property
    def nested_indices(self):
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
        return self.nested_indices

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
    def _load_gather(self, index, src, dst, **kwargs):
        pos = self.index.get_pos(index)
        concat_id, survey_index = index[0], index[1:]
        # Unpack tuple in case of non-multiindex survey
        if len(survey_index) == 1:
            survey_index = survey_index[0]
        # Guarantee, that a DataFrame is always returned, regardless of pandas behaviour.
        survey_index = slice(survey_index, survey_index)
        getattr(self, dst)[pos] = self.index.get_gather(survey_name=src, concat_id=concat_id,
                                                        survey_index=survey_index, **kwargs)

    @apply_to_each_component(target="for", fetch_method_target=False)
    def _load_combined_gather(self, index, src, dst, parent_index, **kwargs):
        pos = self.index.get_pos(index)
        survey_index = parent_index.indices.to_frame().loc[index].index
        getattr(self, dst)[pos] = parent_index.get_gather(survey_name=src, concat_id=index,
                                                          survey_index=survey_index, **kwargs)

    @action
    def update_velocity_cube(self, velocity_cube, src):
        velocity_cube.update(getattr(self, src))
        return self

    @action
    def make_data(self, src, dst, concat_axis=None, stack_axis=None, expand_dims_axis=None):
        if isinstance(src, str):
            data = getattr(self, src)
        else:
            # what about copy here?
            data = src

        if concat_axis is not None:
            data = np.concatenate(data, axis=concat_axis)
        elif stack_axis is not None:
            data = np.stack(data, axis=stack_axis)

        if expand_dims_axis is not None:
            data = np.expand_dims(data, axis=expand_dims_axis)
        setattr(self, dst, data)
        return self

    @action
    def split_results(self, src, dst, shapes):
        data = getattr(self, src)
        shapes = np.cumsum(shapes)[:-1]
        splitted_data = np.split(data, shapes)

        if isinstance(dst, str):
            setattr(self, dst, splitted_data)
        elif isinstance(dst, NamedExpression):
            dst.set(value=splitted_data)
        else:
            ValueError(f'dst must be `str` or `SU named expression`, not {type(dst)}.')
        return self
