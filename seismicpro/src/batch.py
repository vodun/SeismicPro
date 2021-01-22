from ..batchflow import Batch, action, inbatch_parallel


def apply_to_each_component(method):
    """Combine list of src items and list dst items into pairs of src and dst items
    and apply the method to each pair.
    Parameters
    ----------
    method : callable
        Method to be decorated.
    Returns
    -------
    decorator : callable
        Decorated method.
    """
    def decorator(self, *args, src, dst=None, **kwargs):
        """Returned decorator."""
        if isinstance(src, str):
            src = (src,)
        if dst is None:
            dst = src
        elif isinstance(dst, str):
            dst = (dst,)

        res = []
        for isrc, idst in zip(src, dst):
            res.append(method(self, *args, src=isrc, dst=idst, **kwargs))
        return self if isinstance(res[0], SeismicBatch) else res
    return decorator


def add_gather_methods(cls):
    # TODO: redirect to a class of an `src` component automatically
    def create_method(method):
        def batch_method(self, index, *args, src=None, dst=None, **kwargs):
            pos = self.index.get_pos(index)
            getattr(self, dst)[pos] = getattr(getattr(self, src)[pos], method)(*args, **kwargs)
        # TODO: fix target
        return action(inbatch_parallel(init="_init_component", target="for")(apply_to_each_component(batch_method)))

    methods_list = ["plot", "sort"]  # TODO: parse abstract class and check that method is absent in `SeismicBatch`
    for method in methods_list:
        setattr(cls, method, create_method(method))
    return cls


@add_gather_methods
class SeismicBatch(Batch):
    @action
    def load(self, src=None, fmt=None, components=None, **kwargs):
        if fmt.lower() in ["sgy", "segy"]:
            if src is None:
                src = components
            components = (components, ) if isinstance(components, str) else components

            self.add_components(components, init=[self.array_of_nones for _ in range(len(components))])
            l = self._load_gather(src=src, dst=components, **kwargs)
            return l
        return super().load(src=src, fmt=fmt, components=components, **kwargs)

    @property
    def wrapped_indices(self):
        return [[index] for index in self.indices.values.tolist()]

    @inbatch_parallel(init="wrapped_indices", target="threads")
    @apply_to_each_component
    def _load_gather(self, index, *args, src, dst, **kwargs):
        pos = self.index.get_pos(index)
        getattr(self, dst)[pos] = self.index.get_gather(survey_name=src, index=index)
        return self

    def _init_component(self, *args, dst, **kwargs):
        """Create and preallocate a new attribute with the name ``dst`` if it
        does not exist and return batch indices."""
        _ = args, kwargs
        dst = (dst,) if isinstance(dst, str) else dst

        for comp in dst:
            if self.components is None or comp not in self.components:
                self.add_components(comp, init=self.array_of_nones)
        return self.wrapped_indices
