from functools import partial, wraps
from collections import defaultdict

from .utils import to_list
from ..batchflow import action, inbatch_parallel


def batch_method(*args, target="for"):
    if len(args) == 1 and callable(args[0]):
        method = args[0]
        method.inbatch_parallel_target = target
        return method

    if len(args) > 0:
        raise ValueError("batch_method decorator does not accept positional arguments")

    def decorator(method):
        method.inbatch_parallel_target = target
        return method
    return decorator


def _apply_to_each_component(method, target="for", target_dict=None, check_src_type=True, method_name=None):
    if target_dict is None:
        target_dict = {}
    @wraps(method)
    def decorated_method(self, *args, src, dst=None, **kwargs):
        src_list = to_list(src)
        dst_list = to_list(dst) if dst is not None else src_list
        for src, dst in zip(src_list, dst_list):
            src_target = target
            if check_src_type:
                src_type_set = {type(elem) for elem in getattr(self, src)}
                if len(src_type_set) != 1:
                    err_msg = "Component elements must have the same type, but {} found"
                    raise ValueError(err_msg.format(", ".join([str(t) for t in src_type_set])))
                class_type = src_type_set.pop()
                src_target = target_dict.get(class_type, target)
                method_call = getattr(class_type, method_name)
                if hasattr(method_call, "inbatch_parallel_target"):
                    # We need this to benchmark our model.
                    if src_target != method_call.inbatch_parallel_target:
                        src_target = method_call.inbatch_parallel_target
            parallel_method = inbatch_parallel(init="_init_component", target=src_target)(method)
            parallel_method(self, *args, src=src, dst=dst, **kwargs)
        return self
    return decorated_method


def apply_to_each_component(*args, target="for", target_dict=None, check_src_type=True, method_name=None):
    if len(args) == 1 and callable(args[0]):
        return _apply_to_each_component(args[0])
    return partial(_apply_to_each_component, target=target, target_dict=target_dict, check_src_type=check_src_type, method_name=method_name)


def get_methods(cls):
    return {func for func in dir(cls) if callable(getattr(cls, func))}

def add_batch_methods(*component_classes):
    def decorator(cls):
        methods_dict = defaultdict(dict)
        batch_methods = get_methods(cls)
        for component_class in component_classes:
            for method_name in get_methods(component_class) - batch_methods:
                method = getattr(component_class, method_name)
                if hasattr(method, "inbatch_parallel_target"):
                    methods_dict[method_name][component_class] = getattr(method, "inbatch_parallel_target")

        def create_method(method_name, target_dict):
            def method(self, index, *args, src=None, dst=None, **kwargs):
                pos = self.index.get_pos(index)
                obj = getattr(self, src)[pos]
                if src != dst:
                    obj = obj.copy()
                getattr(self, dst)[pos] = getattr(obj, method_name)(*args, **kwargs)
            return action(apply_to_each_component(target_dict=target_dict, method_name=method_name)(method))

        for method_name, target_dict in methods_dict.items():
            setattr(cls, method_name, create_method(method_name, target_dict))
        return cls
    return decorator
