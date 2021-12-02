"""Implements decorators to add new methods to a SeismicBatch class"""

import inspect
from functools import partial, wraps

from .utils import to_list
from ..batchflow import action, inbatch_parallel
from ..batchflow.decorators import _make_action_wrapper_with_args


def batch_method(*args, target="for", args_to_unpack=None, force=False, copy_src=True, **kwargs):
    """Mark a method as being added to `SeismicBatch` class.

    The new method is added by :func:`~decorators.create_batch_methods` decorator of `SeismicBatch` if the parent class
    is listed in its arguments and parallelly redirects calls to elements of the batch. A method will be created only
    if there is no method with the same name in the batch class or if `force` flag was set to `True`.

    Two new arguments are added for each of the created batch methods:
    src : str or list of str
        Names of components whose elements will be processed by the method.
    dst : str or list of str, optional
        Names of components to store the results. Must match the length of `src`. If not given, the processing is
        performed inplace. If a component with a name specified in `dst` does not exist, it will be created using
        :func:`~batch.SeismicBatch._init_component` method.

    Parameters
    ----------
    target : {"for", "threads"}, optional, defaults to "for"
        `inbatch_parallel` target to use when processing batch elements with the method.
    args_to_unpack : str or list of str, optional
        If given, listed arguments are allowed to accept `str` value which will be treated as a name of a batch
        component. In this case, when the call is redirected to a particular element, each argument will be substituted
        by the corresponding value of the specified component.
    force : bool, optional, defaults to False
        Whether to redefine an existing batch method with the method decorated.
    copy_src : bool, optional, defaults to True
        Whether to copy batch elements before processing if `src` component differs from `dst`. Usually, this flag
        is set to `True` to keep `src` data intact since most processing methods are done inplace. Sometimes it should
        be set to `False` to avoid redundant copying e.g. when a new object is returned like in
        :func:`~Gather.calculate_semblance`.
    kwargs : misc, optional
        Additional keyword arguments to `batchflow.decorators.action`.

    Returns
    -------
    decorator : callable
        A decorator, that keeps the method unchanged, but saves all the passed arguments to its `batch_method_params`
        attribute.

    Raises
    ------
    ValueError
        If positional arguments were passed except for the method being decorated.
    """
    batch_method_params = {"target": target, "args_to_unpack": args_to_unpack, "force": force, "copy_src": copy_src}

    def decorator(method):
        """Decorate a method by setting passed batch method params to its attributes."""
        method.batch_method_params = batch_method_params
        method.action_params = kwargs
        return method

    if len(args) == 1 and callable(args[0]):
        return decorator(args[0])
    if len(args) > 0:
        raise ValueError("batch_method decorator does not accept positional arguments")
    return decorator


def _apply_to_each_component(method, target, fetch_method_target):
    """Decorate a method so that it is parallelly applied to elements of each component in `src` and stores the result
    in the corresponding components of `dst`."""
    @wraps(method)
    def decorated_method(self, *args, src, dst=None, **kwargs):
        src_list = to_list(src)
        dst_list = to_list(dst) if dst is not None else src_list

        for src, dst in zip(src_list, dst_list):  # pylint: disable=redefined-argument-from-local
            # Set src_method_target default
            src_method_target = target

            # Dynamically fetch target from method attribute
            if fetch_method_target:
                src_types = {type(elem) for elem in getattr(self, src)}
                if len(src_types) != 1:
                    err_msg = "All elements in {src} component must have the same type, but {src_types} found"
                    raise ValueError(err_msg.format(src=src, src_types=", ".join(map(str, src_types))))
                src_method_target = getattr(src_types.pop(), method.__name__).batch_method_params["target"]

            # Fetch target from passed kwargs
            src_method_target = kwargs.pop("target", src_method_target)

            # Set method target to for if the batch contains only one element
            if len(self) == 1:
                src_method_target = "for"

            parallel_method = inbatch_parallel(init="_init_component", target=src_method_target)(method)
            parallel_method(self, *args, src=src, dst=dst, **kwargs)
        return self
    return decorated_method


def apply_to_each_component(*args, target="for", fetch_method_target=True):
    """Decorate a method so that it is parallelly applied to elements of each component in `src` and stores the result
    in the corresponding components of `dst`.

    If a component with a name specified in `dst` does not exist, it will be created using
    :func:`~batch.SeismicBatch._init_component` method.

    Parameters
    ----------
    target : {"for", "threads"}, optional, defaults to "for"
        Default `inbatch_parallel` target to use when processing component elements with the method if it was not
        decorated by `batch_method` or `target` was not passed in `kwargs` during method call.
    fetch_method_target : bool, optional, defaults to True
        Whether to try to fetch `target` from method attributes if it was decorated by `batch_method`.

    Returns
    -------
    decorator : callable
        A decorator, that parallelly applies a method to elements of specified components.
    """
    partial_apply = partial(_apply_to_each_component, target=target, fetch_method_target=fetch_method_target)
    if len(args) == 1 and callable(args[0]):
        return partial_apply(args[0])
    return partial_apply


def _get_class_methods(cls):
    """Return all methods of the class."""
    return {func for func in dir(cls) if callable(getattr(cls, func))}


def create_batch_methods(*component_classes):
    """Create new batch methods from those decorated by `batch_method` in classes listed in `component_classes`.

    A new batch method is created for a method decorated by `batch_method` in classes defined by `component_classes`
    only if there is no method with the same name in the decorated class or if `force` flag was set to `True` in the
    `batch_method` arguments. Created methods parallelly redirect calls to elements of the batch and each of them has
    two new arguments added:
    src : str or list of str
        Names of components whose elements will be processed by the method.
    dst : str or list of str, optional
        Names of components to store the results. Must match the length of `src`. If not given, the processing is
        performed inplace. If a component with a name specified in `dst` does not exist, it will be created using
        :func:`~batch.SeismicBatch._init_component` method.

    Parameters
    ----------
    component_classes : tuple of type
        Classes to search for methods to create in.

    Returns
    -------
    decorator : callable
        A decorator, that adds new methods to the batch class.
    """
    def decorator(cls):
        decorated_methods = set()
        force_methods = set()
        action_params = {}
        for component_class in component_classes:
            for method_name in _get_class_methods(component_class):
                method = getattr(component_class, method_name)
                if hasattr(method, "batch_method_params"):
                    decorated_methods.add(method_name)
                    action_params[method_name] = getattr(method, 'action_params')
                    if getattr(method, "batch_method_params")["force"]:
                        force_methods.add(method_name)
        methods_to_add = (decorated_methods - _get_class_methods(cls)) | force_methods

        # TODO: dynamically generate docstring
        def create_method(method_name):
            def method(self, index, *args, src=None, dst=None, **kwargs):
                # Get an object corresponding to the given index from src component and copy it if needed
                pos = self.index.get_pos(index)
                obj = getattr(self, src)[pos]
                if getattr(obj, method_name).batch_method_params["copy_src"] and src != dst:
                    obj = obj.copy()

                # Unpack required method arguments by getting the value of specified component with index pos
                # and perform the call with updated args and kwargs
                obj_method = getattr(obj, method_name)
                obj_arguments = inspect.signature(obj_method).bind(*args, **kwargs)
                obj_arguments.apply_defaults()
                args_to_unpack = obj_method.batch_method_params["args_to_unpack"]
                if args_to_unpack is not None:
                    for arg_name in to_list(args_to_unpack):
                        arg_val = obj_arguments.arguments[arg_name]
                        if isinstance(arg_val, str):
                            obj_arguments.arguments[arg_name] = getattr(self, arg_val)[pos]
                getattr(self, dst)[pos] = obj_method(*obj_arguments.args, **obj_arguments.kwargs)
            method.__name__ = method_name
            if action_params[method_name]:
                return action(**action_params[method_name])(apply_to_each_component(method))
            return action(apply_to_each_component(method))

        for method_name in methods_to_add:
            setattr(cls, method_name, create_method(method_name))
        return cls
    return decorator
