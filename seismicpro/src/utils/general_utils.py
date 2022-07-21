"""Miscellaneous general utility functions"""

from functools import partial
from concurrent.futures import Future, Executor

import numpy as np


def to_list(obj):
    """Cast an object to a list. Almost identical to `list(obj)` for 1-D objects, except for `str`, which won't be
    split into separate letters but transformed into a list of a single element."""
    if isinstance(obj, (list, tuple, set, np.ndarray)):
        return list(obj)
    return [obj]


def maybe_copy(obj, inplace=False, **kwargs):
    """Copy an object if `inplace` flag is set to `False`. Otherwise return the object unchanged."""
    return obj if inplace else obj.copy(**kwargs)


def int_linspace(start, stop, num):
    if (num < 1) or (num > stop - start + 1):
        raise ValueError("num must be between 1 and stop - start + 1")
    if num == 1:
        return np.array([start], dtype=np.int32)
    div, mod = divmod(stop - start, num - 1)
    steps = np.zeros(num, dtype=np.int32)
    steps[1 : mod + 1] = div + 1
    steps[mod + 1 :] = div
    return start + np.cumsum(steps)


def unique_indices_sorted(arr):
    """Return indices of the first occurrences of the unique values in a sorted array."""
    mask = np.empty(len(arr), dtype=np.bool_)
    np.any(arr[1:] != arr[:-1], axis=1, out=mask[1:])
    mask[0] = True
    return np.where(mask)[0]


def align_args(reference_arg, *args):
    """Convert `reference_arg` and each arg from `args` to lists so that their lengths match the number of elements in
    the `reference_arg`. If some arg contains a single element, its value will is repeated. If some arg is an
    array-like whose length does not match the number of elements in the `reference_arg` an error is raised."""
    reference_arg = to_list(reference_arg)
    processed_args = []
    for arg in args:
        arg = to_list(arg)
        if len(arg) == 1:
            arg *= len(reference_arg)
        if len(arg) != len(reference_arg):
            raise ValueError("Lengths of all passed arguments must match")
        processed_args.append(arg)
    return reference_arg, *processed_args


def get_first_defined(*args):
    """Return the first non-`None` argument. Return `None` if no `args` are passed or all of them are `None`s."""
    return next((arg for arg in args if arg is not None), None)


def validate_cols_exist(df, cols):
    """Check if each column from `cols` is present either in the `df` DataFrame columns or index."""
    df_cols = set(df.columns) | set(df.index.names)
    missing_cols = set(to_list(cols)) - df_cols
    if missing_cols:
        raise KeyError(f"The following headers must be preloaded: {', '.join(missing_cols)}")


def get_cols(df, cols):
    """Extract columns from `cols` from the `df` DataFrame columns or index as a 2d `np.ndarray`."""
    validate_cols_exist(df, cols)
    # Avoid using direct pandas indexing to speed up selection of multiple columns from small DataFrames
    res = []
    for col in to_list(cols):
        col_values = df[col] if col in df.columns else df.index.get_level_values(col)
        res.append(col_values.values)
    return np.column_stack(res)


class MissingModule:
    """Postpone raising missing module error for `module_name` until it is being actually accessed in code."""
    def __init__(self, module_name):
        self._module_name = module_name

    def __getattr__(self, name):
        _ = name
        raise ImportError(f"No module named {self._module_name}")

    def __call__(self, *args, **kwargs):
        _ = args, kwargs
        raise ImportError(f"No module named {self._module_name}")


class ForPoolExecutor(Executor):
    """A sequential executor of tasks in a for loop. Inherits `Executor` interface thus can serve as a drop-in
    replacement for both `ThreadPoolExecutor` and `ProcessPoolExecutor` when threads or processes spawning is
    undesirable."""

    def __init__(self, *args, **kwargs):
        _ = args, kwargs
        self.task_queue = []

    def submit(self, fn, /, *args, **kwargs):
        """Schedule `fn` to be executed with given arguments."""
        future = Future()
        self.task_queue.append((future, partial(fn, *args, **kwargs)))
        return future

    def shutdown(self, *args, **kwargs):
        """Signal the executor to finish all scheduled tasks and free its resources."""
        _ = args, kwargs
        for future, fn in self.task_queue:
            future.set_result(fn())
        self.task_queue = None
