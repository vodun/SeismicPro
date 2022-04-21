"""Miscellaneous general utility functions"""

from functools import partial
from concurrent.futures import Future, Executor

import numpy as np
from numba import njit

from .interpolation import interpolate


def to_list(obj):
    """Cast an object to a list. Almost identical to `list(obj)` for 1-D objects, except for `str`, which won't be
    split into separate letters but transformed into a list of a single element."""
    if isinstance(obj, (list, tuple, set)):
        return list(obj)
    return [obj]


def maybe_copy(obj, inplace=False, **kwargs):
    """Copy an object if `inplace` flag is set to `False`. Otherwise return the object unchanged."""
    return obj if inplace else obj.copy(**kwargs)


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


INDEX_TO_COORDS = {
    # Shot index
    "FieldRecord": ("SourceX", "SourceY"),
    ("SourceX", "SourceY"): ("SourceX", "SourceY"),

    # Receiver index
    ("GroupX", "GroupY"): ("GroupX", "GroupY"),

    # Trace index
    "TRACE_SEQUENCE_FILE": ("CDP_X", "CDP_Y"),
    ("FieldRecord", "TraceNumber"): ("CDP_X", "CDP_Y"),
    ("SourceX", "SourceY", "GroupX", "GroupY"): ("CDP_X", "CDP_Y"),

    # Bin index
    "CDP": ("CDP_X", "CDP_Y"),
    ("CDP_X", "CDP_Y"): ("CDP_X", "CDP_Y"),
    ("INLINE_3D", "CROSSLINE_3D"): ("INLINE_3D", "CROSSLINE_3D"),

    # Supergather index
    ("SUPERGATHER_SourceX", "SUPERGATHER_SourceY"): ("SUPERGATHER_SourceX", "SUPERGATHER_SourceY"),
    ("SUPERGATHER_GroupX", "SUPERGATHER_GroupY"): ("SUPERGATHER_GroupX", "SUPERGATHER_GroupY"),
    ("SUPERGATHER_CDP_X", "SUPERGATHER_CDP_Y"): ("SUPERGATHER_CDP_X", "SUPERGATHER_CDP_Y"),
    ("SUPERGATHER_INLINE_3D", "SUPERGATHER_CROSSLINE_3D"): ("SUPERGATHER_INLINE_3D", "SUPERGATHER_CROSSLINE_3D"),
}
# Ignore order of elements in each key
INDEX_TO_COORDS = {frozenset(to_list(key)): val for key, val in INDEX_TO_COORDS.items()}


def get_coords_cols(index_cols):
    """Return headers columns to get coordinates from depending on the type of headers index. See the mapping in
    `INDEX_TO_COORDS`."""
    coords_cols = INDEX_TO_COORDS.get(frozenset(to_list(index_cols)))
    if coords_cols is None:
        raise KeyError(f"Unknown coordinates columns for {index_cols} index")
    return coords_cols


def validate_cols_exist(df, cols):
    """Check if each column from `cols` is present either in the `df` DataFrame columns or index."""
    df_cols = set(df.columns) | set(df.index.names)
    missing_cols = set(to_list(cols)) - df_cols
    if missing_cols:
        raise ValueError(f"The following headers must be preloaded: {', '.join(missing_cols)}")


def get_cols(df, cols):
    """Extract columns from `cols` from the `df` DataFrame columns or index as a 2d `np.ndarray`."""
    validate_cols_exist(df, cols)
    # Avoid using direct pandas indexing to speed up selection of multiple columns from small DataFrames
    res = []
    for col in to_list(cols):
        col_values = df[col] if col in df.columns else df.index.get_level_values(col)
        res.append(col_values.values)
    return np.column_stack(res)


@njit(nogil=True)
def times_to_indices(times, samples, round=False):
    """Convert `times` to their indices in the increasing `samples` array. If some value of `times` is not present
    in `samples`, its index is linearly interpolated or extrapolated by the other indices of `samples`.

    Parameters
    ----------
    times : 1d np.ndarray of floats
        Time values to convert to indices.
    samples : 1d np.ndarray of floats
        Recording time for each trace value.
    round : bool, optional, defaults to False
        If `True`, round the obtained float indices to the nearest integer. Values exactly halfway between two adjacent
        integers are rounded to the nearest even one.

    Returns
    -------
    indices : 1d np.ndarray
        Array with positions of `times` in `samples`.

    Raises
    ------
    ValueError
        If `samples` is not increasing.
    """
    for i in range(len(samples) - 1):
        if samples[i+1] <= samples[i]:
            raise ValueError('The `samples` array must be increasing.')
    left_slope = 1 / (samples[1] - samples[0])
    right_slope = 1 / (samples[-1] - samples[-2])
    float_position = interpolate(times, samples, np.arange(len(samples), dtype=np.float32), left_slope, right_slope)
    return np.rint(float_position) if round else float_position


class Coordinates:
    """Define spatial coordinates of an object."""
    def __init__(self, *args, names=None):
        if names is None:
            names = ("X", "Y")
        names = tuple(to_list(names))
        if len(names) != 2:
            raise ValueError("Exactly two names must be passed.")

        if not args:
            args = (None, None)
        if len(args) != 2:
            raise ValueError("Exactly two coordinates must be passed.")

        self.coords = args
        self.names = names

    def __repr__(self):
        return f"Coordinates({self.coords[0]}, {self.coords[1]}, names={self.names})"

    def __str__(self):
        return f"({self.names[0]}: {self.coords[0]}, {self.names[1]}: {self.coords[1]})"

    def __iter__(self):
        return iter(self.coords)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, key):
        return self.coords[key]

    def __array__(self, dtype=None):
        return np.array(self.coords, dtype=dtype)


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
