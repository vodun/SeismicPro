"""Miscellaneous general utility functions"""

from functools import partial
from concurrent.futures import Future, Executor
import numpy as np

from numba import njit
from scipy import signal, fft

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

        
def dump_header(obj_with_headers, path, header_col, trace_id_cols=('FieldRecord', 'TraceNumber'), col_space=8, encoding="UTF-8"):
    """ Save values from a heders column to a file.

    Each line in the resulting file corresponds to one trace, where all columns but
    the last one store values from `trace_id_cols` headers and identify the trace
    while the last column stores the desired header.

    Parameters
    ----------
    obj_with_headers : Survey, or Gather
        Object, that contains headers to dump
    path : str
        Path to the file.
    trace_id_cols : tuple of str, defaults to ('FieldRecord', 'TraceNumber')
        Columns names from `self.headers` that act as trace id. These would be present in the file.
    header_col : str'
        Column name from `self.headers` to dump.
    col_space : int, defaults to 8
        The minimum width of each column.
    encoding : str, optional, defaults to "UTF-8"
        File encoding.

    Returns
    -------
    self : Gather
        Gather unchanged
    """

    rows = obj_with_headers[to_list(trace_id_cols) + [header_col]]

    # SEG-Y specification states that all headers values are integers, but first break values can be float
    row_fmt = '{:{col_space}.0f}' * (rows.shape[1] - 1) + '{:{col_space}.2f}\n'
    fmt = row_fmt * len(rows)
    rows_as_str = fmt.format(*rows.ravel(), col_space=col_space)

    with open(path, 'a', encoding=encoding) as f:
        f.write(rows_as_str)



@njit(nogil=True)
def strip_clip_indicator(clip_ind):
    n = len(clip_ind)
    for i0 in range(n):
        if not clip_ind[i0]:
            break

    i1 = n
    for j in range(n - 1 - i0):
        if not clip_ind[n - 1 - j]:
            break
        else:
            i1 -= 1
    return i0, i1


@njit(nogil=True)
def get_cliplen_indicator(traces):
    traces = np.atleast_1d(traces)

    indicator = (traces[..., 1:] == traces[..., :-1]).astype(np.int32)

    for i in range(1, indicator.shape[-1]):
        indicator[..., i] += indicator[..., i-1] * (indicator[..., i] == 1)

    return indicator


@njit(nogil=True)
def get_clip_indicator(traces, clip_len):
    traces = np.atleast_1d(traces)

    n_samples = traces.shape[0]

    if clip_len < 2 or clip_len > n_samples - 1:
        raise ValueError("Incorrect `clip_len`")

    ind_len = n_samples - clip_len + 1

    clip_indicator = np.full_like(traces[:ind_len], True, dtype=np.bool8)
    for curr_shift in range(1, clip_len):
        clip_indicator &= (traces[curr_shift:ind_len + curr_shift] == traces[:ind_len])

    return clip_indicator


@njit(nogil=True)
def has_clips(trace, clip_len):

    trace = np.asarray(trace)
    if trace.ndim != 1:
        raise ValueError("Only 1-D traces are allowed")

    clip_res = get_clip_indicator(trace, clip_len)

    i0, i1 = strip_clip_indicator(clip_res)

    return np.any(clip_res[i0:i1])


def get_maxabs_clips(traces):
    maxes = traces.max(axis=-1, keepdims=True)
    mins = traces.min(axis=-1, keepdims=True)

    res_plus = np.isclose(traces, maxes, atol=0)
    res_minus = np.isclose(traces, mins, atol=0)

    return ((res_plus[..., :-2] & res_plus[..., 1:-1] & res_plus[..., 2:])
            | (res_minus[..., :-2] & res_minus[..., 1:-1] & res_minus[..., 2:]))

def has_maxabs_clips(traces):
    return np.any(get_maxabs_clips(traces), axis=-1)

def calc_spikes(arr):
    with fft.set_workers(25):
        running_mean = signal.fftconvolve(arr, [[1,1,1]], mode='valid', axes=1)/3
    return (np.abs(arr[...,1:-1] - running_mean))

@njit
def fill_nulls(arr):

    n_samples = arr.shape[1]

    for i in range(arr.shape[0]):
        nan_indices = np.nonzero(np.isnan(arr[i]))[0]
        if len(nan_indices) > 0:
            j = nan_indices[-1]+1
            if j < n_samples:
                arr[i, :j] = arr[i, j]

@njit(nogil=True)
def get_const_indicator(traces, cmpval=None):

    if cmpval is None:
        indicator = (traces[..., 1:] == traces[..., :-1])
    else:
        indicator = (traces[..., 1:] == cmpval)

    brdr_zeros = np.zeros(traces.shape[:-1]+(1,), dtype=np.bool8)
    indicator = np.concatenate((brdr_zeros, indicator), axis=-1)

    return indicator.astype(np.int32)

@njit(nogil=True)
def get_constlen_indicator(traces, cmpval=None):

    old_shape = traces.shape

    traces = np.atleast_2d(traces)

    indicator = get_const_indicator(traces, cmpval)


    for i in range(1, indicator.shape[-1]):
        indicator[..., i] += indicator[..., i-1] * (indicator[..., i] == 1)


    for j in range(0, indicator.shape[-2]):
        for i in range(indicator.shape[-1] - 1, 0, -1):
            if indicator[..., j, i] != 0 and indicator[..., j, i - 1] != 0:
                indicator[..., j, i - 1] = indicator[..., j, i]

    if cmpval is None:
        indicator += (indicator > 0).astype(np.int32)

    return indicator.reshape(*old_shape)

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
