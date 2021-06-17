import numpy as np
from numba import njit, prange


def to_list(obj):
    """Cast an object to a list. Almost identical to `list(obj)` for 1-D
    objects, except for `str`, which won't be split into separate letters but
    transformed into a list of a single element.
    """
    return np.array(obj).ravel().tolist()


def maybe_copy(obj, inplace=False):
    return obj if inplace else obj.copy()


@njit(nogil=True)
def calculate_stats(trace):
    trace_min, trace_max = np.inf, -np.inf
    trace_sum, trace_sq_sum = 0, 0
    for sample in trace:
        trace_min = min(sample, trace_min)
        trace_max = max(sample, trace_max)
        trace_sum += sample
        trace_sq_sum += sample**2
    return trace_min, trace_max, trace_sum, trace_sq_sum


@njit(nogil=True)
def create_supergather_index(lines, size):
    area_size = size[0] * size[1]
    shifts_i = np.arange(size[0]) - size[0] // 2
    shifts_x = np.arange(size[1]) - size[1] // 2
    supergather_lines = np.empty((len(lines) * area_size, 4), dtype=lines.dtype)
    for ix, (i, x) in enumerate(lines):
        for ix_i, shift_i in enumerate(shifts_i):
            for ix_x, shift_x in enumerate(shifts_x):
                row = np.array([i, x, i + shift_i, x + shift_x])
                supergather_lines[ix * area_size + ix_i * size[1] + ix_x] = row
    return supergather_lines


@njit(nogil=True)
def convert_times_to_mask(times, sample_rate, mask_length):
    times_ixs = np.rint(times / sample_rate).astype(np.int32)
    mask = (np.arange(mask_length) - times_ixs.reshape(-1, 1)) >= 0
    return mask


@njit(nogil=True, parallel=True)
def convert_mask_to_pick(mask, sample_rate, threshold):
    picking_array = np.empty(len(mask), dtype=np.int32)
    for i in prange(len(mask)):
        trace = mask[i]
        max_len, curr_len, picking_ix = 0, 0, 0
        for j, sample in enumerate(trace):
            # Count length of current sequence of ones
            if sample >= threshold:
                curr_len += 1
            else:
                # If the new longest sequence found
                if curr_len > max_len:
                    max_len = curr_len
                    picking_ix = j
                curr_len = 0
        # If the longest sequence found in the end of the trace
        if curr_len > max_len:
            picking_ix = len(trace)
            max_len = curr_len
        picking_array[i] = picking_ix - max_len
    return picking_array * sample_rate


@njit(nogil=True)
def mute_gather(gather_data, muting_times, sample_rate, fill_value):
    mask = convert_times_to_mask(times=muting_times, sample_rate=sample_rate, mask_length=gather_data.shape[1])
    data_shape = gather_data.shape
    gather_data = gather_data.reshape(-1)
    mask = mask.reshape(-1)
    gather_data[~mask] = fill_value
    return gather_data.reshape(data_shape)


@njit(nogil=True)
def clip(data, data_min, data_max):
    """Limit the `data` values.

    Given an interval [`data_min`, `data_max`], values outside the interval are clipped to the interval edges.

    Parameters
    ----------
    data : np.ndarray
        Data to clip.
    data_min : int, float
        Minimum value of interval.
    data_max : int, float
        Maximum value of interval.

    Returns
    -------
    data : np.ndarray
        Clipped data with the same shape.
    """
    data_shape = data.shape
    data = data.reshape(-1)
    for i in range(len(data)):
        data[i] = min(max(data[i], data_min), data_max)
    return data.reshape(data_shape)
