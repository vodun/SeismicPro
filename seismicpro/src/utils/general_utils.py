import numpy as np
from numba import njit


def to_list(obj):
    """Cast an object to a list. Almost identical to `list(obj)` for 1-D
    objects, except for `str`, which won't be split into separate letters but
    transformed into a list of a single element.
    """
    return np.array(obj).ravel().tolist()


@njit
def find_stats(array):
    min_value = max_value = array[0]
    tr_sum = tr_sq_sum = 0
    for i in range(1, len(array)):
        min_value = min(array[i], min_value)
        max_value = max(array[i], max_value)
        tr_sum += array[i]
        tr_sq_sum += array[i]**2
    return min_value, max_value, tr_sum, tr_sq_sum


@njit
def create_supergather_index(lines, size):
    area_size = size[0] * size[1]
    shifts_i = np.arange(size[0]) - size[0] // 2
    shifts_x = np.arange(size[1]) - size[1] // 2
    supergather_lines = np.empty((len(lines) * area_size, 4), dtype=lines.dtype)
    for ix in range(len(lines)):
        i, x = lines[ix]
        for ix_i in range(size[0]):
            for ix_x in range(size[1]):
                row = np.array([i, x, i + shifts_i[ix_i], x + shifts_x[ix_x]])
                supergather_lines[ix * area_size + ix_i * size[1] + ix_x] = row
    return supergather_lines
