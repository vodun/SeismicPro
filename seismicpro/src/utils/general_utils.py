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
