import numpy as np
from numba import njit, prange


@njit(nogil=True)
def interpolate(x_new, x, y, left_slope, right_slope):
    res = np.interp(x_new, x, y)
    for i, curr_x in enumerate(x_new):
        if curr_x < x[0]:
            res[i] = y[0] - left_slope * (x[0] - curr_x)
        elif curr_x > x[-1]:
            res[i] = y[-1] + right_slope * (curr_x - x[-1])
    return res


class interp1d:
    def __init__(self, x, y):
        x = np.array(x)
        y = np.array(y)

        if len(x) < 2:
            raise ValueError

        ind = np.argsort(x, kind="mergesort")
        self.x = x[ind]
        self.y = y[ind]

        self.left_slope = (y[1] - y[0]) / (x[1] - x[0])
        self.right_slope = (y[-1] - y[-2]) / (x[-1] - x[-2])

    def __call__(self, x):
        x = np.array(x)
        res = interpolate(x, self.x, self.y, self.left_slope, self.right_slope)
        return res.item() if x.ndim == 0 else res
