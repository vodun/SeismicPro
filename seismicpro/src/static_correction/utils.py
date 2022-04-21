"""Static correction utils. """

import numpy as np


def calculate_depth_coefs(v1, v2, avg_v2):
    if np.any(v2 - v1 <= 0):
        raise ValueError("v1 > v2")
    return (avg_v2 * v2 - v1**2) / (v1*avg_v2*(v2**2 - v1**2)**.5)


def calculate_velocities(v2, avg_v2, coefs, max_wv):
    sq_v2 = v2**2
    sq_avg_v2 = avg_v2**2
    sq_coefs = coefs**2

    b = 2*v2*avg_v2 + sq_v2*sq_avg_v2*sq_coefs
    D = b**2 - 4 * sq_v2 * sq_avg_v2 * (1 + sq_avg_v2*sq_coefs)
    D = np.sqrt(np.clip(D, 0, None))

    x1 = (b - D) / (2*sq_avg_v2*sq_coefs + 2)
    x1 = np.clip(np.sqrt(np.clip(x1, 0, None)), None, max_wv)

    x2 = (b + D) / (2*sq_avg_v2*sq_coefs + 2)
    x2 = np.clip(np.sqrt(np.clip(x2, 0, None)), None, max_wv)
    return x1, x2
