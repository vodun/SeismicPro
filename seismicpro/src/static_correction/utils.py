"""Static correction utils. """

import numpy as np


def calculate_depth_coefs(v_curr, v_next):
    if np.any(v_next - v_curr <= 0):
        raise ValueError("v_curr > v_next")
    return ((v_next**2 - v_curr**2)**.5) / (v_curr*v_next)

def calculate_layer_coefs(v_curr, v_next, v_last):
    return (v_next * v_last - v_curr**2) / (v_curr*v_last*(v_next**2 - v_curr**2)**.5)

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


def calculate_wv_by_v2(v2, coefs, max_wv):
    v1 = v2 / (coefs**2 * v2**2 + 1)**.5
    v1 = np.clip(v1, None, max_wv)
    return v1
