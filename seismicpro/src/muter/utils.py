import numpy as np
from numba import njit

@njit(nogil=True)
def compute_crossovers_times(times, velocity, offsets):
    N = len(velocity) - 1
    crossover_times = np.zeros_like(offsets)
    for i in range(len(offsets)):
        t0 = times[N] ** 2 + offsets[i] ** 2 / velocity[N] ** 2
        for j in range(1, N):
            t = times[N - j] ** 2 + offsets[i] ** 2 / velocity[N - j] ** 2
            if t > t0:
                crossover_times[i] = times[N - j]
                break
            t0 = t
    return crossover_times
