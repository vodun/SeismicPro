import numpy as np
from numba import njit


@njit(nogil=True)
def compute_crossovers_times(hodograph_times):
    N = len(hodograph_times) - 1
    crossover_times = np.zeros(hodograph_times.shape[1])

    for i in range(hodograph_times.shape[1]):
        t0 =  hodograph_times[N, i]
        for j in range(1, N):
            t = hodograph_times[N - j, i]
            if t > t0:
                crossover_times[i] = N - j
                break
            t0 = t
    return crossover_times
