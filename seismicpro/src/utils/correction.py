import numpy as np
from numba import njit


@njit(nogil=True, fastmath=True)
def get_hodograph(gather_data, time, offsets, velocity, sample_rate, fill_value=0):
    r""" Default approach for normal moveout computation for single hodograph. Corrected gather calculates
    as following:
    :math:`t_c = \sqrt{t^2 + \frac{l^2}{v^2}}`, where
        t_c - corrected time value.
        t - specified time value.
        l - distance from the source to receiver.
        v - velocity.

    Parameters
    ----------
    gather : np.ndarray
        Data for calculating normal moveout.
    time : int
        Time value to calculate normal moveout.
    offsets : array-like
        The distance from the source to the receiver for each trace.
    velocity : array-like
        Velocity value for NMO computation.
    sample_rate : int
        Step in milliseconds between signal amplitude measurements during shooting.

    Returns
    -------
    hodograph : 1d array
        NMO corrected hodograph.
    """
    hodograph = np.full(len(offsets), fill_value, dtype=gather_data.dtype)
    hodograph_times = (np.sqrt(time**2 + offsets**2/velocity**2) / sample_rate).astype(np.int32)
    for i in range(len(offsets)):
        hodograph_time = hodograph_times[i]
        if hodograph_time < len(gather_data):
            hodograph[i] = gather_data[hodograph_time, i]
    return hodograph


@njit(nogil=True)
def apply_nmo(gather_data, times, offsets, stacking_velocities, sample_rate):
    corrected_gather_data = np.empty_like(gather_data)
    for i, (time, stacking_velocity) in enumerate(zip(times, stacking_velocities)):
        corrected_gather_data[i] = get_hodograph(gather_data, time, offsets, stacking_velocity, sample_rate,
                                                 fill_value=np.nan)
    return np.ascontiguousarray(corrected_gather_data.T)
