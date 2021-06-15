"""Implements functions for various gather corrections"""

import numpy as np
from numba import njit


@njit(nogil=True, fastmath=True)
def get_hodograph(gather_data, time, offsets, velocity, sample_rate, fill_value=0):
    r"""Return gather amplitudes for a reflection traveltime curve starting from time `time` with velocity `velocity`,
    assuming that it follows hyperbolic trajectory as a function of offset given by:
    :math:`t(l) = \sqrt{t(0)^2 + \frac{l^2}{v^2}}`, where:
        t(l) - travel time at offset `l`,
        t(0) - travel time at zero offset,
        l - seismic trace offset,
        v - seismic wave velocity.

    Parameters
    ----------
    gather_data : 2d np.ndarray
        Gather data to apply NMO correction to. The data is stored in a transposed form, compared to `Gather.data` due
        to performance reasons, so that `gather_data.shape` is (trace_lenght, num_traces).
    time : float
        Seismic wave travel time at zero offset. Measured in milliseconds.
    offsets : 1d np.ndarray
        The distance between source and receiver for each trace. Measured in meters.
    velocity : float
        Seismic velocity value for traveltime calculation. Measured in meters/milliseconds.
    sample_rate : float
        Sample rate of seismic traces. Measured in milliseconds.
    fill_value : float, defaults to 0
        Fill value to use if the traveltime is outside the gather bounds for some given offsets.

    Returns
    -------
    hodograph : 1d array
        Gather amplitudes along a hyperbolic traveltime.
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
    r"""Perform gather normal moveout correction with given stacking velocities for each timestamp.

    The process of NMO correction removes the moveout effect on traveltimes, assuming that reflection traveltimes in a
    CMP gather follow hyperbolic trajectories as a function of offset:
    :math:`t(l) = \sqrt{t(0)^2 + \frac{l^2}{v^2}}`, where:
        t(l) - travel time at offset `l`,
        t(0) - travel time at zero offset,
        l - seismic trace offset,
        v - seismic wave velocity.

    If stacking velocity was picked correctly, the reflection events of a CDP gather are mostly flattened across the
    offset range.

    Parameters
    ----------
    gather_data : 2d np.ndarray
        Gather data to apply NMO correction to with an ordinary shape of (num_traces, trace_lenght).
    times : 1d np.ndarray
        Recording time for each trace value. Measured in milliseconds.
    offsets : 1d np.ndarray
        The distance between source and receiver for each trace. Measured in meters.
    stacking_velocities : 1d np.ndarray
        Stacking velocities for each time. Matches the length of `times`. Measured in meters/milliseconds.
    sample_rate : float
        Sample rate of seismic traces. Measured in milliseconds.

    Returns
    -------
    corrected_gather : 2d array
        NMO corrected gather with an ordinary shape of (num_traces, trace_lenght).
    """
    # Transpose gather_data to increase performance
    gather_data = gather_data.T
    corrected_gather_data = np.empty_like(gather_data)
    for i, (time, stacking_velocity) in enumerate(zip(times, stacking_velocities)):
        corrected_gather_data[i] = get_hodograph(gather_data, time, offsets, stacking_velocity, sample_rate,
                                                 fill_value=np.nan)
    return np.ascontiguousarray(corrected_gather_data.T)
