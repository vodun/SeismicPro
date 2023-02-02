"""Implements functions for various gather corrections"""

import math

import numpy as np
from numba import njit, prange

from .general_utils import mute_gather


@njit(nogil=True, fastmath=True)
def get_hodograph(gather_data, hodograph_times, sample_rate, interpolate=True, fill_value=np.nan, out=None):
    """Retrieve hodograph amplitudes from the `gather_data`.
    Hodograph is defined by `hodograph_times`: the event time for each trace of the gather.

    Parameters
    ----------
    gather_data : 2d np.ndarray
        Gather data to retrieve hodograph amplitudes from.
    hodograph_times : 1d np.array
        Event time for each trace of the hodograph, e.g `len(hodograph_times) == len(gather_data)`.
        Measured in milliseconds.
    sample_rate : float
        Sample rate of seismic traces. Measured in milliseconds.
    interpolate: bool, defaults to True
        Whether to perform linear interpolation to retrieve the hodograph event from the trace.
        If `False`, the nearest time sample amplitude is obtained.
    fill_value : float, defaults to np.nan
        Fill value to use if the traveltime is outside the gather bounds.
    out : np.array, optional
        The buffer to store result in. If not provided, allocate new array.

    Returns
    -------
    out : 1d array
        Gather amplitudes along a hodograph.
    """
    if out is None:
        out = np.empty(len(hodograph_times), dtype=gather_data.dtype)
    for i, hodograph_time in enumerate(hodograph_times / sample_rate):
        amplitude = fill_value
        if hodograph_time <= gather_data.shape[1] - 1:
            if interpolate:
                time_prev = math.floor(hodograph_time)
                time_next = math.ceil(hodograph_time)
                weight = time_next - hodograph_time
                amplitude = gather_data[i, time_prev] * weight + gather_data[i, time_next] * (1 - weight)
            else:
                amplitude = gather_data[i, round(hodograph_time)]
        out[i] = amplitude
    return out


@njit(nogil=True)
def compute_hodograph_times(offsets, times, velocities):
    """ Calculate the times of hyperbolic hodographs for each time of the gather with given stacking velocities. 
    Offsets, times and velocities are 1d np.arrays. 
    The result is 2d np.array with shape `(len(offsets), len(times))`."""
    return np.sqrt(times.reshape(-1, 1) ** 2 + (offsets / np.asarray(velocities).reshape(-1, 1)) **2)


@njit(nogil=True)
def compute_crossovers_times(hodograph_times):
    """ Given times for gather NMO correction, for each trace, find the latest time when a crossover event occurs.
    Used to mute the trace above this event. """
    n = len(hodograph_times) - 1
    crossover_times = np.zeros(hodograph_times.shape[1])

    for i in range(hodograph_times.shape[1]):
        t_prev =  hodograph_times[n, i]
        for j in range(n-1, 0):
            t = hodograph_times[j, i]
            if t > t_prev:
                crossover_times[i] = j
                break
            t_prev = t
    return crossover_times


@njit(nogil=True, parallel=True)
def apply_nmo(gather_data, times, offsets, stacking_velocities, sample_rate, mute_crossover=False, mute_stretch=False, fill_value=np.nan):
    r"""Perform gather normal moveout correction with given stacking velocities for each timestamp.

    The process of NMO correction removes the moveout effect on traveltimes, assuming that reflection traveltimes in a
    CDP gather follow hyperbolic trajectories as a function of offset:
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
        Gather data to apply NMO correction to with an ordinary shape of (num_traces, trace_length).
    times : 1d np.ndarray
        Recording time for each trace value. Measured in milliseconds.
    offsets : 1d np.ndarray
        The distance between source and receiver for each trace. Measured in meters.
    stacking_velocities : 1d np.ndarray or scalar
        Stacking velocities for each time. If scalar value, perform nmo with given velocity for each time.
        Measured in meters/milliseconds.
    sample_rate : float
        Sample rate of seismic traces. Measured in milliseconds.
    mute_crossover: bool, optional, defaults to False
        Whether to mute areas where the time reversal occurred after nmo corrections.
    mute_stretch: bool, optional, defaults to False
        Whether to mute areas where the stretching effect occurred after nmo corrections.
    fill_value : float, optional, defaults to np.nan
        Value used to fill the amplitudes outside the gather bounds after moveout.

    Returns
    -------
    corrected_gather_data : 2d array
        NMO corrected gather data with an ordinary shape of (num_traces, trace_length).
    """
    corrected_gather_data = np.full_like(gather_data, fill_value=fill_value)
    hodograph_times = compute_hodograph_times(offsets, times, stacking_velocities)

    for i in prange(times.shape[0]): # pylint: disable=not-an-iterable
        get_hodograph(gather_data, hodograph_times[i], sample_rate, fill_value=fill_value, out=corrected_gather_data[:, i])

    if mute_stretch:
        max_stretch_factor = 0.65 # Reasonable default value for max_stretch_factor
        stretch_times = np.interp(offsets, times * stacking_velocities * np.sqrt((1 + max_stretch_factor) ** 2 - 1), times)
        corrected_gather_data = mute_gather(corrected_gather_data, stretch_times, times, fill_value)

    if mute_crossover:
        crossovers_times = compute_crossovers_times(hodograph_times) * sample_rate
        corrected_gather_data = mute_gather(corrected_gather_data, crossovers_times, times, fill_value)

    return corrected_gather_data


@njit(nogil=True)
def apply_lmo(gather_data, trace_delays, fill_value):
    """Perform gather linear moveout correction with given delay for each trace.

    Parameters
    ----------
    gather_data : 2d np.ndarray
        Gather data to apply LMO correction to with shape (num_traces, trace_length).
    trace_delays : 1d np.ndarray
        Delay in samples introduced in each trace, positive values result in shifting gather traces down.
    fill_value: float
        Value used to fill the amplitudes outside the gather bounds after moveout.

    Returns
    -------
    corrected_gather : 2d array
        LMO corrected gather with shape (num_traces, trace_length).
    """
    corrected_gather = np.full_like(gather_data, fill_value)
    n_traces, trace_length = gather_data.shape
    for i in range(n_traces):
        if trace_delays[i] < 0:
            corrected_gather[i, :trace_delays[i]] = gather_data[i, -trace_delays[i]:]
        else:
            corrected_gather[i, trace_delays[i]:] = gather_data[i, :trace_length - trace_delays[i]]
    return corrected_gather
