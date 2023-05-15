"""Implements functions for various gather corrections"""

# pylint: disable=not-an-iterable
import math

import numpy as np
from numba import njit, prange


@njit(nogil=True, fastmath=True)
def get_hodograph(gather_data, offsets, hodograph_times, sample_interval, interpolate=True, fill_value=np.nan,
                  max_offset=np.inf, out=None):
    """Retrieve hodograph amplitudes from the `gather_data`.

    Hodograph is defined by `hodograph_times`: an array of event times for each trace of the gather.

    Parameters
    ----------
    gather_data : 2d np.ndarray
        Gather data to retrieve hodograph amplitudes from.
    offsets : 1d np.ndarray
        The distance between source and receiver for each trace. Measured in meters.
    hodograph_times : 1d np.array
        Event time for each gather trace. Must match the length of `gather_data`. Measured in milliseconds.
    sample_interval : float
        Sample interval of seismic traces. Measured in milliseconds.
    interpolate: bool, optional, defaults to True
        Whether to perform linear interpolation to retrieve the hodograph event from the trace.
        If `False`, the nearest time sample amplitude is obtained.
    fill_value : float, optional, defaults to np.nan
        Fill value to use if the traveltime is outside the gather bounds.
    max_offset: float, optional, defaults to np.inf
        The maximum offset value for which the hodograph being tracked.
    out : np.array, optional
        The buffer to store result in. If not provided, allocate new array.

    Returns
    -------
    out : 1d array
        Gather amplitudes along a hodograph.
    """
    if out is None:
        out = np.empty(len(hodograph_times), dtype=gather_data.dtype)
    for i, hodograph_sample in enumerate(hodograph_times / sample_interval):
        amplitude = fill_value
        if offsets[i] <= max_offset and hodograph_sample <= gather_data.shape[1] - 1:
            if interpolate:
                time_prev = math.floor(hodograph_sample)
                time_next = math.ceil(hodograph_sample)
                weight = time_next - hodograph_sample
                amplitude = gather_data[i, time_prev] * weight + gather_data[i, time_next] * (1 - weight)
            else:
                amplitude = gather_data[i, round(hodograph_sample)]
        out[i] = amplitude
    return out


@njit(nogil=True, parallel=True)
def compute_hodograph_times(offsets, times, velocities):
    """Calculate times of hyperbolic hodographs for each start time, corresponding stacking velocity and all offsets.
    Offsets and times are 1d `np.ndarray`s. Velocities are either a 1d `np.ndarray` or a scalar.
    The result is a 2d `np.ndarray` with shape `(len(times), len(offsets))`."""
    # Explicit broadcasting velocities, in case it's scalar. Required for `parallel=True` flag
    velocities = np.ascontiguousarray(np.broadcast_to(velocities, times.shape))
    return np.sqrt(times.reshape(-1, 1) ** 2 + (offsets / velocities.reshape(-1, 1)) ** 2)


@njit(nogil=True, parallel=True)
def compute_crossover_offsets(hodograph_times, times, offsets):
    """Given `hodograph_times` for gather NMO correction, find an offset after which the crossover events occur for
    each timestamp.

    Parameters
    ----------
    hodograph_times : 2d np.ndarray
        Array storing the times of hyperbolic hodographs for gather NMO correction. Has shape
        `(len(times), len(offsets))`.
    times : 1d np.ndarray
        Gather timestamps.
    offsets : 1d np.ndarray
        Gather offsets.

    Returns
    -------
    crossover_offsets : 1d np.array
        An array of offsets where crossover events occur for each timestamp. Has shape `(len(times),)`.
    """
    n = len(hodograph_times) - 1
    crossover_times = np.zeros(hodograph_times.shape[1])

    for i in prange(hodograph_times.shape[1]):
        t_prev = hodograph_times[n, i]
        for j in range(n-1, 0, -1):
            t = hodograph_times[j, i]
            if t > t_prev:
                crossover_times[i] = j
                break
            t_prev = t

    time_max = crossover_times[0]
    for i in range(1, len(crossover_times)):
        if crossover_times[i] > time_max:
            time_max = crossover_times[i]
        else:
            crossover_times[i] = time_max

    return np.interp(times, crossover_times, offsets)

@njit(nogil=True, parallel=True)
def apply_nmo(gather_data, times, offsets, stacking_velocities, sample_interval, mute_crossover=False,
              max_stretch_factor=np.inf, fill_value=np.nan):
    r"""Perform gather normal moveout correction with given stacking velocities for each timestamp.

    The process of NMO correction removes the moveout effect on traveltimes, assuming that reflection traveltimes in a
    CDP gather follow hyperbolic trajectories as a function of offset:
    :math:`t(l) = \sqrt{t(0)^2 + \frac{l^2}{v^2}}`, where:
        t(l) - travel time at offset `l`,
        t(0) - travel time at zero offset,
        l - seismic trace offset,
        v - seismic wave velocity.

    If stacking velocity was properly picked, the reflection events on a corrected CDP gather are mostly flattened
    across the offset range.

    Parameters
    ----------
    gather_data : 2d np.ndarray
        Gather data to apply NMO correction to with an ordinary shape of (n_traces, n_samples).
    times : 1d np.ndarray
        Recording time for each trace value. Measured in milliseconds.
    offsets : 1d np.ndarray
        The distance between source and receiver for each trace. Measured in meters.
    stacking_velocities : 1d np.ndarray or scalar
        Stacking velocities for each time. If scalar value, perform nmo with given velocity for each time.
        Measured in meters/milliseconds.
    sample_interval : float
        Sample interval of seismic traces. Measured in milliseconds.
    mute_crossover: bool, optional, defaults to False
        Whether to mute areas where the time reversal occurred after nmo corrections.
    max_stretch_factor : float, optional, defaults to np.inf
        Max allowable factor for the muter that attenuates the effect of waveform stretching after nmo correction.
        This mute is applied after nmo correction for each provided velocity and before coherency calculation.
        The lower the value, the stronger the mute. In case np.inf (default) no mute is applied. 
        Reasonably good value is 0.65
    fill_value : float, optional, defaults to np.nan
        Value used to fill the amplitudes outside the gather bounds after moveout.

    Returns
    -------
    corrected_gather_data : 2d array
        NMO corrected gather data with an ordinary shape of (num_traces, trace_length).
    """
    corrected_gather_data = np.full_like(gather_data, fill_value=fill_value)
    hodograph_times = compute_hodograph_times(offsets, times, stacking_velocities)

    max_offsets = times * stacking_velocities * np.sqrt((1 + max_stretch_factor) ** 2 - 1)

    if mute_crossover:
        crossover_offsets = compute_crossover_offsets(hodograph_times, times, offsets)
        max_offsets = np.minimum(max_offsets, crossover_offsets)

    for i in prange(times.shape[0]):
        get_hodograph(gather_data, offsets, hodograph_times[i], sample_interval, fill_value=fill_value,
                      max_offset=max_offsets[i], out=corrected_gather_data[:, i])

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
