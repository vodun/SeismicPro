"""Implements functions for various gather corrections"""

# pylint: disable=not-an-iterable
import math

import numpy as np
from numba import njit, prange


@njit(nogil=True, fastmath=True)
def get_hodograph(gather_data, offsets, sample_interval, delay, hodograph_times, interpolate=True, fill_value=np.nan,
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
    delay : float
        Delay recording time of seismic traces. Measured in milliseconds.
    interpolate: bool, optional, defaults to True
        Whether to perform linear interpolation to retrieve the hodograph event from the trace. If `False`, the nearest
        time sample amplitude is used.
    fill_value : float, optional, defaults to np.nan
        Fill value to use if the traveltime is outside the gather bounds.
    max_offset: float, optional, defaults to np.inf
        The maximum offset value for which the hodograph is being tracked.
    out : np.array, optional
        The buffer to store the result in. If not provided, a new array is allocated.

    Returns
    -------
    out : 1d array
        Gather amplitudes along a hodograph.
    """
    n_times = len(hodograph_times)
    if out is None:
        out = np.empty(n_times, dtype=gather_data.dtype)
    for i in range(n_times):
        amplitude = fill_value
        hodograph_sample = (hodograph_times[i] - delay) / sample_interval
        if offsets[i] < max_offset and 0 <= hodograph_sample <= gather_data.shape[1] - 1:
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
def apply_constant_velocity_nmo(gather_data, offsets, sample_interval, delay, times, velocity, interpolate=True,
                                max_stretch_factor=np.inf, fill_value=np.nan):
    corrected_gather_data = np.full((len(offsets), len(times)), fill_value=fill_value, dtype=gather_data.dtype)
    for i in prange(len(times)):
        hodograph_times = np.sqrt(times[i]**2 + (offsets / velocity)**2)
        max_offset = times[i] * velocity * np.sqrt((1 + max_stretch_factor)**2 - 1)
        get_hodograph(gather_data, offsets, sample_interval, delay, hodograph_times, interpolate=interpolate,
                      fill_value=fill_value, max_offset=max_offset, out=corrected_gather_data[:, i])
    return corrected_gather_data


@njit(nogil=True, parallel=True)
def apply_nmo(gather_data, offsets, sample_interval, delay, times, velocities, velocities_grad, interpolate=True,
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
        Stacking velocities for each time. If scalar, the same velocity is used for all times.
        Measured in meters/milliseconds.
    sample_interval : float
        Sample interval of seismic traces. Measured in milliseconds.
    delay : float
        Delay recording time of seismic traces. Measured in milliseconds.
    mute_crossover: bool, optional, defaults to False
        Whether to mute areas where time reversal occurred after NMO correction.
    max_stretch_factor : float, optional, defaults to np.inf
        Maximum allowable factor for the muter that attenuates the effect of waveform stretching after NMO correction.
        The lower the value, the stronger the mute. In case np.inf (default) no mute is applied. Reasonably good value
        is 0.65.
    fill_value : float, optional, defaults to np.nan
        Value used to fill the amplitudes outside the gather bounds after moveout.

    Returns
    -------
    corrected_gather_data : 2d array
        NMO corrected gather data with an ordinary shape of (num_traces, trace_length).
    """
    hodograph_times = np.empty((len(times), len(offsets)), dtype=np.float32)
    max_offsets = np.empty(len(times), dtype=np.float32)
    for i in prange(len(times)):
        hodograph_times[i] = np.sqrt(times[i]**2 + (offsets / velocities[i])**2)
        corrected_t0 = times[i] - offsets**2 * velocities_grad[i] / velocities[i]**3
        stretch_mask = (corrected_t0 <= 0) | (hodograph_times[i] > (1 + max_stretch_factor) * corrected_t0)
        muted_offsets = offsets[stretch_mask]
        max_offsets[i] = muted_offsets.min() if muted_offsets.size > 0 else np.inf

    min_max_offset = max_offsets[-1]
    for i in range(len(max_offsets) - 2, -1, -1):
        if max_offsets[i] < min_max_offset:
            min_max_offset = max_offsets[i]
        else:
            max_offsets[i] = min_max_offset

    corrected_gather_data = np.full((len(offsets), len(times)), fill_value=fill_value, dtype=gather_data.dtype)
    for i in prange(len(times)):
        get_hodograph(gather_data, offsets, sample_interval, delay, hodograph_times[i], interpolate=interpolate,
                      fill_value=fill_value, max_offset=max_offsets[i], out=corrected_gather_data[:, i])
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
