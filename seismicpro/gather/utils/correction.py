"""Implements functions for various gather corrections"""

# pylint: disable=not-an-iterable
import math

import numpy as np
from numba import njit, prange


@njit(nogil=True, parallel=True)
def compute_hodograph_times(offsets, times, velocities):
    """Calculate times of hyperbolic hodographs for each start time, corresponding stacking velocity and all offsets.
    Offsets and times are 1d `np.ndarray`s. Velocities are either a 1d `np.ndarray` or a scalar.
    The result is a 2d `np.ndarray` with shape `(len(times), len(offsets))`."""
    return np.sqrt(times.reshape(-1, 1) ** 2 + (offsets / velocities.reshape(-1, 1)) ** 2)


@njit(nogil=True, parallel=True)
def compute_crossover_mask(hodograph_times):
    crossover_mask = np.empty_like(hodograph_times, dtype=np.bool_)
    n = len(hodograph_times) - 1
    for i in prange(hodograph_times.shape[1]):
        t_next = hodograph_times[n, i]
        for j in range(n-1, -1, -1):
            t = hodograph_times[j, i]
            if t > t_next:
                crossover_mask[:j+1, i] = True
                crossover_mask[j+1:, i] = False
                break
            t_next = t
    return crossover_mask


@njit(nogil=True, parallel=True)
def compute_stretch_mask(hodograph_times, offsets, times, velocities, velocities_grad, max_stretch_factor):
    corrected_t0 = times.reshape(-1, 1) - (offsets**2 * velocities_grad.reshape(-1, 1)) / velocities.reshape(-1, 1)**3
    stretch_mask = (corrected_t0 <= 0) | (hodograph_times > (1 + max_stretch_factor) * corrected_t0)
    for i in prange(stretch_mask.shape[1]):
        for j in range(stretch_mask.shape[0] - 1, -1, -1):
            if stretch_mask[j, i]:
                stretch_mask[:j, i] = True
                break
    return stretch_mask


@njit(nogil=True, parallel=True)
def compute_muting_offsets(hodograph_times, offsets, times, velocities, velocities_grad, mute_crossover=False,
                           max_stretch_factor=None):
    muting_mask = np.full_like(hodograph_times, 0, dtype=np.bool_)
    if mute_crossover:
        muting_mask |= compute_crossover_mask(hodograph_times)
    if max_stretch_factor is not None:
        muting_mask |= compute_stretch_mask(hodograph_times, offsets, times, velocities, velocities_grad,
                                            max_stretch_factor)

    muting_offsets = np.full(len(times), np.inf, dtype=np.float32)
    for i in prange(len(muting_mask)):
        muted_offsets = offsets[muting_mask[i]]
        if muted_offsets.size > 0:
            muting_offsets[i] = muted_offsets.min()
    return muting_offsets


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
    if out is None:
        out = np.empty(len(hodograph_times), dtype=gather_data.dtype)
    for i, hodograph_sample in enumerate((hodograph_times - delay) / sample_interval):
        amplitude = fill_value
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
def apply_nmo(gather_data, offsets, sample_interval, delay, times, velocities, velocities_grad, interpolate=True,
              mute_crossover=False, max_stretch_factor=np.inf, fill_value=np.nan):
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
    # Explicit broadcasting of velocities and their gradients in case they are scalars.
    # Required for `parallel=True` flag
    velocities = np.ascontiguousarray(np.broadcast_to(velocities, times.shape))
    velocities_grad = np.ascontiguousarray(np.broadcast_to(velocities_grad, times.shape))

    hodograph_times = compute_hodograph_times(offsets, times, velocities)
    muting_offsets = compute_muting_offsets(hodograph_times, offsets, times, velocities, velocities_grad,
                                            mute_crossover=mute_crossover, max_stretch_factor=max_stretch_factor)

    corrected_gather_data = np.full_like(gather_data, fill_value=fill_value)
    for i in prange(times.shape[0]):
        get_hodograph(gather_data, offsets, sample_interval, delay, hodograph_times[i], interpolate=interpolate,
                      fill_value=fill_value, max_offset=muting_offsets[i], out=corrected_gather_data[:, i])
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
