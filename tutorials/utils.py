"""Utility funcitons for tutorials"""

import numpy as np
from scipy import signal

def generate_trace(reflection_event_time=(10,700,1200), reflection_event_amplitude=(6,-12,8),
                    nmo_velocity=(1.6,2.,2.4), wavelet_lenght=50, wavelet_width=5, **kwargs):

    """Generates a seismic trace using reflectivity parameters and trace's headers.

    Parameters
    ----------
    reflection_event_time : 1d-darray, defaults to (10, 700, 1200)
        Zero-offset times of a reflection events measured in samples.
    reflection_event_amplitude : 1d-darray, defaults to (6,-12,8)
        Amplitudes of reflection events.
    nmo_velocity : 1d-darray, defaults to (1.6, 2., 2.4)
        NMO velocities for the reflection events, m/ms.
    wavelet_lenght : int, defaults to 50
        Overall lenght of the vector with Ricker vawelet. Equivalent of `points` parameter of `scipy.signal.ricker`.
    wavelet_width : int, defaults to 5
         Width parameter of the wavelet itself. Equivalent of `a` parameter of `scipy.signal.ricker`.
    kwargs : dict
        Dict with trace header values.

    Returns
    -------
    trace : 1d-ndarray
        Generated seismic trace.
    """

    # inline3d = kwargs.get('INLINE_3D')
    n_samples = kwargs.get('TRACE_SAMPLE_COUNT')
    sample_rate = kwargs.get('TRACE_SAMPLE_INTERVAL')
    offset = kwargs.get('offset')
    times = np.array(reflection_event_time)
    reflections = np.array(reflection_event_amplitude)
    velocities = np.array(nmo_velocity)

    # Inversed normal moveout
    shifted_times = ((times**2 + offset**2 / (velocities * (sample_rate / 1000))**2)**0.5).astype(int)
    ref_series = np.zeros(max(n_samples, max(shifted_times)) + 1)

    # Tweak reflectivity of traces based in indline3D header to make Survey and Gathers statistics differ
    # reflections = reflections * (1 + 10 / (10 + inline3d))
    reflections = np.random.normal(reflections, np.abs(reflections) / 5)

    ref_series[shifted_times] = reflections
    ref_series[min(shifted_times):] += np.random.normal(1, 0.5, size=len(ref_series)-min(shifted_times))

    trace = np.convolve(ref_series, signal.ricker(wavelet_lenght, wavelet_width), mode='same')[:n_samples]
    trace = trace.astype(np.float32)

    return trace
