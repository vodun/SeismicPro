""" Functions for estimating hodograph coherency. """

# pylint: disable=not-an-iterable, missing-function-docstring
import numpy as np
from numba import njit, prange, jit_module


def stacked_amplitude(corrected_gather):
    numerator = np.zeros(corrected_gather.shape[0])
    denominator = np.ones(corrected_gather.shape[0])
    for i in prange(corrected_gather.shape[0]):
        numerator[i] = np.nanmean(corrected_gather[i, :])
    return numerator, denominator


def normalized_stacked_amplitude(corrected_gather):
    numerator = np.zeros(corrected_gather.shape[0])
    denominator = np.zeros(corrected_gather.shape[0])
    for i in prange(corrected_gather.shape[0]):
        numerator[i] = np.abs(np.nansum(corrected_gather[i, :]))
        denominator[i] = np.nansum(np.abs(corrected_gather[i, :]))
    return numerator, denominator


def semblance(corrected_gather):
    numerator = np.zeros(corrected_gather.shape[0])
    denominator = np.zeros(corrected_gather.shape[0])
    for i in prange(corrected_gather.shape[0]):
        numerator[i] = (np.nansum(corrected_gather[i, :]) ** 2)
        denominator[i] = np.nansum(corrected_gather[i, :] ** 2) * sum(~np.isnan(corrected_gather[i, :]))
    return numerator, denominator


def crosscorrelation(corrected_gather):
    numerator = np.zeros(corrected_gather.shape[0])
    denominator = np.full(corrected_gather.shape[0], 2)
    for i in prange(corrected_gather.shape[0]):
        numerator[i] = (np.nansum(corrected_gather[i, :]) ** 2) - np.nansum(corrected_gather[i, :] ** 2)
    return numerator, denominator


def energy_normalized_crosscorrelation(corrected_gather):
    numerator = np.zeros(corrected_gather.shape[0])
    denominator = np.zeros(corrected_gather.shape[0])
    for i in prange(corrected_gather.shape[0]):
        input_enerty =  np.nansum(corrected_gather[i, :] ** 2)
        output_energy = np.nansum(corrected_gather[i, :]) ** 2
        numerator[i] = output_energy - input_enerty
        denominator[i] = input_enerty * sum(~np.isnan(corrected_gather[i, :])) / 2
    return numerator, denominator


ALL_FASTMATH_FLAGS  = {'nnan', 'ninf', 'nsz', 'arcp', 'contract', 'afn', 'reassoc'}
ALLOWED_FASTMATH_FLAGS = ALL_FM_FLAGS - {'nnan'}

jit_module(nopython=True, nogil=True, parallel=True, fastmath=ALLOWED_FASTMATH_FLAGS)
