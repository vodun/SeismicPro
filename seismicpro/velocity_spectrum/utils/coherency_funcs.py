""" Functions for estimating hodograph coherency. """

# pylint: disable=not-an-iterable, missing-function-docstring
import numpy as np
from numba import njit, prange, jit_module


def stacked_amplitude(corrected_gather, w=1, abs=True):
    numerator = np.zeros_like(corrected_gather[:, 0])
    denominator = np.ones_like(corrected_gather[:, 0])
    for i in prange(corrected_gather.shape[0]):
        numerator[i] = np.nansum(corrected_gather[i, :])
        if abs:
            numerator[i] = np.abs(numerator[i])
        n = max(np.sum(~np.isnan(corrected_gather[i, :])), np.int64(1))
        denominator[i] = n / ((1 - w) * n ** 0.5 + w)
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
        numerator[i] = np.nansum(corrected_gather[i, :]) ** 2 
        denominator[i] = np.nansum(corrected_gather[i, :] ** 2) * np.sum(~np.isnan(corrected_gather[i, :])) 
    return numerator, denominator


def crosscorrelation(corrected_gather):
    numerator = np.zeros(corrected_gather.shape[0])
    denominator = np.ones(corrected_gather.shape[0])
    for i in prange(corrected_gather.shape[0]):
        numerator[i] = ((np.nansum(corrected_gather[i, :]) ** 2) - np.nansum(corrected_gather[i, :] ** 2)) / 2
    return numerator, denominator


def energy_normalized_crosscorrelation(corrected_gather):
    numerator = np.zeros(corrected_gather.shape[0])
    denominator = np.zeros(corrected_gather.shape[0])
    for i in prange(corrected_gather.shape[0]):
        input_enerty =  np.nansum(corrected_gather[i, :] ** 2)
        output_energy = np.nansum(corrected_gather[i, :]) ** 2
        numerator[i] = 2 * (output_energy - input_enerty) 
        denominator[i] = input_enerty * (np.sum(~np.isnan(corrected_gather[i, :])) - 1)
    return numerator, denominator


ALL_FASTMATH_FLAGS  = {'nnan', 'ninf', 'nsz', 'arcp', 'contract', 'afn', 'reassoc'}
jit_module(nopython=True, nogil=True, parallel=True, fastmath=ALL_FASTMATH_FLAGS - {'nnan'})
