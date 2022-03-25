"""Implements a class for linear 1d interpolation and extrapolation"""

import numpy as np
from numba import njit, prange


@njit(nogil=True)
def interpolate(x_new, x, y, left_slope, right_slope):
    """Return a 1d piecewise linear interpolant to a function defined by pairs of data points `(x, y)`, evaluated at
    `x_new`. Function values at points outside the `x` range will be linearly extrapolated using passed slopes."""
    res = np.interp(x_new, x, y)
    for i, curr_x in enumerate(x_new):
        if curr_x < x[0]:
            res[i] = y[0] - left_slope * (x[0] - curr_x)
        elif curr_x > x[-1]:
            res[i] = y[-1] + right_slope * (curr_x - x[-1])
    return res


#pylint: disable=invalid-name
class interp1d:
    """Return a 1d piecewise linear interpolant to a function defined by pairs of data points `(x, y)`. Function values
    at points outside the `x` range will be linearly extrapolated.

    Parameters
    ----------
    x : 1d array-like
        X coordinates of function values.
    y : 1d array-like
        Function values, evaluated at `x`. Must match the length of `x`.
    """
    def __init__(self, x, y):
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        if len(x) < 2:
            raise ValueError("At least two points should be passed to perform interpolation")

        ind = np.argsort(x, kind="mergesort")
        self.x = x[ind]
        self.y = y[ind]

        self.left_slope = (self.y[1] - self.y[0]) / (self.x[1] - self.x[0])
        self.right_slope = (self.y[-1] - self.y[-2]) / (self.x[-1] - self.x[-2])

    def __call__(self, x):
        """Evaluate the interpolant at passed coordinates `x`.

        Parameters
        ----------
        x : 1d array-like
            Points to evaluate the interpolant at.

        Returns
        -------
        y : 1d array-like
            Interpolated values, matching the length of `x`.
        """
        x = np.array(x)
        is_scalar_input = (x.ndim == 0)
        res = interpolate(x.ravel(), self.x, self.y, self.left_slope, self.right_slope)
        return res.item() if is_scalar_input else res


@njit
def _times_to_indices(times, samples, round):
    left_slope = 1 / (samples[1] - samples[0])
    right_slope = 1 / (samples[-1] - samples[-2])
    float_position = interpolate(times, samples, np.arange(len(samples), dtype=np.float32), left_slope, right_slope)
    return np.rint(float_position) if round else float_position


@njit(nogil=True)
def binomial(n, r):
    """ Binomial coefficient, nCr,
    n! / (r! * (n - r)!) """

    p = 1
    for i in range(1, min(r, n - r) + 1):
        p *= n
        p //= i
        n -= 1
    return p


@njit(nogil=True)
def calculate_basis_polynomials(n, new_samples, old_samples, leftmost_indices):
    """ Calculate the values of basis polynomials for Lagrange interpolation. """
    polynomials = np.empty((len(new_samples), n + 1))
    binomials = [binomial(n, k) for k in range(n+1)]

    sample_rate = old_samples[1] - old_samples[0]
    for i, (ix, it) in enumerate(zip(leftmost_indices, new_samples)):
        y = (it - old_samples[ix]) / sample_rate

        common_multiplier = y
        for k in range(1, n + 1):
            common_multiplier = common_multiplier * (y - k) / k

        for k in range(n + 1):
            if y == k:
                polynomials[i, k] = 1
            else:
                polynomials[i, k] = common_multiplier * binomials[k] * (-1) ** (n - k)  / (y - k)
    return polynomials


@njit(nogil=True, parallel=True)
def piecewise_polynomial(n, new_samples, old_samples, data):
    """" Perform piecewise polynomial(with degree n) interpolation ."""
    res = np.empty((len(data), len(new_samples)), dtype=data.dtype)

    # for given point, n + 1 neighbor samples are required to construct polynomial, find the index of leftmsot one
    indices = np.ceil(_times_to_indices(new_samples, old_samples, False))
    leftmost_indices = np.clip(indices - (n + 1) / 2, 0, len(old_samples) - n - 1).astype(np.int32)

    # calculate Lagrange basis polynomials only once: they are the same at given position for all the traces
    polynomials = calculate_basis_polynomials(n, new_samples, old_samples, leftmost_indices)

    for j in prange(len(data)):  # pylint: disable=not-an-iterable
        for i, ix in enumerate(leftmost_indices):
            # interpolate at given point: multiply base polynomials and correspondoing function values and sum
            res[j, i] = np.sum(polynomials[i] * data[j, ix: ix + n + 1])
    return res
