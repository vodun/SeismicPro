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


@njit(nogil=True)
def times_to_indices(times, samples, round=False):
    """Convert `times` to their indices in the increasing `samples` array. If some value of `times` is not present
    in `samples`, its index is linearly interpolated or extrapolated by the other indices of `samples`.

    Parameters
    ----------
    times : 1d np.ndarray of floats
        Time values to convert to indices.
    samples : 1d np.ndarray of floats
        Recording time for each trace value.
    round : bool, optional, defaults to False
        If `True`, round the obtained float indices to the nearest integer. Values exactly halfway between two adjacent
        integers are rounded to the nearest even one.

    Returns
    -------
    indices : 1d np.ndarray
        Array with positions of `times` in `samples`.

    Raises
    ------
    ValueError
        If `samples` is not increasing.
    """
    for i in range(len(samples) - 1):
        if samples[i+1] <= samples[i]:
            raise ValueError('The `samples` array must be increasing.')
    return _times_to_indices(times=times, samples=samples, round=round)


@njit(nogil=True)
def _times_to_indices(times, samples, round):
    left_slope = 1 / (samples[1] - samples[0])
    right_slope = 1 / (samples[-1] - samples[-2])
    float_position = interpolate(times, samples, np.arange(len(samples), dtype=np.float32), left_slope, right_slope)
    return np.rint(float_position) if round else float_position


@njit(nogil=True)
def calculate_basis_polynomials(x_new, x, n):
    """ Calculate the values of basis polynomials for Lagrange interpolation. """

    # Shift x to the zero, shift x_new accordingly. This does not affect interpolation
    x_new -= x.min()
    x -= x.min()

    N = n + 1
    polynomials = np.ones((len(x_new), N))

    # For given point, n + 1 neighbor samples are required to construct polynomial, find the index of leftmsot one
    if N % 2 == 1:
        leftmost_indices = np.rint(_times_to_indices(x_new , x, False)) - N // 2
    else:
        leftmost_indices = np.ceil(_times_to_indices(x_new , x, False)) - N // 2

    indices = leftmost_indices.reshape(-1, 1) + np.arange(N)
    sign = np.sign(indices + 1e-3)

    # Reflect indices from array borders
    div, mod = np.divmod(np.abs(indices), len(x) - 1)
    indices = np.where(div % 2, np.abs(len(x) - mod - 1), mod).astype(np.int32)

    times = np.empty_like(indices, dtype=np.float32)
    for i, ind in enumerate(indices):
        times[i] = x[ind]

    # Redflect times accordingly
    times = np.where(div % 2, x.max() - times,  times)
    times = (times + x.max() * div) * sign

    for i, (time, it) in enumerate(zip(times, x_new)):

        for k in range(N):
            for j in range(N):
                if k != j:
                    polynomials[i, k] *= (it - time[j]) / (time[k] - time[j])

    return polynomials, indices


@njit(nogil=True, parallel=True)
def piecewise_polynomial(x_new, x, y, n):
    """" Perform piecewise polynomial (with degree n) interpolation ."""
    is_1d = (y.ndim == 1)
    y = np.atleast_2d(y)
    res = np.zeros((len(y), len(x_new)), dtype=y.dtype)

    # calculate Lagrange basis polynomials only once: they are the same at given position for all the traces
    polynomials, indices = calculate_basis_polynomials(x_new, x, n)

    for j in prange(len(y)):  # pylint: disable=not-an-iterable
        for i, ix in enumerate(indices):
            # interpolate at given point: multiply base polynomials and correspondoing function values and sum
            for p in range(n + 1):
                res[j, i] += polynomials[i, p] * y[j, ix[p]]

    if is_1d:
        return res[0]
    return res
