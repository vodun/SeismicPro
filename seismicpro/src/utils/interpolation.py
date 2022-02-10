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


@njit(nogil=True, fastmath=True)
def binomial(n, r):
    ''' Binomial coefficient, nCr, aka the "choose" function 
        n! / (r! * (n - r)!)
    '''
    p = 1    
    for i in range(1, min(r, n - r) + 1):
        p *= n
        p //= i
        n -= 1
    return p


@njit(nogil=True, parallel=True, fastmath=True)
def piecewise_polynomial(n, new_samples, old_samples, indices, data):
    res = np.empty((len(data), len(new_samples)), dtype=data.dtype)
    L = np.empty((len(new_samples), n + 1))
    
    for j in prange(len(data)):
        for i in range(len(new_samples)):
            # the index of the closest value from old_samples to the given interpolation point
            ix = indices[i] 
            
            # for given n, n + 1 points is required to construct polynom, find the index of leftmsot one
            ix_0 = max(ix - (n + 1) // 2, 0)
            ix_0 = min(ix_0, len(old_samples) - n - 1)
            
            # calculate Lagrange polynomials only once: they are the same at given position for all the traces    
            if j == 0:
                y = (new_samples[i] - old_samples[ix_0]) / (old_samples[1] - old_samples[0])
                
                common_multiplier = y
                for k in range(1, n + 1):
                    common_multiplier = common_multiplier * (y - k) / k
                    
                
                for k in range(n + 1):
                    if y == k:
                        L[i, k] = 1
                    else:
                        L[i, k] = common_multiplier * binomial(n, k) * (-1) ** (n - k)  / (y - k)
                        
            # perform the interpolation at given point: multiply and sum Lagrange polynomials and correspondoing function values
            res[j, i] = np.sum(data[j, ix_0: ix_0 + n + 1] * L[i])
    return res
