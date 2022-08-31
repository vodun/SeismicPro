"""Miscellaneous utility functions for refractor velocity estimation"""

import numpy as np


def get_param_names(n_refractors):
    """Return names of parameters of a near-surface velocity model describing given number of refractors."""
    return ["t0"] + [f"x{i}" for i in range(1, n_refractors)] + [f"v{i}" for i in range(1, n_refractors + 1)]


def postprocess_params(params):
    """Postprocess array of parameters of a near-surface velocity model so that the following constraints are
    satisfied:
    - Intercept time is non-negative,
    - Crossover offsets are non-negative and increasing,
    - Velocities of refractors are non-negative and increasing.
    """
    is_1d = (params.ndim == 1)
    params = np.atleast_2d(params).copy()
    n_refractors = params.shape[1] // 2

    # Ensure that t0 is non-negative
    np.clip(params[:, 0], 0, None, out=params[:, 0])

    # Ensure that velocities of refractors are non-negative and increasing
    velocities = params[:, n_refractors:]
    np.clip(velocities[:, 0], 0, None, out=velocities[:, 0])
    np.maximum.accumulate(velocities, axis=1, out=velocities)

    # Ensure that crossover offsets are non-negative and increasing
    if n_refractors > 1:
        cross_offsets = params[:, 1:n_refractors]
        np.clip(cross_offsets[:, 0], 0, None, out=cross_offsets[:, 0])
        np.maximum.accumulate(cross_offsets, axis=1, out=cross_offsets)

    if is_1d:
        return params[0]
    return params
