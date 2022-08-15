import numpy as np
import pandas as pd

from ..utils import to_list, Coordinates


def get_param_names(n_refractors):
    return ["t0"] + [f"x{i}" for i in range(1, n_refractors)] + [f"v{i}" for i in range(1, n_refractors + 1)]


def postprocess_params(params):
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

def calc_df_to_dump(rv):
    columns = [*rv.coords.names] + list(rv.params.keys())
    data = [*rv.coords.coords] + list(rv.params.values())
    return pd.DataFrame.from_dict({col: [data] for col, data in zip(columns, data)})

def dump_rv(df_list, path, encoding, col_space):
    df_list = to_list(df_list)
    with open(path, 'w', encoding=encoding) as f:
        pd.concat(df_list).to_string(buf=f, col_space=col_space, float_format="%.2f", index=False)

def read_rv(path, encoding):
    df = pd.read_csv(path, sep=r'\s+', encoding=encoding)
    n_refractors = (len(df.columns) - 2) // 2
    coords_list, params_list = [], []
    for row in df.to_numpy():
        coords_list.append(Coordinates(names=tuple(df.columns[:2]), coords=tuple(row[:2])))
        params_list.append(dict(zip(get_param_names(n_refractors), row[2:])))
    return coords_list, params_list
