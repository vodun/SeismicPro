from math import nan
from warnings import WarningMessage
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
    """Calculate a DataFrame with coordinates and parameter of the passed RefractorVelocity.

    Parameters
    ----------
    rv : RefractorVelocity
        RefractorVelocity instance.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with the coordinates and parameters of a RefractorVelocity.
    """
    max_offset = np.nan if rv.offsets is None else rv.offsets.max()
    columns = [*rv.coords.names] + list(rv.params.keys()) + ["max_offset"]
    data = [*rv.coords.coords] + list(rv.params.values()) + [max_offset]
    return pd.DataFrame.from_dict({col: [data] for col, data in zip(columns, data)})

def dump_rv(df_list, path, encoding, min_col_size):
    """Dump list of DataFrame to a file.

    Each DataFrame in the list should have the coords (x and y) and parameters ("t0", "x1"..."x{n-1}", "v1"..."v{n}")
    of one RefractorVelocity instance.
    DataFrame example :
        SourceX   SourceY        t0        x1        v1        v2 max_offset
        1111100   2222220     50.00   1000.00   1500.00   2000.00    2000.00

    Parameters
    ----------
    df_list : iterable of :class:`~pandas.DataFrame`
        Each DataFrame in the passed list should contain coordinates and parameters of a RefractorVelocity.
    path : str
        Path to the created file.
    encoding : str, optional, defaults to "UTF-8"
        File encoding.
    min_col_size : int
        Minimum size of each columns in file.

    Returns
    -------
    None
    """
    with open(path, 'w', encoding=encoding) as f:
        pd.concat(df_list).to_string(buf=f, col_space=min_col_size, float_format="%.2f", index=False)

def load_rv(path, encoding):
    """Load the coordinates and parameters of RefractorVelocity from a file.

    Parameters
    ----------
    path : str
        Path to the file.
    encoding : str, optional, defaults to "UTF-8"
        File encoding.

    Returns
    -------
    coords_list : list of :class:`~utils.Coordinates`
        List of Coordinates instances loaded from a file.
    params_list : list of dict
        List of parameters of :class:`~RefractorVelocity`
    max_offset_list : list of float
        List of max offsets.
    """
    df = pd.read_csv(path, sep=r'\s+', encoding=encoding)
    n_refractors = (len(df.columns) - 3) // 2
    coords_list, params_list, max_offset_list = [], [], []
    for row in df.to_numpy(na_value=np.nan):
        if np.any(np.isnan(row[:-1])):
            raise ValueError(f"Insufficient parameters in the file to create a correct RefractorVelocity instance.")
        coords_list.append(Coordinates(names=tuple(df.columns[:2]), coords=tuple(row[:2].astype(int))))
        params_list.append(dict(zip(get_param_names(n_refractors), row[2:-1])))
        max_offset_list.append(row[-1])
    return coords_list, params_list, max_offset_list
