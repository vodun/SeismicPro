import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from tqdm.auto import tqdm

from .utils import calculate_depth_coefs, calculate_wv_by_v2, calculate_velocities
from seismicpro.src.utils import to_list
from seismicpro.src.metrics import MetricMap


DEFAULT_COLS = {
    "source": ["SourceX", "SourceY"],
    "rec": ["GroupX", "GroupY"]
}


USED_COLS = [*sum(DEFAULT_COLS.values(), []), "offset", "SourceDepth", "CDP_X", "CDP_Y"]


class StaticCorrection:
    def __init__(self, survey, first_breaks_col, interpolator):
        self.survey = survey.copy()
        self.first_breaks_col = first_breaks_col
        self.n_layers = None

        headers_cols = USED_COLS + [first_breaks_col]
        self.headers = pd.DataFrame(self.survey[headers_cols], columns=headers_cols)

        self.source_params = self._create_params_df(name="source")
        self.rec_params = self._create_params_df(name="rec")

        self._add_cols_to_params("source", columns="SourceDepth")
        self._add_wv_to_params("source", interpolator=interpolator)
        self._add_wv_to_params("rec", interpolator=interpolator)

        self._set_traces_layer(interpolator=interpolator)

    def _create_params_df(self, name):
        coord_names = self._get_cols(name)
        unique_coords = np.unique(self.headers[coord_names], axis=0).astype(np.int32)
        return pd.DataFrame(unique_coords, columns=coord_names).set_index(coord_names)

    def _add_cols_to_params(self, name, columns):
        coord_names = self._get_cols(name)
        data = self.headers[coord_names + to_list(columns)].drop_duplicates().values
        if data.shape[0] != getattr(self, f"{name}_params").shape[0]:
            raise ValueError("Value in column(s) to add must be unique for each source/rec.")
        self._update_params(name=name, coords=data[:, :2], values=data[:, 2:], columns=columns)

    def _add_wv_to_params(self, name, interpolator):
        unique_coords = to_list(getattr(self, f"{name}_params").index)
        wv_params = interpolator(unique_coords)
        self.n_layers = wv_params.shape[1] // 2 + 1

        names = [f"x{i+1}" for i in range(self.n_layers-1)] + [f"v{i+1}" for i in range(self.n_layers)]
        self._update_params(name=name, coords=unique_coords, values=wv_params, columns=names)

    def _set_traces_layer(self, interpolator):
        coords = self.headers[["CDP_X", "CDP_Y"]].values
        crossovers = interpolator(coords)[:, :self.n_layers-1]
        offsets = self.headers['offset'].values
        layers = np.sum((crossovers - offsets.reshape(-1, 1)) < 0, axis=1)
        self.headers['layer'] = layers

    def _get_cols(self, name):
        cols = DEFAULT_COLS.get(name)
        if cols is None:
            raise ValueError(f"Given unknown 'name' {name}")
        return cols

    def _update_params(self, name, coords, values, columns):
        coord_names = self._get_cols(name)
        columns = to_list(columns)
        data = np.hstack((coords, values))
        df = pd.DataFrame(data, columns=[*coord_names, *columns]).set_index(coord_names)

        # If column from `columns` already exists in df params, it will be overwritten
        updated_params = getattr(self, f"{name}_params").merge(df, how='outer', on=coord_names, suffixes=("_drop", ""))
        updated_params = updated_params.drop(columns=updated_params.filter(regex="_drop$").columns)
        setattr(self, f"{name}_params", updated_params)

    def optimize(self, n_iters=1, max_wv=None, depth_tol=1e-4, vel_tol=1e-8):
        """!!!"""
        self.update_depth(layer=1, tol=depth_tol)
        for _ in tqdm(range(1, n_iters+1)):
            self.update_velocity(max_wv=max_wv, tol=vel_tol)
            self.update_depth(layer=1, tol=depth_tol)

    def update_depth(self, layer, n_iters=3000, tol=1e-4):
        layer_headers = self._fill_layer_params(headers=self.headers, layer=layer)
        ohe_source, unique_sources, ix_sources = self._get_sparse_depths("source", layer_headers, layer)
        ohe_rec, unique_recs, ix_recs = self._get_sparse_depths("rec", layer_headers, layer)
        matrix = sparse.hstack((ohe_source, ohe_rec))

        if 'depth_1' in self.source_params.columns:
            coefs = layer_headers.iloc[ix_sources]['depth_1_source'].tolist() + layer_headers.iloc[ix_recs]['depth_1_rec'].tolist()
        else:
            coefs = np.zeros(matrix.shape[1])
        # Add param for n_iters to stop
        prev_iters = [np.inf for _ in range(10)]
        prev_coefs = coefs
        for _ in tqdm(range(n_iters)):
            coefs = sparse.linalg.lsmr(matrix, layer_headers['y'], x0=np.array(coefs), maxiter=1)[0]
            coefs = np.clip(coefs, 0, None)
            prev_iters.append(np.mean(coefs - prev_coefs))
            prev_iters.pop(0)
            if np.mean(prev_iters) < tol:
                break
            prev_coefs = coefs

        self._update_params("source", unique_sources, coefs[:len(unique_sources)].reshape(-1, 1), 'depth_1')
        self._update_params("rec", unique_recs, coefs[len(unique_sources):].reshape(-1, 1), 'depth_1')

    def _fill_layer_params(self, headers, layer):
        headers = headers[headers["layer"] == layer]
        headers = headers.merge(self.source_params, on=self._get_cols("source"))
        headers = headers.merge(self.rec_params, on=self._get_cols("rec"), suffixes=("_source", "_rec"))

        headers[f'v{layer+1}_avg'] = (headers[f'v{layer+1}_source'] + headers[f'v{layer+1}_rec']) / 2
        offsets = headers["offset"]
        headers['y'] = headers[self.first_breaks_col] - offsets / headers[f'v{layer+1}_avg']
        # Drop traces with y < 0 since this cannot happend in real world
        headers = headers[headers['y'] > 0]
        return headers

    def _get_sparse_depths(self, name, headers, layer):
        coord_names = self._get_cols(name)
        uniques, index, inverse = np.unique(headers[coord_names], axis=0, return_index=True, return_inverse=True)
        wv_params = headers[[f'v{layer}_{name}', f'v{layer+1}_{name}', f'v{layer+1}_avg']].values.T
        coefs = calculate_depth_coefs(*wv_params)
        eye = sparse.eye((len(uniques)), format='csc')[inverse]
        matrix = eye.multiply(coefs.reshape(-1, 1)).tocsc()
        return matrix, uniques, index

    def update_velocity(self, max_wv=None, tol=1e-8, approach='rough'):
        layer_headers = self._fill_layer_params(headers=self.headers, layer=1)
        ohe_source, unique_sources, ix_sources = self._get_sparse_velocities("source", layer_headers, 1)
        ohe_rec, unique_recs, ix_recs = self._get_sparse_velocities("rec", layer_headers, 1)
        matrix = sparse.hstack((ohe_source, ohe_rec))

        coefs = layer_headers.iloc[ix_sources]['v1_source'].tolist() + layer_headers.iloc[ix_recs]['v1_rec'].tolist()
        result = sparse.linalg.lsmr(matrix, layer_headers['y'], x0=np.array(coefs), atol=tol, btol=tol)[0]

        velocities = self._calculate_velocities(result, layer_headers, ix_sources, ix_recs, max_wv, approach)
        self._update_params("source", unique_sources, velocities[0], 'v1')
        self._update_params("rec", unique_recs, velocities[1], 'v1')

    def _get_sparse_velocities(self, name, headers, layer):
        coord_names = self._get_cols(name)
        uniques, index, inverse = np.unique(headers[coord_names], axis=0, return_index=True, return_inverse=True)
        coefs = headers.iloc[index][f'depth_{layer}_{name}'].values
        eye = sparse.eye((len(uniques)), format='csc')
        matrix = eye.multiply(coefs).tocsc()[inverse]
        return matrix, uniques, index

    def _calculate_velocities(self, result, headers, ix_sources, ix_recs, max_wv, approach):
        max_wv = np.min(headers[["v2_source", "v2_rec"]].min()) if max_wv is None else max_wv
        v2 = headers.iloc[ix_sources]["v2_source"].tolist() + headers.iloc[ix_recs]["v2_rec"].tolist()
        # What velocity to choose x1 or x2? min?
        if approach == 'rough':
            x1 = calculate_wv_by_v2(np.array(v2), result, max_wv)
        elif approach == 'full':
            avg_v2 = headers.iloc[ix_sources]["v2_avg"].tolist() + headers.iloc[ix_recs]["v2_avg"].tolist()
            x1 = calculate_velocities(np.array(v2), np.array(avg_v2), result, max_wv)
        else:
            raise ValueError('!!!')
        sources = x1[: len(ix_sources)].reshape(-1, 1)
        recs = x1[len(ix_sources): ].reshape(-1, 1)
        return sources, recs

    ### dump ###
    # Raw dumps, just to be able to somehow save results
    def dump(self, name, path, layer, fillna=0):
        columns = self._get_cols(name)
        if name == 'source':
            columns = columns + ["EnergySourcePoint", "SourceWaterDepth", "GroupWaterDepth"]
        elif name == 'rec':
            columns = columns + ["ReceiverDatumElevation", "SourceDatumElevation", "ReceiverGroupElevation"]
        else:
            raise ValueError('!!!')
        depths = getattr(self, f"{name}_params")[[f"depth_{layer}"]].fillna(fillna).round().astype(np.int32)
        sub_headers = self.survey.headers[columns].reset_index(drop=True)
        sub_headers = sub_headers.set_index(columns[:2]).drop_duplicates()
        dump_df = depths.join(sub_headers).sort_values(by=columns[2: 4])
        self._dump(path, dump_df, columns[2:] + [f"depth_{layer}"])

    def _dump(self, path, df, columns):
        with open(path, 'w', encoding="UTF-8") as f:
            for _, row in df.iterrows():
                nums = "{:8}" * len(columns)
                line = (nums + "\n").format(*row[columns].values)
                f.write(line)

    ### plotters ###

    def plot_depths(self, layer):
        _, ax = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
        mm_source = MetricMap(self.source_params.index, self.source_params[f'depth_{layer}'])
        mm_source.plot(title='sources depth', ax=ax[0])
        mm_rec = MetricMap(self.rec_params.index, self.rec_params[f'depth_{layer}'])
        mm_rec.plot(title='receivers depth', ax=ax[1])

    def plot_applied_static_map(self, column, datum, **kwargs):
        survey = self.survey.copy()
        survey.headers = survey.headers.merge(self.headers, on=USED_COLS)
        survey.reindex(['GroupX', 'GroupY'], inplace=True)
        def _gather_plot(fontsize, coords, ax):
            g = survey.get_gather(coords)
            g.plot(ax=ax, title='before')

        def _plot_statics(fontsize, coords, ax):
            g = survey.get_gather(coords)
            g = g.apply_static_correciton(datum=datum)
            g.plot(ax=ax, title='after')

        mmap = MetricMap(survey.headers.index, survey.headers[column].values)
        mmap.plot(interactive=True, plot_on_click=(_gather_plot, _plot_statics), **kwargs)
