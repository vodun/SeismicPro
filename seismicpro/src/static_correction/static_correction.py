from email import header
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse

from .utils import calculate_depth_coefs, calculate_velocities
from seismicpro.src.metrics import MetricMap


DEFAULT_COLS = {
    "source": ["SourceX", "SourceY"],
    "rec": ["GroupX", "GroupY"]
}


USED_COLS = [*sum(DEFAULT_COLS.values(), []), "offset"]


class StaticCorrection:
    def __init__(self, survey, first_breaks_col, interpolator):
        self.survey = survey.copy()
        self.first_breaks_col = first_breaks_col

        headers = self.survey.headers.reset_index()
        self.headers = headers[USED_COLS + [first_breaks_col]]

        self.source_params = self._create_params_df(name="source")
        self.rec_params = self._create_params_df(name="rec")

        self._add_wv_to_params("source", interpolator=interpolator)
        self._add_wv_to_params("rec", interpolator=interpolator)

        self.n_layers = self.source_params.shape[1] // 2 + 1

    def _create_params_df(self, name):
        cols = self._get_cols(name)
        unique_coords = np.unique(self.headers[cols], axis=0)
        return pd.DataFrame(unique_coords, columns=cols).set_index(cols)

    def _get_cols(self, name):
        cols = DEFAULT_COLS.get(name)
        if cols is None:
            raise ValueError(f"Given unknown 'name' {name}")
        return cols

    def _add_wv_to_params(self, name, interpolator):
        cols = self._get_cols(name)
        unique_coords = np.unique(self.headers[cols], axis=0)
        wv_params = interpolator(unique_coords)
        length = wv_params.shape[1] // 2 + 1
        names = [f"{name}_x{i+1}" for i in range(length-1)] + [f"{name}_v{i+1}" for i in range(length)]
        self._update_params(name=name, coords=unique_coords, values=wv_params, columns=names)

    def _update_params(self, name, coords, values, columns):
        cols = self._get_cols(name)
        columns = [columns] if isinstance(columns, str) else columns

        data = np.hstack((coords, values))
        df = pd.DataFrame(data, columns=[*cols, *columns]).set_index(cols)

        # If column from `columns` already exists in df params, it will be overwritten
        updated_params = getattr(self, f"{name}_params").merge(df, how='outer', on=cols, suffixes=("_drop", ""))
        updated_params = updated_params.drop(columns=updated_params.filter(regex="_drop$").columns)
        setattr(self, f"{name}_params", updated_params)

    def optimize(self, n_iters=1, max_wv=None):
        """!!!"""
        self.update_layer_params(layer=1, to="depth")
        for layer in range(1, n_iters+1):
            self.update_layer_params(layer=1, to="vel", max_wv=max_wv)
            self.update_layer_params(layer=layer, to="depth")

    def update_layer_params(self, layer, to, max_wv=None):
        """!!!"""
        layer_headers = self._get_layer_headers(layer=layer)
        layer_headers = self._fill_layer_params(headers=layer_headers, layer=layer)

        ohe_source, unique_sources, ix_sources = self._get_sparse_coefs("source", layer_headers, layer, to=to)
        ohe_rec, unique_recs, ix_recs = self._get_sparse_coefs("rec", layer_headers, layer, to=to)
        matrix = sparse.hstack((ohe_source, ohe_rec))
        # Add opp to print info about results
        parameters = sparse.linalg.lsqr(matrix, layer_headers['y'])
        coefs = parameters[0]

        if to == "depth":
            results = self._get_depths(coefs, unique_sources)
            names = [f"depth_{layer}", f"depth_{layer}"]
        elif to == "vel":
            results = self._calculate_velocities(coefs, layer_headers, ix_sources, ix_recs, max_wv=max_wv)
            names = ["source_v1", "rec_v1"]
        self._update_params("source", unique_sources, results[0], names[0])
        self._update_params("rec", unique_recs, results[1], names[1])

    def _get_layer_headers(self, layer):
        # Add velocity and crossover params for sources and recs
        headers = self.headers.merge(self.source_params, on=self._get_cols("source"))
        headers = headers.merge(self.rec_params, on=self._get_cols("rec"))
        # Get traces for current layer by crossover offset
        cross_left = headers.get([f"source_x{layer}", f"rec_x{layer}"]).min(axis=1)
        cross_right = np.inf
        if f"source_x{layer+1}" in headers.columns:
            cross_right = headers.get([f"source_x{layer+1}", f"rec_x{layer+1}"]).max(axis=1)
        return headers[(headers["offset"] >= cross_left) & (headers["offset"] <= cross_right)].copy()

    def _fill_layer_params(self, headers, layer):
        headers[f'avg_v{layer+1}'] = (headers[f'source_v{layer+1}'] + headers[f'rec_v{layer+1}']) / 2
        offsets = headers["offset"]
        headers['y'] = headers[self.first_breaks_col] - offsets / headers[f'avg_v{layer+1}']
        # Drop traces with y < 0 since this cannot happend in real world
        headers = headers[headers['y'] > 0]
        return headers

    def _get_sparse_coefs(self, name, headers, layer, to):
        cols = self._get_cols(name)
        uniques, index, inverse = np.unique(headers[cols], axis=0, return_index=True, return_inverse=True)
        if to == 'depth':
            wv_params = headers.iloc[index][[name+f'_v{layer}', name+f'_v{layer+1}', f'avg_v{layer+1}']].values.T
            coefs = calculate_depth_coefs(*wv_params)
        elif to == 'vel':
            coefs = getattr(self, f"{name}_params").loc[list(map(tuple, uniques))]["depth_1"].values
        else:
            raise ValueError('!!!')

        eye = sparse.eye((len(uniques)), format='csc')
        return eye.multiply(coefs).tocsc()[inverse], uniques, index

    def _get_depths(self, results, unique_sources):
        sources_depth = results[:len(unique_sources)].reshape(-1, 1)
        recs_depth = results[len(unique_sources):].reshape(-1, 1)
        return [sources_depth, recs_depth]

    def _calculate_velocities(self, result, headers, ix_sources, ix_recs, max_wv=None):
        max_wv = np.min(headers[["source_v2", "rec_v2"]].min()) if max_wv is None else max_wv
        v2 = headers.iloc[ix_sources]["source_v2"].tolist() + headers.iloc[ix_recs]["rec_v2"].tolist()
        avg_v2 = headers.iloc[ix_sources]["avg_v2"].tolist() + headers.iloc[ix_recs]["avg_v2"].tolist()
        # What velocity to choose x1 or x2? min?
        x1, _ = calculate_velocities(np.array(v2), np.array(avg_v2), result, max_wv)
        sources = x1[: len(ix_sources)].reshape(-1, 1)
        recs = x1[len(ix_sources): ].reshape(-1, 1)
        return [sources, recs]


    ### dump ###
    # Raw dumps, just to be able to somehow save results
    def dump(self, name, path, layer, fillna=0):
        columns = self._get_cols(name)
        if name == 'source':
            columns.extend(["EnergySourcePoint", "SourceWaterDepth", "GroupWaterDepth"])
        elif name == 'rec':
            columns.extend(["ReceiverDatumElevation", "SourceDatumElevation", "ReceiverGroupElevation"])
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
