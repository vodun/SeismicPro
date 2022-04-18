import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse

from .utils import calculate_depth_coefs
from seismicpro.src.metrics import MetricMap


DEFAULT_COLS = {
    'source': ['SourceX', 'SourceY'],
    'rec': ['GroupX', 'GroupY']
}


USED_COLS = [*sum(DEFAULT_COLS.values(), []), "SourceUpholeTime", "CDP_X", "CDP_Y", "offset"]


class StaticCorrection:
    def __init__(self, survey, first_breaks_col, interpolator):
        self.survey = survey.copy()
        self.first_breaks_col = first_breaks_col

        self.headers = self.survey.headers[USED_COLS + [first_breaks_col]]
        self.headers.reset_index(inplace=True)

        self.source_params = self._create_params_df(name='source')
        self.rec_params = self._create_params_df(name='rec')

        self._add_wv_to_params('source', interpolator=interpolator)
        self._add_wv_to_params('rec', interpolator=interpolator)

        self.n_layers = self.source_params.shape[1] // 2 + 1

    def _create_params_df(self, name):
        cols = self._get_cols(name)
        unique_coords = np.unique(self.headers[cols], axis=0)
        return pd.DataFrame(unique_coords, columns=cols).set_index(cols)

    def _add_wv_to_params(self, name, interpolator):
        cols = self._get_cols(name)
        unique_coords = np.unique(self.headers[cols], axis=0)
        wv_params = interpolator(unique_coords)
        length = wv_params.shape[1] // 2 + 1
        names = [f'{name}_x{i+1}' for i in range(length-1)] + [f'{name}_v{i+1}' for i in range(length)]
        self._update_params(name=name, coords=unique_coords, values=wv_params, columns=names)
        self.headers = self.headers.merge(getattr(self, f"{name}_params"), on=cols)

    def _update_params(self, name, coords, values, columns):
        cols = self._get_cols(name)
        data = np.hstack((coords, values))
        df = pd.DataFrame(data, columns=[*cols, *columns]).set_index(cols)
        setattr(self, f'{name}_params', getattr(self, f'{name}_params').join(df, on=cols))

    def _get_cols(self, name):
        cols = DEFAULT_COLS.get(name)
        if cols is None:
            raise ValueError(f'Given unknown "name" {name}')
        return cols

    def calculate_thicknesses(self):
        # TODO:
        # 1. Add processing for self.n_layers = 1
        # 2. Add accumulation of loss and plot for it
        # 3. Add processing for sources / receivers with missing depths
        for i in range(1, self.n_layers):
            cross_left = self.headers.get([f"source_x{i}", f"rec_x{i}"]).min(axis=1)
            cross_right = np.inf
            if f"source_x{i+1}" in self.headers.columns:
                cross_right = self.headers.get([f"source_x{i+1}", f"rec_x{i+1}"]).max(axis=1)

            # Sometimes some sources or receivers haven't got any traces on this range, thus they won't have i-th depth
            offsets = self.headers['offset']
            layer_headers = self.headers[(offsets >= cross_left) & (offsets <= cross_right)].copy()

            layer_headers[f'avg_v{i+1}'] = (layer_headers[f'source_v{i+1}'] + layer_headers[f'rec_v{i+1}']) / 2
            layer_headers['y'] = layer_headers[self.first_breaks_col] - offsets / layer_headers[f'avg_v{i+1}']
            # Drop traces with y < 0 since this cannot happend in real world
            layer_headers = layer_headers[layer_headers['y'] > 0]

            ohe_source, unique_sources = self._get_sparse_coefs(name='source', layer_headers=layer_headers, layer=i)
            ohe_rec, unique_recs = self._get_sparse_coefs(name='rec', layer_headers=layer_headers, layer=i)
            matrix = sparse.hstack((ohe_source, ohe_rec))

            right_vector = layer_headers['y']
            lsqr_res = sparse.linalg.lsqr(matrix, right_vector)
            res = lsqr_res[0]

            self._update_params('source', unique_sources, res[:len(unique_sources)].reshape(-1, 1), [f'depth_{i}'])
            self._update_params('rec', unique_recs, res[len(unique_sources):].reshape(-1, 1), [f'depth_{i}'])

    def _get_sparse_coefs(self, name, layer_headers, layer):
        cols = self._get_cols(name)
        uniques, index, inverse = np.unique(layer_headers[cols], axis=0, return_index=True, return_inverse=True)
        wv_params = layer_headers.iloc[index][[name+f'_v{layer}', name+f'_v{layer+1}', f'avg_v{layer+1}']].values.T
        coefs = calculate_depth_coefs(*wv_params)
        eye = sparse.eye((len(uniques)), format='csc')
        return eye.multiply(coefs).tocsc()[inverse], uniques

    ### dump ###
    # Raw dumps, just to be able to somehow save results
    def dump_sources(self, path, layer, fillna=0):
        columns = ["SourceX", "SourceY", "EnergySourcePoint", "SourceWaterDepth", "GroupWaterDepth"]
        by = columns[2: 4]
        sources = self.source_params[[f"depth_{layer}"]].fillna(fillna).round().astype(np.int32)
        sub_headers = self.survey.headers[columns].reset_index(drop=True)
        sub_headers = sub_headers.set_index(columns[:2]).drop_duplicates()
        dump_df = sources.join(sub_headers).sort_values(by=by)
        self._dump(path, dump_df, columns[2:] + [f"depth_{layer}"])

    def dump_recs(self, path, layer, fillna=0):
        columns = ["GroupX", "GroupY", "ReceiverDatumElevation", "SourceDatumElevation", "ReceiverGroupElevation"]
        by = columns[2: 4]
        recs = self.rec_params[[f"depth_{layer}"]].fillna(fillna).round().astype(np.int32)
        sub_headers = self.survey.headers[columns].reset_index(drop=True)
        sub_headers = sub_headers.set_index(columns[:2]).drop_duplicates()
        dump_df = recs.join(sub_headers).sort_values(by=by)
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
