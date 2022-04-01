import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse

from .utils import calculate_depth_coef
from seismicpro.src.metrics import MetricMap


DEFAULT_COLS = {
    'source': ['SourceX', 'SourceY'],
    'rec': ['GroupX', 'GroupY']
}


USEFUL_COLS = [*sum(DEFAULT_COLS.values(), []), "SourceUpholeTime", "CDP_X", "CDP_Y", "offset"]


class StaticCorrection:
    def __init__(self, survey, first_breaks_col, interpolator):
        self.survey = survey.copy()
        self.first_breaks_col = first_breaks_col

        self.headers = self.survey.headers[USEFUL_COLS + [first_breaks_col]]
        self.headers_index = self.headers.index.names
        self.headers.reset_index(inplace=True)

        self.interpolator = interpolator
        self.n_layers = None

        self.source_params = self._create_depth_df('source')
        self.rec_params = self._create_depth_df('rec')

    ### main actions ###

    def fill_traces_params(self):
        # TODO: stop if velocities already added into headers
        self._add_wv_to_params('source')
        self._add_wv_to_params('rec')
        self.headers = self.headers.merge(self.source_params, on=DEFAULT_COLS['source']).merge(self.rec_params, on=DEFAULT_COLS['rec'])

    def calculate_thicknesses(self):
        # TODO: add accumulation of loss and plot for it.
        for i in range(1, self.n_layers):
            cross_left = self.headers.get([f"source_x{i}", f"rec_x{i}"], 0)
            cross_left = cross_left.min(axis=1) if isinstance(cross_left, type(self.headers)) else cross_left
            cross_right = self.headers.get([f"source_x{i+1}", f"rec_x{i+1}"], np.inf)
            cross_right = cross_right.max(axis=1) if isinstance(cross_right, type(self.headers)) else cross_right
            # Sometimes some sources or receivers haven't got any traces on this range, thus they won't have i-th depth
            layer_headers = self.headers[(self.headers["offset"] >= cross_left) & (self.headers['offset'] <= cross_right)].copy()

            layer_headers[f'avg_v{i+1}'] = (layer_headers[f'source_v{i+1}'] + layer_headers[f'rec_v{i+1}']) / 2
            layer_headers['y'] = layer_headers[self.first_breaks_col] - layer_headers['offset'] / layer_headers[f'avg_v{i+1}']
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

    ### supporting function ###
    def _create_depth_df(self, name):
        cols = self._get_cols(name)
        unique_recs =np.unique(self.headers[cols], axis=0)
        depths = pd.DataFrame(unique_recs, columns=cols)
        return depths.set_index(cols)

    def _get_cols(self, name):
        cols = DEFAULT_COLS.get(name)
        if cols is None:
            raise ValueError(f'Given unknown "name" {name}')
        return cols

    def _add_wv_to_params(self, name):
        cols = self._get_cols(name)
        uniques = np.unique(self.headers[cols], axis=0)
        wv_params = self.interpolator(uniques)
        self.n_layers = wv_params.shape[1] // 2 + 1
        names = [f'{name}_x{i+1}' for i in range(self.n_layers-1)] + [f'{name}_v{i+1}' for i in range(self.n_layers)]
        self._update_params(name, uniques, wv_params, names)

    def _update_params(self, name, coords, values, columns):
        cols = self._get_cols(name)
        data = np.hstack((coords, values))
        df = pd.DataFrame(data, columns=[*cols, *columns]).set_index(cols)
        setattr(self, f'{name}_params', getattr(self, f'{name}_params').join(df, on=cols))

    def _get_sparse_coefs(self, name, layer_headers, layer):
        cols = self._get_cols(name)
        uniques, index, inverse = np.unique(layer_headers[cols], axis=0, return_index=True, return_inverse=True)
        coefs = calculate_depth_coef(*layer_headers.iloc[index][[name+f'_v{layer}', name+f'_v{layer+1}', f'avg_v{layer+1}']].values.T)
        eye = sparse.eye((len(uniques)), format='csc')
        return eye.multiply(coefs).tocsc()[inverse], uniques

    ### plotters ###

    def plot_depths(self, layer):
        _, ax = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
        mm_source = MetricMap(self.source_params.index, self.source_params[f'depth_{layer}'])
        mm_source.plot(title='sources depth', ax=ax[0])
        mm_rec = MetricMap(self.rec_params.index, self.rec_params[f'depth_{layer}'])
        mm_rec.plot(title='receivers depth', ax=ax[1])

    def plot_applied_static_map(self, column, datum, **kwargs):
        survey = self.survey.copy()
        survey.headers = survey.headers.merge(self.headers, on=USEFUL_COLS)
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
