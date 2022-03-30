import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import sparse

from seismicpro.src.metrics import MetricMap


DEFAULT_COLS = {
    'source': ['SourceX', 'SourceY'],
    'rec': ['GroupX', 'GroupY']
}


class StaticCorrection:
    def __init__(self, survey, coords_to_params, first_breaks_col="FirstBreakShifted"):
        self.survey = survey.copy()
        self.headers = self.survey.headers
        self.headers_index = self.headers.index.names
        self.headers.reset_index(inplace=True)

        self.first_breaks_col = first_breaks_col
        self.get_wv_params(coords_to_params=coords_to_params)

    def get_wv_params(self, coords_to_params):
        wv_params = np.array(list(coords_to_params.values()))
        nn = NearestNeighbors(n_neighbors=1).fit(list(coords_to_params.keys()))

        sources_df = self._get_params('source', wv_params, nn)
        recs_df = self._get_params('rec', wv_params, nn)
        self.headers = self.headers.merge(sources_df, on=DEFAULT_COLS['source']).merge(recs_df, on=DEFAULT_COLS['rec'])

    def _get_params(self, name, wv_params, nn):
        cols = DEFAULT_COLS.get(name)
        if cols is None:
            raise ValueError(f'Given unknown "name" {name}')
        uniques = np.unique(self.headers[cols], axis=0)
        ix = nn.kneighbors(uniques, return_distance=False).reshape(-1)
        df = pd.DataFrame(np.hstack((uniques, wv_params[ix])),
                          columns=[*cols, f'{name}_v1', f'{name}_v2', f'{name}_x1'])
        return df

    def calculate_thicknesses(self, debug=False):
        crossovers = self.headers[['source_x1', 'rec_x1']].max(axis=1)
        self.headers = self.headers[self.headers['offset'] >= crossovers].copy()

        self.headers['avg_v2'] = (self.headers['source_v2'] + self.headers['rec_v2']) / 2
        self.headers['y'] = self.headers[self.first_breaks_col] - self.headers['offset'] / self.headers['avg_v2']
        # Drop traces with y < 0 since this cannot happend in real world
        self.headers = self.headers[self.headers['y'] > 0]

        # Construct ohe matrics for pairs of sources and receivers
        ohe_shot = self.get_spase_coefs(name='source')
        ohe_rec = self.get_spase_coefs(name='rec')
        matrix = sparse.hstack((ohe_shot, ohe_rec))

        right_vector = self.headers['y']
        lsqr_res = sparse.linalg.lsqr(matrix, right_vector)
        res = lsqr_res[0]

        sources = np.unique(self.headers[DEFAULT_COLS['source']], axis=0)
        recs = np.unique(self.headers[DEFAULT_COLS['rec']], axis=0)
        sources_depths = res[: len(sources)]
        recs_depths = res[len(sources):]

        dfs = pd.DataFrame(np.hstack((sources, sources_depths.reshape(-1, 1))), columns=['SourceX', 'SourceY', 'source_depth_1'])
        dfr = pd.DataFrame(np.hstack((recs, recs_depths.reshape(-1, 1))), columns=['GroupX', 'GroupY', 'rec_depth_1'])
        headers = self.headers.merge(dfs, on=DEFAULT_COLS['source']).merge(dfr, on=DEFAULT_COLS['rec'])

        if debug:
            print(f"Reason {lsqr_res[1]}, Iter {lsqr_res[2]}, MAE {lsqr_res[3]}")
            estimation = matrix.dot(res)
            error = np.abs((np.array(right_vector) - estimation) / np.array(right_vector))
            df = pd.DataFrame(np.hstack((self.headers[DEFAULT_COLS['source']], self.headers[DEFAULT_COLS['rec']],
                              np.array(error).reshape(-1, 1), estimation.reshape(-1, 1))),
                              columns=[*sum(DEFAULT_COLS.values(), []), 'static_error', 'pred'])
            headers = headers.merge(df)
        self.headers = headers.set_index(self.headers_index)
        self.survey.headers = self.headers

    def get_spase_coefs(self, name):
        cols = DEFAULT_COLS.get(name)
        if cols is None:
            raise ValueError(f'Given unknown "name" {name}')
        uniques, index, inverse = np.unique(self.headers[cols], axis=0, return_index=True, return_inverse=True)
        coefs = self._calculate_depth_coef(*self.headers.iloc[index][[name+'_v1', name+'_v2', 'avg_v2']].values.T)
        eye = sparse.eye((len(uniques)), format='csc')
        return eye.multiply(coefs).tocsc()[inverse]

    @staticmethod
    def _calculate_depth_coef(v1, v2, avg_v2):
        return (avg_v2 * v2 - v1**2) / (v1*avg_v2*(v2**2 - v1**2)**.5)

    def plot_depths(self):
        if 'source_depth_1' not in self.headers.columns or 'rec_depth_1' not in self.headers.columns:
            raise ValueError('Depths are not calculated yet, call `calculate_thicknesses` before.')
        _, ax = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
        sources_df = self.headers[[*DEFAULT_COLS['source'], 'source_depth_1']].drop_duplicates().values
        recs_df = self.headers[[*DEFAULT_COLS['rec'], 'rec_depth_1']].drop_duplicates().values
        mm_rec = MetricMap(sources_df[:, :2], sources_df[:, -1])
        mm_rec.plot(title='sources depth', ax=ax[0])
        mm_source = MetricMap(recs_df[:, :2], recs_df[:, -1])
        mm_source.plot(title='receivers depth', ax=ax[1])

    def plot_applied_static_map(self, column, datum, sp_params, **kwargs):
        survey = self.survey
        survey = survey.generate_supergathers(**sp_params)
        def _gather_plot(fontsize, coords, ax):
            g = survey.get_gather(coords)
            g.sort('offset').plot(ax=ax, title='before')

        def _plot_statics(fontsize, coords, ax):
            g = survey.get_gather(coords)
            g = g.apply_static_correciton(datum=datum)
            g.sort('offset').plot(ax=ax, title='after')

        mmap = MetricMap(survey.headers.index, survey.headers[column].values)
        mmap.plot(interactive=True, plot_on_click=(_gather_plot, _plot_statics), **kwargs)
