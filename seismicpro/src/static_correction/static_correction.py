import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.neighbors import NearestNeighbors
from scipy import sparse

from .utils import calculate_crossover, calculate_depth_coef
from seismicpro.src.metrics import MetricMap


DEFAULT_COLS = {
    'source': ['SourceX', 'SourceY'],
    'rec': ['GroupX', 'GroupY']
}


USEFUL_COLS = [*sum(DEFAULT_COLS.values(), []), "SourceUpholeTime", "CDP_X", "CDP_Y", "offset"]


class WeatheringVelocityInterpolator():
    def __init__(self, coords_to_params):
        self.wv_params = np.array(list(coords_to_params.values()))
        self.nn = NearestNeighbors(n_neighbors=1).fit(list(coords_to_params.keys()))

    def __call__(self, coords):
        ixs = self.nn.kneighbors(coords, return_distance=False).reshape(-1)
        return self.wv_params[ixs]


class StaticCorrection:
    def __init__(self, survey, first_breaks_col):
        self.survey = survey.copy()
        self.first_breaks_col = first_breaks_col

        self.headers = self.survey.headers[USEFUL_COLS + [first_breaks_col]]
        self.headers_index = self.headers.index.names
        self.headers.reset_index(inplace=True)

        self.interpolator = None
        self.n_layers = None

        self.source_depths = pd.DataFrame(np.unique(self.headers[DEFAULT_COLS['source']], axis=0),
                                          columns=DEFAULT_COLS['source'])
        self.rec_depths = pd.DataFrame(np.unique(self.headers[DEFAULT_COLS['rec']], axis=0),
                                       columns=DEFAULT_COLS['rec'])

    def _get_cols(self, name):
        cols = DEFAULT_COLS.get(name)
        if cols is None:
            raise ValueError(f'Given unknown "name" {name}')
        return cols

    def create_wv_interpolator(self, supergather_kwargs, weathering_velocity=None, wv_kwargs=None):
        # Move source on the surface, if wv is not passed to find correct value of V0.
        # All operations with passed weathering_velocity are applied under the assumption that weathering_velocity
        # is a direct wave.
        supergather_survey = self.survey.generate_supergathers(**supergather_kwargs)
        if weathering_velocity is None:
            supergather_survey.apply(lambda FirstBreak, SourceUpholeTime: np.clip(FirstBreak - SourceUpholeTime, 1, None),
                                     cols=[self.first_breaks_col, "SourceUpholeTime"],
                                     res_cols=self.first_breaks_col,
                                     inplace=True,
                                     unpack_args=True)

        coords_to_params = {}
        wv_kwargs = dict() if wv_kwargs is None else wv_kwargs
        wv_kwargs = {"n_layers": 2, "init": {"t0": 0}, **wv_kwargs}
        for ix in tqdm(supergather_survey.headers.index.unique()):
            g = supergather_survey.get_gather(ix)
            wv = g.calculate_weathering_velocity(first_breaks_col=self.first_breaks_col, **wv_kwargs)
            wv_params = [getattr(wv, param) for param in wv._valid_keys[1:]]
            if weathering_velocity is not None:
                wv_params.insert(0, calculate_crossover(weathering_velocity, 0, wv.v1, wv.t0))
                wv_params.insert(len(wv._valid_keys) - wv_kwargs.get("n_layers"), weathering_velocity)

            coords = self.survey.headers.loc[ix][["CDP_X", "CDP_Y"]].values[0]
            coords_to_params.update({tuple(coords): wv_params})

        self.interpolator = WeatheringVelocityInterpolator(coords_to_params)
        self.n_layers = wv_kwargs.get('n_layers') + (weathering_velocity is not None)

    def fill_traces_params(self):
        # TODO: stop if velocities already added into headers
        sources_df = self._get_params('source')
        recs_df = self._get_params('rec')
        self.headers = self.headers.merge(sources_df, on=DEFAULT_COLS['source']).merge(recs_df, on=DEFAULT_COLS['rec'])

    def _get_params(self, name):
        cols = self._get_cols(name)
        uniques = np.unique(self.headers[cols], axis=0)
        wv_params = self.interpolator(uniques)
        n_params = len(wv_params[0])
        names = [f'{name}_x{i+1}' for i in range(n_params // 2)] + [f'{name}_v{i+1}' for i in range(n_params // 2 + 1)]
        df = pd.DataFrame(np.hstack((uniques, self.interpolator(uniques))),
                          columns=[*cols, *names])
        return df

    def calculate_thicknesses(self):
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

            ohe_source, unique_sources = self.get_sparse_coefs(name='source', layer_headers=layer_headers, layer=i)
            ohe_rec, unique_recs = self.get_sparse_coefs(name='rec', layer_headers=layer_headers, layer=i)
            matrix = sparse.hstack((ohe_source, ohe_rec))

            right_vector = layer_headers['y']
            lsqr_res = sparse.linalg.lsqr(matrix, right_vector)
            res = lsqr_res[0]

            self._update_depths('source', unique_sources, res[:len(unique_sources)], i)
            self._update_depths('rec', unique_recs, res[len(unique_sources):], i)

        self.headers = self.headers.merge(self.source_depths, on=DEFAULT_COLS['source'])
        self.headers = self.headers.merge(self.rec_depths, on=DEFAULT_COLS['rec'])

    def get_sparse_coefs(self, name, layer_headers, layer):
        cols = self._get_cols(name)
        uniques, index, inverse = np.unique(layer_headers[cols], axis=0, return_index=True, return_inverse=True)
        coefs = calculate_depth_coef(*layer_headers.iloc[index][[name+f'_v{layer}', name+f'_v{layer+1}', f'avg_v{layer+1}']].values.T)
        eye = sparse.eye((len(uniques)), format='csc')
        return eye.multiply(coefs).tocsc()[inverse], uniques

    def _update_depths(self, name, uniques, depths, layer):
        cols = self._get_cols(name)
        stack = np.hstack((uniques, depths.reshape(-1, 1)))
        df = pd.DataFrame(stack, columns=[*cols, f'{name}_depth_{layer}'])
        setattr(self, f'{name}_depths', getattr(self, f'{name}_depths').merge(df, on=cols))

    def plot_depths(self, layer):
        if f'source_depth_{layer}' not in self.headers.columns or f'rec_depth_{layer}' not in self.headers.columns:
            raise ValueError('Depths are not calculated yet, call `calculate_thicknesses` before.')
        _, ax = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
        mm_source = MetricMap(self.source_depths[DEFAULT_COLS['source']], self.source_depths[f'source_depth_{layer}'])
        mm_source.plot(title='sources depth', ax=ax[0])
        mm_rec = MetricMap(self.rec_depths[DEFAULT_COLS['rec']], self.rec_depths[f'rec_depth_{layer}'])
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
