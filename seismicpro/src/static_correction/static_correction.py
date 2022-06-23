import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.stats import hmean

from .utils import calculate_depth_coefs, calculate_wv_by_v2, calculate_velocities
from .interactive_slice_plot import StaticsPlot
from seismicpro.src.utils import to_list, IDWInterpolator
from seismicpro.src.metrics import MetricMap


DEFAULT_COLS = {
    "source": ["SourceX", "SourceY"],
    "rec": ["GroupX", "GroupY"]
}


USED_COLS = [*sum(DEFAULT_COLS.values(), []), "offset", "SourceDepth", "CDP_X", "CDP_Y", "SourceSurfaceElevation",
             "ReceiverGroupElevation"]


class StaticCorrection:
    def __init__(self, survey, first_breaks_col, interpolator, n_avg_coords=5, interp_neighbors=100):
        self.survey = survey.copy()
        self.first_breaks_col = first_breaks_col
        self.n_layers = None
        self.interp_neighbors = interp_neighbors

        headers_cols = USED_COLS + [first_breaks_col]
        self.headers = pd.DataFrame(self.survey[headers_cols], columns=headers_cols)

        self.source_params = self._create_params_df(name="source")
        self.source_headers = []
        self.rec_params = self._create_params_df(name="rec")
        self.rec_headers = []

        self._add_cols_to_params("source", columns="SourceDepth")
        self._add_wv_to_params("source", interpolator=interpolator)
        self._add_wv_to_params("rec", interpolator=interpolator)

        self._set_traces_layer(interpolator=interpolator)

        self.interp_elevations = self._construct_elevations_interpolatior()
        self.interp_layers_els = [lambda x: np.inf for _ in range(self.n_layers)]
        self.interp_v1 = None

        xs = np.linspace(self.headers['SourceX'], self.headers['GroupX'], n_avg_coords, dtype=np.int32).T.reshape(-1)
        ys = np.linspace(self.headers['SourceY'], self.headers['GroupY'], n_avg_coords, dtype=np.int32).T.reshape(-1)
        subw_velocities = interpolator(np.stack((xs, ys)).T)[:, self.n_layers+1:]
        for i in range(self.n_layers):
            self.headers[f'v{i+2}_avg'] = hmean(subw_velocities[:, i].reshape(-1, n_avg_coords), axis=1)

    def _create_params_df(self, name):
        coord_names = self._get_cols(name)
        unique_coords = np.unique(self.headers[coord_names], axis=0).astype(np.int32)
        return pd.DataFrame(unique_coords, columns=coord_names).set_index(coord_names)

    def _add_cols_to_params(self, name, columns):
        coord_names = self._get_cols(name)
        data = self.headers[coord_names + to_list(columns)].drop_duplicates().values
        if data.shape[0] != getattr(self, f"{name}_params").shape[0]:
            raise ValueError("Value in column(s) to add must be unique for each source/rec.")
        getattr(self, f"{name}_headers").extend(to_list(columns))
        self._update_params(name=name, coords=data[:, :2], values=data[:, 2:], columns=columns)

    def _add_wv_to_params(self, name, interpolator):
        unique_coords = getattr(self, f"{name}_params").index.to_frame().values
        wv_params = interpolator(unique_coords)
        self.n_layers = wv_params.shape[1] // 2

        names = [f"v{i}" for i in range(1, self.n_layers+2)]
        self._update_params(name=name, coords=unique_coords, values=wv_params[:, self.n_layers:], columns=names)

    def _set_traces_layer(self, interpolator):
        coords = self.headers[["CDP_X", "CDP_Y"]].values
        crossovers = interpolator(coords)[:, :self.n_layers]
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

    def _construct_elevations_interpolatior(self):
        headers = self.survey.headers.reset_index()
        sources_el = headers[["SourceX", "SourceY", "SourceSurfaceElevation"]].drop_duplicates()
        sources_el = sources_el.set_index(["SourceX", "SourceY"])[~sources_el.index.duplicated(keep=False)]
        rec_el = headers[["GroupX", "GroupY", "ReceiverGroupElevation"]].drop_duplicates()
        rec_el = rec_el.set_index(["GroupX", "GroupY"])

        source_el_coords = np.array(sources_el.index.to_frame().values)
        rec_el_coords = np.array(rec_el.index.to_frame().values)

        coords = np.concatenate((source_el_coords, rec_el_coords))
        elevations = np.concatenate((sources_el["SourceSurfaceElevation"].values,
                                     rec_el["ReceiverGroupElevation"].values))

        return IDWInterpolator(coords, elevations, neighbors=self.interp_neighbors)

    def optimize(self, depth_tol=1e-7, smoothing_radius=None):
        """!!!"""
        self.update_depth(layer=1, tol=depth_tol, smoothing_radius=smoothing_radius)

    def update_depth(self, layer, tol=1e-7, smoothing_radius=None):
        interp_kwargs = {}
        layer_headers = self._fill_layer_params(headers=self.headers, layer=layer)
        ohe_source, unique_sources, ix_sources = self._get_sparse_depths("source", layer_headers, layer)
        ohe_rec, unique_recs, ix_recs = self._get_sparse_depths("rec", layer_headers, layer)
        matrix = sparse.hstack((ohe_source, ohe_rec))

        if f'depth_{layer}' in self.source_params.columns:
            coefs = (layer_headers.iloc[ix_sources][f'depth_{layer}_source'].tolist()
                     + layer_headers.iloc[ix_recs][f'depth_{layer}_rec'].tolist())
        else:
            coefs = np.zeros(matrix.shape[1])
        coefs = sparse.linalg.lsqr(matrix, layer_headers['y'], atol=tol, btol=tol, x0=coefs)[0]


        upholes = layer_headers.iloc[ix_sources]["SourceDepth"].values
        source_els = self.interp_elevations(unique_sources)
        rec_els = self.interp_elevations(unique_recs)

        # Distance from 0 elevation to current sub layer.
        source_elevations = source_els - (coefs[:len(unique_sources)] + upholes)
        rec_elevations = rec_els - coefs[len(unique_sources):]

        interp_source = IDWInterpolator(unique_sources, source_elevations, neighbors=self.interp_neighbors)
        interp_rec = IDWInterpolator(unique_recs, rec_elevations, neighbors=self.interp_neighbors)

        source_elevations = (source_elevations + interp_rec(unique_sources)) / 2
        rec_elevations = (rec_elevations + interp_source(unique_recs)) / 2

        coords = np.concatenate([unique_sources, unique_recs])
        elevations = np.concatenate([source_elevations, rec_elevations])

        if smoothing_radius is not None:
            interp_kwargs.update({"radius": smoothing_radius, "dist_transform": 0})

        joint_interp = IDWInterpolator(coords, elevations, **interp_kwargs)
        joint_interp = IDWInterpolator(coords, joint_interp(coords), neighbors=self.interp_neighbors)
        self.interp_layers_els[layer-1] = joint_interp

        source_depths = source_els - joint_interp(unique_sources) - upholes
        rec_depths = rec_els - joint_interp(unique_recs)

        self._update_params("source", unique_sources, source_depths.reshape(-1, 1), f"depth_{layer}")
        self._update_params("rec", unique_recs, rec_depths.reshape(-1, 1), f"depth_{layer}")

    def _fill_layer_params(self, headers, layer):
        headers = headers[headers["layer"] == layer]
        headers = headers.merge(self.source_params, on=self._get_cols("source") + self.source_headers)
        headers = headers.merge(self.rec_params, on=self._get_cols("rec") + self.rec_headers,
                                suffixes=("_source", "_rec"))

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

    def update_velocity(self, max_wv=None, tol=1e-8, smoothing_radius=None):
        interp_kwargs = {}
        layer_headers = self._fill_layer_params(headers=self.headers, layer=1)
        ohe_source, unique_sources, ix_sources = self._get_sparse_velocities("source", layer_headers, 1)
        ohe_rec, unique_recs, ix_recs = self._get_sparse_velocities("rec", layer_headers, 1)
        matrix = sparse.hstack((ohe_source, ohe_rec))

        coefs = layer_headers.iloc[ix_sources]['v1_source'].tolist() + layer_headers.iloc[ix_recs]['v1_rec'].tolist()
        result = sparse.linalg.lsmr(matrix, layer_headers['y'], x0=np.array(coefs), atol=tol, btol=tol)[0]

        source_v1, rec_v1 = self._calculate_velocities(result, layer_headers, ix_sources, ix_recs, max_wv)

        interp_source_v1 = IDWInterpolator(unique_sources, source_v1, neighbors=self.interp_neighbors)
        interp_rec_v1 = IDWInterpolator(unique_recs, rec_v1, neighbors=self.interp_neighbors)

        mean_source_v1 = (source_v1 + interp_rec_v1(unique_sources)) / 2
        mean_rec_v1 = (rec_v1 + interp_source_v1(unique_recs)) / 2

        coords = np.concatenate([unique_sources, unique_recs])
        v1 = np.concatenate([mean_source_v1, mean_rec_v1])

        if smoothing_radius is not None:
            interp_kwargs.update({"radius": smoothing_radius, "dist_transform": 0})

        joint_interp = IDWInterpolator(coords, v1, **interp_kwargs)
        joint_interp = IDWInterpolator(coords, joint_interp(coords), neighbors=self.interp_neighbors)
        self.interp_v1 = joint_interp

        final_source_v1 = joint_interp(unique_sources)
        final_rec_v1 = joint_interp(unique_recs)

        self._update_params("source", unique_sources, final_source_v1, 'v1')
        self._update_params("rec", unique_recs, final_rec_v1, 'v1')

    def _get_sparse_velocities(self, name, headers, layer):
        coord_names = self._get_cols(name)
        uniques, index, inverse = np.unique(headers[coord_names], axis=0, return_index=True, return_inverse=True)
        coefs = headers.iloc[index][f'depth_{layer}_{name}'].values
        eye = sparse.eye((len(uniques)), format='csc')
        matrix = eye.multiply(coefs).tocsc()[inverse]
        return matrix, uniques, index

    def _calculate_velocities(self, result, headers, ix_sources, ix_recs, max_wv):
        max_wv = np.min(headers[["v2_source", "v2_rec"]].min()) if max_wv is None else max_wv
        v2 = headers.iloc[ix_sources]["v2_source"].tolist() + headers.iloc[ix_recs]["v2_rec"].tolist()
        # What velocity to choose x1 or x2? min?
        x1 = calculate_wv_by_v2(np.array(v2), result, max_wv)
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
    def construct_velicities_interpolatior(self, layer):
        vel_names = [f"v{i}" for i in [layer, min(layer+1, self.n_layers+1)]]
        sources = self.source_params.reset_index()[['SourceX', 'SourceY', *vel_names]].dropna().values
        recs = self.rec_params.reset_index()[['GroupX', 'GroupY', *vel_names]].dropna().values

        coords = np.concatenate((sources[:, :2], recs[:, :2])).astype(np.int32)
        velocities = np.concatenate((sources[:, 2:], recs[:, 2:]))
        vmin = np.min(velocities)
        vmax = np.max(velocities)
        interp_velocities = IDWInterpolator(coords, velocities)
        return interp_velocities, vmin, vmax


    def plot_slice(self, layer, n_points=100):
        interp_el = self._construct_elevations_interpolatior()
        interp_velocities, vmin, vmax = self.construct_velicities_interpolatior(layer=layer)
        sources = self.source_params.index.to_frame().values
        recs = self.rec_params.index.to_frame().values
        coords = np.unique(np.concatenate((sources, recs)), axis=0)

        obj = MetricMap(coords, self.interp_layers_els[layer-1](coords))
        StaticsPlot(obj, self.interp_layers_els[layer-1], interp_el, interp_velocities, n_points=n_points, vmin=vmin,
                    vmax=vmax).plot()

    def plot_depths(self, layer):
        _, ax = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
        source_cols = self._get_cols("source")
        source_depths = self.source_params[[f"depth_{layer}"]]
        uphole = self.headers[source_cols + ["SourceDepth"]].drop_duplicates()
        source_depths = source_depths.merge(uphole, on=source_cols)[[f"depth_{layer}", "SourceDepth"]].sum(axis=1)
        mm_source = MetricMap(self.source_params.index, source_depths)
        mm_source.plot(title="sources depth", ax=ax[0])
        mm_rec = MetricMap(self.rec_params.index, self.rec_params[f"depth_{layer}"])
        mm_rec.plot(title="receivers depth", ax=ax[1])

    def plot_attrs(self, name):
        _, ax = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
        mm_source = MetricMap(self.source_params.index, self.source_params[name])
        mm_source.plot(title=f"sources {name}", ax=ax[0])
        mm_rec = MetricMap(self.rec_params.index, self.rec_params[name])
        mm_rec.plot(title=f"receivers {name}", ax=ax[1])

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
