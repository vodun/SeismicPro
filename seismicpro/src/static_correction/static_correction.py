import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.stats import hmean

from .utils import calculate_depth_coefs, calculate_wv_by_v2, calculate_velocities, calculate_prev_layer_coefs
from .interactive_slice_plot import StaticsPlot
from seismicpro.src.utils import to_list, IDWInterpolator
from seismicpro.src.metrics import MetricMap


DEFAULT_COLS = {
    "source": ["SourceX", "SourceY"],
    "rec": ["GroupX", "GroupY"]
}

METRICS = {
    "mape" : lambda headers: np.abs(headers['y'] - headers['pred']) / headers['FirstBreak']
}


USED_COLS = [*sum(DEFAULT_COLS.values(), []), "offset", "SourceDepth", "CDP_X", "CDP_Y", "SourceSurfaceElevation",
             "ReceiverGroupElevation"]


class UniqueCoords:
    def __init__(self, coords):
        uniques, indices, inverse = np.unique(coords, axis=0, return_index=True, return_inverse=True)
        self.uniques = uniques
        self.indices = indices
        self.inverse = inverse


class StaticCorrection:
    def __init__(self, survey, first_breaks_col, interpolator, n_avg_coords=5, radius=500, n_neighbors=100):
        self.survey = survey.copy()
        self.first_breaks_col = first_breaks_col
        self.n_layers = None
        self.radius = radius
        self.n_neighbors = n_neighbors

        headers_cols = USED_COLS + [first_breaks_col]
        self.headers = pd.DataFrame(self.survey[headers_cols], columns=headers_cols)
        self.headers['y'] = np.nan

        self.source_params = self._create_params_df(name="source")
        self.source_headers = []
        self.rec_params = self._create_params_df(name="rec")
        self.rec_headers = []

        self._add_cols_to_params("source", columns="SourceDepth")
        self._add_wv_to_params("source", interpolator=interpolator)
        self._add_wv_to_params("rec", interpolator=interpolator)

        self._set_traces_layer(interpolator=interpolator)

        self.interp_elevations = self._construct_elevations_interpolatior()
        self.interp_layers_els = [None] * self.n_layers
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
        data = np.hstack((coords, values))
        df = pd.DataFrame(data, columns=[*coord_names, *to_list(columns)]).set_index(coord_names)

        # If column from `columns` already exists in df params, it will be overwritten
        updated_params = getattr(self, f"{name}_params").merge(df, how='outer', on=coord_names, suffixes=("_drop", ""))
        updated_params = updated_params.drop(columns=updated_params.filter(regex="_drop$").columns)
        setattr(self, f"{name}_params", updated_params)

    def _construct_elevations_interpolatior(self):
        headers = self.survey.headers.reset_index()
        sources_el = headers[["SourceX", "SourceY", "SourceSurfaceElevation"]].drop_duplicates()
        sources_el = sources_el.set_index(["SourceX", "SourceY"])[~sources_el.index.duplicated(keep=False)]
        rec_el = headers[["GroupX", "GroupY", "ReceiverGroupElevation"]].drop_duplicates()
        rec_el = rec_el.set_index(["GroupX", "GroupY"])[~rec_el.index.duplicated(keep=False)]

        coords = np.concatenate((sources_el.index.to_frame().values, rec_el.index.to_frame().values))
        elevations = np.concatenate((sources_el.values.ravel(), rec_el.values.ravel()))

        return IDWInterpolator(coords, elevations, radius=self.radius, neighbors=self.n_neighbors)

    def optimize(self, depth_tol=1e-7, smoothing_radius=None):
        """!!!"""
        self.update_depth(layer=1, tol=depth_tol, smoothing_radius=smoothing_radius)

    def _calculate_corr_coefs(self, name, layer_headers, ucoords, layer, upholes=None):
        corrs = np.zeros(len(layer_headers))
        elevations = self.interp_elevations(ucoords.uniques)[ucoords.inverse]
        for i in range(1, layer):
            depths = elevations - self.interp_layers_els[i-1](ucoords.uniques)[ucoords.inverse]
            if name == "source" and i==1:
                depths = depths - upholes
            # layer+1 is not a typo
            params = layer_headers[[f'v{i}_{name}', f'v{i+1}_{name}', f'v{layer+1}_{name}']].values.T
            corrs += depths * calculate_prev_layer_coefs(*params)
        return corrs

    def update_depth(self, layer, tol=1e-7, smoothing_radius=None):
        layer_headers = self._fill_layer_params(headers=self.headers, layer=layer)
        source_ucoords = UniqueCoords(layer_headers[self._get_cols("source")])
        rec_ucoords = UniqueCoords(layer_headers[self._get_cols("rec")])

        upholes = layer_headers.iloc[source_ucoords.indices]["SourceDepth"].values if layer == 1 else 0
        y = (layer_headers[self.first_breaks_col] - layer_headers['offset'] / layer_headers[f'v{layer+1}_avg']).values
        if layer > 1:
            source_corr = self._calculate_corr_coefs("source", layer_headers, source_ucoords, layer, upholes)
            rec_corr = self._calculate_corr_coefs("rec", layer_headers, rec_ucoords, layer)
            y = y - source_corr - rec_corr
        layer_headers['y'] = y
        layer_headers = layer_headers[layer_headers['y'] > 0]

        # ucoords might change to we need to recalculate them
        source_ucoords = UniqueCoords(layer_headers[self._get_cols("source")])
        rec_ucoords = UniqueCoords(layer_headers[self._get_cols("rec")])
        upholes = layer_headers.iloc[source_ucoords.indices]["SourceDepth"].values if layer == 1 else 0

        ohe_source = self._get_sparse_depths("source", layer_headers, source_ucoords, layer)
        ohe_rec = self._get_sparse_depths("rec", layer_headers, rec_ucoords, layer)
        matrix = sparse.hstack((ohe_source, ohe_rec))

        coords = np.concatenate([source_ucoords.uniques, rec_ucoords.uniques])
        coefs = self.interp_layers_els[layer-1](coords) if self.interp_layers_els[layer-1] is not None else None
        if coefs is None:
            coefs = np.zeros(matrix.shape[1])
        coefs = sparse.linalg.lsqr(matrix, layer_headers['y'], atol=tol, btol=tol, x0=coefs)[0]
        source_els = self.interp_elevations(source_ucoords.uniques)
        rec_els = self.interp_elevations(rec_ucoords.uniques)

        # Distance from 0 elevation to current sub layer.
        source_elevations = source_els - (coefs[:len(source_ucoords.uniques)] + upholes)
        rec_elevations = rec_els - coefs[len(source_ucoords.uniques):]
        joint_interp = self._align_by_proximity(source_ucoords.uniques, source_elevations, rec_ucoords.uniques,
                                                rec_elevations, smoothing_radius)
        self.interp_layers_els[layer-1] = joint_interp

        source_coefs = source_els - joint_interp(source_ucoords.uniques) - upholes
        rec_coefs = rec_els - joint_interp(rec_ucoords.uniques)

        # Save reconstructed 'y' after align sources and receivers elevations and update 'y' in self.headers
        # TODO: create mask with traces that used in current computations. (some traces from current layers
        # was dropped if y for them < 0)
        # mask = self.headers['layer'] == layer
        # if 'pred' not in self.headers.columns:
        #     self.headers['pred'] = None
        # self.headers.loc[mask]['pred'] = matrix.dot(np.concatenate([source_coefs, rec_coefs]))
        # self.headers.loc[mask]['y'] = y

    def _fill_layer_params(self, headers, layer):
        headers = headers[headers["layer"] == layer]
        headers = headers.merge(self.source_params, on=self._get_cols("source") + self.source_headers)
        headers = headers.merge(self.rec_params, on=self._get_cols("rec") + self.rec_headers,
                                suffixes=("_source", "_rec"))
        return headers

    def _get_sparse_depths(self, name, headers, ucoords, layer):
        wv_params = headers[[f'v{layer}_{name}', f'v{layer+1}_{name}']].values.T
        coefs = calculate_depth_coefs(*wv_params)
        eye = sparse.eye((len(ucoords.uniques)), format='csc')[ucoords.inverse]
        matrix = eye.multiply(coefs.reshape(-1, 1)).tocsc()
        return matrix

    def _align_by_proximity(self, source_coords, source_values, rec_coords, rec_values, smoothing_radius=0):
        interp_source = IDWInterpolator(source_coords, source_values, radius=self.radius, neighbors=self.n_neighbors)
        interp_rec = IDWInterpolator(rec_coords, rec_values, radius=self.radius, neighbors=self.n_neighbors)

        source_values = (source_values + interp_rec(source_coords)) / 2
        rec_values = (rec_values + interp_source(rec_coords)) / 2

        values = np.concatenate([source_values, rec_values])
        coords = np.concatenate([source_coords, rec_coords])

        joint_interp = IDWInterpolator(coords, values.reshape(-1), radius=smoothing_radius, dist_transform=0)
        joint_interp = IDWInterpolator(coords, joint_interp(coords), radius=self.radius, neighbors=self.n_neighbors)
        return joint_interp

    def update_velocity(self, max_wv=None, tol=1e-8, smoothing_radius=None):
        layer_headers = self._fill_layer_params(headers=self.headers, layer=1)
        ohe_source, unique_sources, ix_sources = self._get_sparse_velocities("source", layer_headers, 1)
        ohe_rec, unique_recs, ix_recs = self._get_sparse_velocities("rec", layer_headers, 1)
        matrix = sparse.hstack((ohe_source, ohe_rec))

        coefs = layer_headers.iloc[ix_sources]['v1_source'].tolist() + layer_headers.iloc[ix_recs]['v1_rec'].tolist()
        result = sparse.linalg.lsmr(matrix, layer_headers['y'], x0=np.array(coefs), atol=tol, btol=tol)[0]

        source_v1, rec_v1 = self._calculate_velocities(result, layer_headers, ix_sources, ix_recs, max_wv)

        self.interp_v1 = self._align_by_proximity(unique_sources, source_v1, unique_recs, rec_v1, smoothing_radius)

        final_source_v1 = self.interp_v1(unique_sources)
        final_rec_v1 = self.interp_v1(unique_recs)

        self._update_params("source", unique_sources, final_source_v1.reshape(-1, 1), 'v1')
        self._update_params("rec", unique_recs, final_rec_v1.reshape(-1, 1), 'v1')

    def _get_sparse_velocities(self, name, headers, layer):
        coord_names = self._get_cols(name)
        uniques, index, inverse = np.unique(headers[coord_names], axis=0, return_index=True, return_inverse=True)
        elevations = self.interp_elevations(uniques)
        layer_elevations = self.interp_layers_els[layer-1](uniques)
        coefs = elevations - layer_elevations
        if name == "source":
            coefs -= headers.iloc[index]["SourceDepth"].values

        eye = sparse.eye((len(uniques)), format='csc')
        matrix = eye.multiply(coefs).tocsc()[inverse]
        return matrix, uniques, index

    def _calculate_velocities(self, result, headers, ix_sources, ix_recs, max_wv):
        max_wv = np.min(headers[["v2_source", "v2_rec"]].min()) if max_wv is None else max_wv
        v2 = headers.iloc[ix_sources]["v2_source"].tolist() + headers.iloc[ix_recs]["v2_rec"].tolist()
        x1 = calculate_wv_by_v2(np.array(v2), result, max_wv)
        sources = x1[: len(ix_sources)].reshape(-1, 1)
        recs = x1[len(ix_sources): ].reshape(-1, 1)
        return sources, recs

    def calculate_dt(self, datum):
        self._calculate_dt("source", datum)
        self._calculate_dt("rec", datum)

    def _calculate_dt(self, name, datum):
        params = getattr(self, f"{name}_params")
        coords = params.index.to_frame().values
        layers = np.array([self.interp_elevations(coords),
                           *[layer(coords) for layer in self.interp_layers_els if layer is not None],
                           datum])

        layers_width = layers[:-1] - layers[1:]
        if name == "source":
            layers_width[0] -= params["SourceDepth"].values

        v1 = self.interp_v1(coords).reshape(-1, 1) if self.interp_v1 is not None else None
        velocities = params[[f'v{i}' for i in range(1 + (v1 is not None), self.n_layers+2)]].values
        if v1 is not None:
            velocities = np.concatenate((v1, velocities), axis=1)

        dt = np.zeros(len(coords))
        for layer_widths, vels in zip(layers_width, velocities.T):
            dt += layer_widths / vels

        self._update_params(name, coords, dt.reshape(-1, 1), "dt")

    def calculate_metric(self, metrics="mape"):
        metrics = to_list(metrics)
        for metric in metrics:
            metric_call = METRICS.get(metric, None) if not callable(metirc) else metric
            if metric_call is None:
                raise ValueError(f"metrics must be one of the .., not {metric}")
            self.headers[metric] = metric_call(self.headers)

    ### dump ###
    # Raw dumps, just to be able to somehow save results
    def dump(self, name, path, layer):
        columns = self._get_cols(name)
        params = getattr(self, f"{name}_params")
        coords = params.index.to_frame().values
        depths = np.zeros(coords.shape[0])
        if name == 'source':
            columns = columns + ["EnergySourcePoint", "SourceWaterDepth", "GroupWaterDepth"]
            depths = depths - params['SourceDepth'].values
        elif name == 'rec':
            columns = columns + ["ReceiverDatumElevation", "SourceDatumElevation", "ReceiverGroupElevation"]
        else:
            raise ValueError('!!!')
        layer_elevations = self.interp_layers_els[layer-1](coords)
        depths = self.interp_elevations(coords) - layer_elevations
        depths = pd.DataFrame(depths.astype(np.int32), columns=[f"depth_{layer}"], index=params.index)
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

    def plot_layer_elevations(self, layer, **kwargs):
        _, ax = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
        source_coords = self.source_params.index.to_frame().values
        mm_source = MetricMap(source_coords, self.interp_layers_els[layer-1](source_coords))
        mm_source.plot(title="Sources depth", ax=ax[0], **kwargs)
        rec_coords = self.rec_params.index.to_frame().values
        mm_rec = MetricMap(rec_coords, self.interp_layers_els[layer-1](rec_coords))
        mm_rec.plot(title="Receivers depth", ax=ax[1], **kwargs)

    def plot_traveltime_metric(self, layer, by, metric="mape", **kwargs):
        if isinstance(by, str):
            by_to_coords_cols = {
                "shot": ["SourceX", "SourceY"],
                "receiver": ["GroupX", "GroupY"],
                "midpoint": ["CDP_X", "CDP_Y"],
                "bin": ["INLINE_3D", "CROSSLINE_3D"],
            }
            coords_cols = by_to_coords_cols.get(by)
            if coords_cols is None:
                raise ValueError(f"by must be one of {', '.join(by_to_coords_cols.keys())} but {by} given.")
        else:
            coords_cols = to_list(by)
        if len(coords_cols) != 2:
            raise ValueError("Exactly 2 coordinates headers must be passed")

        if metric not in self.headers.columns:
            self.calculate_metric(metric)
        mean_metrics = self.headers.groupby(coords_cols)[metric].mean()

        mm = MetricMap(mean_metrics.index.to_frame().values, mean_metrics.values)
        mm.plot(title=f"Traveltime {metric}", **kwargs)

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
