import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.stats import hmean
from tqdm.auto import tqdm

from .utils import calculate_layer_coefs, calculate_wv_by_v2
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


class StaticCorrection:
    def __init__(self, survey, first_breaks_col, interpolator, n_avg_coords=5, radius=500, n_neighbors=100):
        self.survey = survey.copy()
        self.first_breaks_col = first_breaks_col
        self.n_layers = None
        self.radius = radius
        self.n_neighbors = n_neighbors

        headers_cols = USED_COLS + [first_breaks_col]
        self.headers = pd.DataFrame(self.survey[headers_cols], columns=headers_cols)

        self.source_params = self._create_params_df(name="source")
        self.source_uniques = self.source_params.index.to_frame().values
        self.source_headers = []

        self.rec_params = self._create_params_df(name="rec")
        self.rec_uniques = self.rec_params.index.to_frame().values
        self.rec_headers = []

        self.coords = np.concatenate((self.source_uniques, self.rec_uniques))

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
        y = np.zeros(len(self.headers))
        for i in range(self.n_layers):
            v_avg = hmean(subw_velocities[:, i].reshape(-1, n_avg_coords), axis=1)
            self.headers[f'v{i+2}_avg'] = v_avg
            mask = self.headers['layer'] == i+1
            layer_headers = self.headers[mask]
            y[mask] = (layer_headers[self.first_breaks_col] - layer_headers['offset'] / layer_headers[f'v{i+2}_avg']).values
        self.headers['y'] = y
        self.headers["pred"] = np.nan
        self.headers = self.headers[self.headers['y'] > 0]

    @property
    def n_sources(self):
        return self.source_params.shape[0]

    @property
    def n_recs(self):
        return self.rec_params.shape[0]

    def _create_params_df(self, name):
        coord_names = self._get_cols(name)
        unique_coords = np.unique(self.headers[coord_names], axis=0).astype(np.int32)
        return pd.DataFrame(unique_coords, columns=coord_names).reset_index().set_index(coord_names)

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
        values = np.array(values)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
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

    def optimize(self, n_iters=5, depths_kwargs=None, vel_kwargs=None):
        """!!!"""
        depths_kwargs = depths_kwargs if depths_kwargs is not None else {}
        vel_kwargs = vel_kwargs if vel_kwargs is not None else {}

        for i in tqdm(range(n_iters)):
            if i == 0:
                self.update_depths(**depths_kwargs)
            self.update_velocity(**vel_kwargs)
            self.update_depths(**depths_kwargs)

    def update_depths(self, tol=1e-7, smoothing_radius=0):
        headers = self._add_params_to_headers(headers=self.headers)
        layer_matrixes = []
        ixs = []
        for layer in range(1, self.n_layers+1):
            layer_headers = headers[headers["layer"] == layer]
            ixs.extend(layer_headers.index.to_list())
            source_matrix = self._get_sparse_layer_depths("source", layer_headers, layer)
            rec_matrix = self._get_sparse_layer_depths("rec", layer_headers, layer)
            layer_matrixes.append(sparse.hstack((source_matrix, rec_matrix)))
        matrix = sparse.vstack(layer_matrixes)

        coefs = np.zeros(matrix.shape[1])
        if self.interp_layers_els[0] is not None:
            coefs = np.concatenate((self._get_coefs("source"), self._get_coefs("rec")))

        result = sparse.linalg.lsqr(matrix, headers['y'].iloc[ixs], atol=tol, btol=tol, x0=coefs)[0]
        sources = result[:self.n_sources * self.n_layers].reshape(self.n_layers, self.n_sources)
        recs = result[self.n_sources * self.n_layers:].reshape(self.n_layers, self.n_recs)

        upholes = self.source_params["SourceDepth"].values # Upholes have the same order as sources
        sources[0] += upholes # Do we always add upholes to the first layer

        self.interp_layers_els = self._mulitlayer_align_by_proximity(sources, recs, smoothing_radius=smoothing_radius)

        self.headers["pred"].iloc[ixs] = matrix.dot(result)
        self.matrix = matrix

    def _add_params_to_headers(self, headers):
        headers = headers.merge(self.source_params, on=self._get_cols("source") + self.source_headers)
        headers = headers.merge(self.rec_params, on=self._get_cols("rec") + self.rec_headers,
                                suffixes=("_source", "_rec"))
        return headers

    def _get_sparse_layer_depths(self, name, headers, layer):
        length = getattr(self, f"n_{name}s")
        matrixes = []
        for i in range(1, layer+1):
            wv_params = headers[[f'v{i}_{name}', f'v{i+1}_{name}', f'v{layer+1}_{name}']].values.T
            coefs = calculate_layer_coefs(*wv_params)
            ixs = headers[f'index_{name}'].values
            eye = sparse.eye((length), format='csc')[ixs]
            matrixes.append(eye.multiply(coefs.reshape(-1, 1)).tocsc())
        zeros = sparse.csc_matrix((len(headers), length * self.n_layers - sum([m.shape[1] for m in matrixes])))
        layer_matrix = sparse.hstack((*matrixes, zeros))
        return layer_matrix

    def _get_coefs(self, name):
        coords = getattr(self, f"{name}_uniques")
        depths = [self.interp_elevations(coords) - self.interp_layers_els[i](coords) for i in range(self.n_layers)]
        return np.concatenate(depths)

    def _mulitlayer_align_by_proximity(self, source_values, rec_values, smoothing_radius=0):
        sources_depths = 0
        recs_depths = 0
        interps = []
        source_els = self.interp_elevations(self.source_uniques)
        rec_els = self.interp_elevations(self.rec_uniques)

        for source_layer, rec_layer in zip(source_values, rec_values):

            source_elevations = source_els - (source_layer + sources_depths)
            rec_elevations = rec_els - (rec_layer + recs_depths)
            interp_source = IDWInterpolator(self.source_uniques, source_elevations, radius=self.radius, neighbors=self.n_neighbors)
            interp_rec = IDWInterpolator(self.rec_uniques, rec_elevations, radius=self.radius, neighbors=self.n_neighbors)

            source_vals = (source_elevations + interp_rec(self.source_uniques)) / 2
            rec_vals = (rec_elevations + interp_source(self.rec_uniques)) / 2

            sources_depths += source_els - source_vals
            recs_depths += rec_els - rec_vals

            values = np.concatenate([source_vals, rec_vals])
            joint_interp = IDWInterpolator(self.coords, values.reshape(-1), radius=smoothing_radius, dist_transform=0)
            interps.append(IDWInterpolator(self.coords, joint_interp(self.coords), radius=self.radius, neighbors=self.n_neighbors))
        return interps

    def update_velocity(self, max_wv=None, tol=1e-8, smoothing_radius=0):
        layer_headers = self._add_params_to_headers(headers=self.headers[self.headers['layer']==1])
        ohe_source, unique_sources, ix_sources = self._get_sparse_velocities("source", layer_headers, 1)
        ohe_rec, unique_recs, ix_recs = self._get_sparse_velocities("rec", layer_headers, 1)
        matrix = sparse.hstack((ohe_source, ohe_rec))

        coefs = layer_headers.iloc[ix_sources]['v1_source'].tolist() + layer_headers.iloc[ix_recs]['v1_rec'].tolist()
        result = sparse.linalg.lsmr(matrix, layer_headers['y'], x0=np.array(coefs), atol=tol, btol=tol)[0]

        source_v1, rec_v1 = self._calculate_velocities(result, layer_headers, ix_sources, ix_recs, max_wv)

        self.interp_v1 = self._align_by_proximity(unique_sources, source_v1, unique_recs, rec_v1, smoothing_radius)

        final_source_v1 = self.interp_v1(self.source_uniques)
        final_rec_v1 = self.interp_v1(self.rec_uniques)

        self._update_params("source", self.source_uniques, final_source_v1, 'v1')
        self._update_params("rec", self.rec_uniques, final_rec_v1, 'v1')

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
        return x1[: len(ix_sources)], x1[len(ix_sources): ]

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

    def calculate_dt(self, datum):
        self._calculate_dt("source", datum)
        self._calculate_dt("rec", datum)

    def _calculate_dt(self, name, datum):
        coords = getattr(self, f"{name}_uniques")
        params = getattr(self, f"{name}_params")

        elevations = self.interp_elevations(coords)
        layers = np.array([elevations,
                           *[layer(coords) for layer in self.interp_layers_els if layer is not None]])

        layers_width = layers[:-1] - layers[1:]
        if name == "source":
            layers_width[0] -= params["SourceDepth"].values

        v1 = self.interp_v1(coords).reshape(-1, 1) if self.interp_v1 is not None else None
        velocities = params[[f'v{i}' for i in range(1 + (v1 is not None), self.n_layers+2)]].values
        if v1 is not None:
            velocities = np.concatenate((v1, velocities), axis=1)

        dt = np.zeros(len(coords))
        dist_to_datum = elevations - datum
        for layer_widths, vels in zip(layers_width, velocities.T):
            layer = np.minimum(dist_to_datum, layer_widths)
            dt += layer / vels
            dist_to_datum -= layer

        self._update_params(name, coords, dt, f"dt_{datum}")

    def calculate_metric(self, metrics="mape"):
        metrics = to_list(metrics)
        for metric in metrics:
            metric_call = METRICS.get(metric, None) if not callable(metric) else metric
            if metric_call is None:
                raise ValueError(f"metrics must be one of the .., not {metric}")
            self.headers[metric] = metric_call(self.headers)

    ### dump ###
    def dump(self, name, path, datum, columns):
        # default for source ["SourceWaterDepth", "GroupWaterDepth"]
        # default for rec ["ReceiverDatumElevation", "SourceDatumElevation"]
        columns = to_list(columns)
        coord_names = self._get_cols(name)
        params = getattr(self, f"{name}_params")
        headers = pd.DataFrame(self.survey[coord_names + columns], columns=coord_names + columns).drop_duplicates()

        headers_dt = headers.merge(params[[f"dt_{datum}"]].round(), on=coord_names).sort_values(by=columns)
        dump_columns = [*columns, f"dt_{datum}"]
        self._dump(path, headers_dt[dump_columns], dump_columns)

    def _dump(self, path, df, columns):
        with open(path, 'w', encoding="UTF-8") as f:
            for _, row in df.iterrows():
                nums = "{:8}" * len(columns)
                line = (nums + "\n").format(*row[columns].values)
                f.write(line)

    def load(self, name, path, datum, columns):
        # default for source ["SourceWaterDepth", "GroupWaterDepth"]
        # default for rec ["ReceiverDatumElevation", "SourceDatumElevation"]
        columns = to_list(columns)
        coord_names = self._get_cols(name)
        headers = pd.DataFrame(self.survey[coord_names + columns], columns=coord_names + columns).drop_duplicates()
        dt = pd.read_csv(path, header=None, names=columns + [f"dt_{datum}"], delim_whitespace=True)
        merged = headers.merge(dt, on=columns)
        self._update_params(name, merged[coord_names].values, merged[f"dt_{datum}"].values, f"dt_{datum}")

    ### plotters ###
    def _construct_velicities_interpolatior(self):
        vel_names = [f"v{i}" for i in range(1, self.n_layers+2)]
        source_vels = self.source_params[vel_names].values
        rec_vels = self.rec_params[vel_names].values

        velocities = np.concatenate((source_vels, rec_vels))
        interp_velocities = IDWInterpolator(self.coords, velocities)
        return interp_velocities, np.min(velocities), np.max(velocities)

    def plot_slice(self, layer=1, n_points=100):
        interp_velocities, vmin, vmax = self._construct_velicities_interpolatior()

        obj = MetricMap(self.coords, self.interp_layers_els[layer-1](self.coords))
        StaticsPlot(obj, self.interp_layers_els, self.interp_elevations, interp_velocities, n_points=n_points,
                    vmin=vmin, vmax=vmax).plot()

    def plot_traveltime_metric(self, by="receiver", metric="mape", **kwargs):
        coords_cols = self._get_coords_col(by=by)

        if metric not in self.headers.columns:
            self.calculate_metric(metric)
        mean_metrics = self.headers.groupby(coords_cols)[metric].mean()

        mm = MetricMap(mean_metrics.index.to_frame().values, mean_metrics.values)
        mm.plot(title=f"Traveltime {metric}", **kwargs)

    def _get_coords_col(self, by):
        _ = self
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
        return coords_cols

    def plot_layers_elevations(self, **kwargs):
        _, ax = plt.subplots(self.n_layers, 2, figsize=(12*self.n_layers, 7*self.n_layers), tight_layout=True)
        ax = ax.ravel()
        for layer in range(self.n_layers):
            mm_source = MetricMap(self.source_uniques, self.interp_layers_els[layer](self.source_uniques))
            mm_source.plot(title=f"Source elevations of layer {layer+1}", ax=ax[2*layer], **kwargs)
            mm_rec = MetricMap(self.rec_uniques, self.interp_layers_els[layer](self.rec_uniques))
            mm_rec.plot(title=f"Receiver elevations of layer {layer+1}", ax=ax[2*layer+1], **kwargs)

    def plot_attrs(self, name):
        _, ax = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
        mm_source = MetricMap(self.source_uniques, self.source_params[name])
        mm_source.plot(title=f"sources {name}", ax=ax[0])
        mm_rec = MetricMap(self.rec_uniques, self.rec_params[name])
        mm_rec.plot(title=f"receivers {name}", ax=ax[1])

    def plot_applied_static_map(self, by, column, datum, **kwargs):
        survey = self.survey.copy()
        survey.static_corr = self
        coords_cols = self._get_coords_col(by)
        survey.reindex(coords_cols, inplace=True)

        def _gather_plot(fontsize, coords, ax):
            g = survey.get_gather(coords)
            g.plot(ax=ax, title="before correction")

        def _plot_statics(fontsize, coords, ax):
            g = survey.get_gather(coords)
            g = g.apply_static_correction(datum=datum)
            g.plot(ax=ax, title="after correction")

        mmap = MetricMap(survey.headers.index, survey.headers[column].values)
        mmap.plot(interactive=True, plot_on_click=(_gather_plot, _plot_statics), **kwargs)
