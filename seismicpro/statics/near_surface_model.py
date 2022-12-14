import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
from scipy.spatial import KDTree

from .data_loader import TensorDataLoader
from .metrics import TRAVELTIME_QC_METRICS
from .interactive_plot import ProfilePlot, StaticsCorrectionPlot
from ..const import HDR_FIRST_BREAK
from ..metrics import PartialMetric
from ..utils import to_list, IDWInterpolator


class NearSurfaceModel:
    def __init__(self, survey, refractor_velocity_field, first_breaks_col=HDR_FIRST_BREAK, is_uphole=None,
                 init_weathering_velocity=None, n_intermediate_points=5, n_smoothing_neighbors=32, device="cpu"):
        survey_list = to_list(survey)
        rvf_list = to_list(refractor_velocity_field)
        if len(survey_list) != len(rvf_list):
            raise ValueError
        if any(rvf_list[0].n_refractors != rvf.n_refractors for rvf in rvf_list):
            raise ValueError

        self.is_uphole = is_uphole
        self.first_breaks_col = first_breaks_col
        survey_data = [self._get_survey_data(sur, rvf) for sur, rvf in zip(survey_list, rvf_list)]
        shots_coords_list, receivers_coords_list, traveltimes_list, field_params_list = zip(*survey_data)
        shots_coords = np.concatenate(shots_coords_list)
        receivers_coords = np.concatenate(receivers_coords_list)
        traveltimes = np.concatenate(traveltimes_list)
        field_params = pd.concat(field_params_list, ignore_index=True)
        field_params = field_params.groupby(by=["X", "Y"], as_index=False, sort=False).mean()
        unique_coords = field_params[["X", "Y"]].to_numpy()
        surface_elevation = field_params["Elevation"].to_numpy()
        self.surface_elevation_interpolator = IDWInterpolator(unique_coords, surface_elevation, neighbors=8)

        self.survey_list = survey_list
        self.field_params = field_params
        self.n_refractors = rvf_list[0].n_refractors

        # Define initial model
        field_params_array = field_params[rvf_list[0].param_names].to_numpy()
        slownesses = 1000 / field_params_array[:, self.n_refractors:]
        if init_weathering_velocity is not None:
            weathering_slowness = np.maximum(1000 / init_weathering_velocity, slownesses[:, 0])
        else:
            weathering_slowness = 2 * slownesses[:, 0]

        intercepts = [field_params_array[:, 0]]
        for i in range(1, self.n_refractors):
            intercepts.append(intercepts[i - 1] + field_params_array[:, i] * (slownesses[:, i - 1] - slownesses[:, i]))

        thicknesses = []
        all_slownesses = np.column_stack([weathering_slowness, slownesses])
        for i in range(self.n_refractors):
            prev_delay = sum(thicknesses[j] * np.sqrt(all_slownesses[:, j]**2 - all_slownesses[:, i]**2)
                             for j in range(i))
            slowness_contrast = np.sqrt(all_slownesses[:, i]**2 - all_slownesses[:, i + 1]**2)
            thicknesses.append(np.nan_to_num((intercepts[i] / 2 - prev_delay) / slowness_contrast))
        thicknesses = np.column_stack(thicknesses)
        thicknesses[:] = thicknesses.mean(axis=0)  # TODO: FIXME use mean refractor velocity
        thicknesses[:, 0] = np.maximum(thicknesses[:, 0], (shots_coords[:, 2] - shots_coords[:, 3]).max() + 1)
        elevations = surface_elevation.reshape(-1, 1) - thicknesses.cumsum(axis=1)

        # Fit nearest neighbors
        self.tree = KDTree(unique_coords)
        self.n_intermediate_points = n_intermediate_points
        neighbors_indices = self.tree.query(unique_coords, k=np.arange(1, n_smoothing_neighbors + 2), workers=-1)[1]
        intermediate_indices = self._get_intermediate_indices(shots_coords[:, :2], receivers_coords[:, :2])

        # Convert model parameters to torch tensors
        self.device = device
        self.weathering_slowness_tensor = torch.tensor(weathering_slowness, dtype=torch.float32, requires_grad=True, device=device)
        self.slownesses_tensor = torch.tensor(slownesses, dtype=torch.float32, requires_grad=True, device=device)
        self.elevations_tensor = torch.tensor(elevations, dtype=torch.float32, requires_grad=True, device=device)
        self.surface_elevation_tensor = torch.tensor(surface_elevation, dtype=torch.float32, device=device)
        self.neighbors_indices = torch.tensor(neighbors_indices, device=device)

        # Convert dataset arrays to torch tensors but don't move them to the device as they can be large
        self.shots_coords = torch.tensor(shots_coords[:, [0, 1, 3]], dtype=torch.float32)
        self.receivers_coords = torch.tensor(receivers_coords, dtype=torch.float32)
        self.intermediate_indices = torch.tensor(intermediate_indices)
        self.traveltimes = torch.tensor(traveltimes, dtype=torch.float32)

        # Define default optimization-related attributes
        self.optimizer = torch.optim.Adam([
            {"params": self.weathering_slowness_tensor, "lr": 0.1},
            {"params": self.slownesses_tensor, "lr": 0.001},
            {"params": self.elevations_tensor, "lr": 1},
        ])
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, threshold=0.01, patience=25)

        # Define history-related attributes
        self.loss_hist = []
        self.slownesses_reg_hist = []
        self.elevations_reg_hist = []

    def _get_survey_data(self, survey, refractor_velocity_field):
        is_uphole = self.is_uphole
        if is_uphole is None:
            loaded_headers = set(survey.headers.columns) | set(survey.headers.index.names)
            is_uphole = "SourceDepth" in loaded_headers

        shots_coords = survey[["SourceX", "SourceY", "SourceSurfaceElevation"]]
        shots_depths = survey["SourceDepth"] if is_uphole else 0
        shots_coords = np.column_stack([shots_coords, shots_coords[:, -1] - shots_depths])
        receivers_coords = survey[["GroupX", "GroupY", "ReceiverGroupElevation"]]
        traveltimes = survey[self.first_breaks_col]

        shots_elevations = pd.DataFrame(shots_coords[:, :3], columns=["X", "Y", "Elevation"])
        shots_elevations = shots_elevations.groupby(by=["X", "Y"], as_index=False, sort=False).mean()
        receivers_elevations = pd.DataFrame(receivers_coords, columns=["X", "Y", "Elevation"])
        receivers_elevations = receivers_elevations.groupby(by=["X", "Y"], as_index=False, sort=False).mean()
        field_params = pd.concat([shots_elevations, receivers_elevations], ignore_index=True)
        field_params = field_params.groupby(by=["X", "Y"], as_index=False, sort=False).mean()
        rvf_params = refractor_velocity_field.interpolate(field_params[["X", "Y"]].to_numpy(), is_geographic=True)
        field_params[refractor_velocity_field.param_names] = rvf_params
        return shots_coords, receivers_coords, traveltimes, field_params

    def _get_intermediate_indices(self, shots, receivers):
        intermediate_coords = np.concatenate([(1 - alpha) * shots + alpha * receivers
                                              for alpha in np.linspace(0, 1, 2 + self.n_intermediate_points)])
        return self.tree.query(intermediate_coords, workers=-1)[1].reshape(2 + self.n_intermediate_points, -1).T

    def _describe_rays(self, coords, slownesses, elevations, dips_cos, dips_tan):
        batch_size = len(coords)
        n_refractors = slownesses.shape[-1] - 1

        incidence_sin = torch.clamp(slownesses[:, 1:, None] / slownesses[:, None, :-1], max=0.999)
        incidence_cos = torch.sqrt(1 - incidence_sin**2)
        incidence_tan = incidence_sin / incidence_cos

        dist_to_layer = coords[:, -1:] - elevations  # (bs, n_ref)
        layer_above_mask = dist_to_layer < 0
        coords_layer = layer_above_mask.sum(axis=1)  # (bs,)
        dist_to_layer[layer_above_mask] = 0
        zeros_tensor = torch.zeros(len(dist_to_layer), 1, dtype=dist_to_layer.dtype, device=dist_to_layer.device)
        vertical_pass_dist = torch.diff(dist_to_layer, prepend=zeros_tensor, axis=-1)  # (bs, n_ref) [0 ... n_ref - 1]

        normal_dist_list = [(vertical_pass_dist[:, 0, None] * dips_cos[:, 0, None]).broadcast_to(batch_size, n_refractors)]
        horizontal_correction_dist = 0  # (bs, n_ref (refracted from))
        for i in range(0, n_refractors - 1):
            horizontal_correction_dist = horizontal_correction_dist + normal_dist_list[i] * dips_cos[:, i, None] * (incidence_tan[:, :, i] + dips_tan[:, i, None])
            corrected_vertical_pass_dist = vertical_pass_dist[:, i + 1, None] + horizontal_correction_dist * (dips_tan[:, i, None] - dips_tan[:, i + 1, None])
            normal_dist_list.append(corrected_vertical_pass_dist * dips_cos[:, i + 1, None])
        normal_dist = torch.stack(normal_dist_list, axis=-1)
        arange = torch.arange(n_refractors, device=self.device)
        zero_mask = (arange.reshape(-1, 1) < arange).broadcast_to(batch_size, n_refractors, n_refractors)
        normal_dist[zero_mask] = 0

        paths_along_refractors = normal_dist * incidence_tan
        incidence_times = (normal_dist / incidence_cos * slownesses[:, None, :-1]).sum(axis=-1)
        return incidence_times, paths_along_refractors, coords_layer

    def _get_vertical_traveltimes(self, src_elevations, dst_elevations, elevations, slownesses):
        high_elevations = torch.maximum(src_elevations, dst_elevations)
        low_elevations = torch.minimum(src_elevations, dst_elevations)
        passes_len = high_elevations - low_elevations
        dist_to_layer = high_elevations.reshape(-1, 1) - elevations
        dist_to_layer[dist_to_layer < 0] = 0
        passes = torch.diff(dist_to_layer, prepend=torch.zeros_like(elevations[:, :1]), append=dist_to_layer[:, -1:], axis=-1)
        zero_mask = passes.cumsum(axis=1) > passes_len.reshape(-1, 1)
        passes[zero_mask] = 0
        overflow_ix = zero_mask.max(axis=1)[1]
        last_ix = torch.where(zero_mask.any(axis=1), overflow_ix, torch.full_like(overflow_ix, -1))
        passes[torch.arange(len(src_elevations)), last_ix] = passes_len - passes.sum(axis=1)
        return (passes * slownesses).sum(axis=1)

    def _estimate_direct_traveltimes(self, shots_coords, shots_elevations, shots_slownesses, shots_layers,
                                     receivers_coords, receivers_elevations, receivers_slownesses, receivers_layers,
                                     offsets, mean_slownesses):
        batch_size = len(shots_coords)
        max_layer = torch.maximum(shots_layers, receivers_layers)
        padded_shots_elevations = torch.column_stack([shots_coords[:, -1], shots_elevations])
        shots_last_elevation = torch.minimum(shots_coords[:, -1], padded_shots_elevations[torch.arange(batch_size), max_layer])
        padded_receivers_elevations = torch.column_stack([receivers_coords[:, -1], receivers_elevations])
        receivers_last_elevation = torch.minimum(receivers_coords[:, -1], padded_receivers_elevations[torch.arange(batch_size), max_layer])
        max_layer_dist = torch.sqrt(offsets**2 + (receivers_last_elevation - shots_last_elevation)**2)
        max_layer_traveltime = max_layer_dist * mean_slownesses[torch.arange(batch_size), max_layer]
        shots_correction = self._get_vertical_traveltimes(shots_coords[:, -1], shots_last_elevation, shots_elevations, shots_slownesses)
        receivers_correction = self._get_vertical_traveltimes(receivers_coords[:, -1], receivers_last_elevation, receivers_elevations, receivers_slownesses)
        return max_layer_traveltime + shots_correction + receivers_correction

    def _estimate_traveltimes(self, shots_coords, shots_slownesses, shots_elevations, receivers_coords,
                              receivers_slownesses, receivers_elevations, mean_slownesses):
        batch_size = len(shots_coords)
        n_refractors = shots_slownesses.shape[-1] - 1

        offsets = torch.sqrt(torch.sum((shots_coords[:, :2] - receivers_coords[:, :2])**2, axis=1))
        dips_tan = (receivers_elevations - shots_elevations) / offsets.reshape(-1, 1)
        dips_cos = torch.sqrt(1 / (1 + dips_tan**2))
        dips_sin = dips_cos * dips_tan
        dips_cos_diff = torch.column_stack([dips_cos[:, 0], dips_cos[:, 1:] * dips_cos[:, :-1] + dips_sin[:, 1:] * dips_sin[:, :-1]])

        shots_stats = self._describe_rays(shots_coords, shots_slownesses, shots_elevations, dips_cos, dips_tan)
        shots_incidence_times, shots_incidence_paths_along_refractors, shots_layers = shots_stats
        receivers_stats = self._describe_rays(receivers_coords, receivers_slownesses, receivers_elevations,
                                              dips_cos, -dips_tan)
        receivers_incidence_times, receivers_incidence_paths_along_refractors, receivers_layers = receivers_stats
        incidence_paths_along_refractors = shots_incidence_paths_along_refractors + receivers_incidence_paths_along_refractors

        paths_list = []
        curr_paths = offsets.reshape(-1, 1)
        for i in range(n_refractors):
            curr_paths = curr_paths * dips_cos_diff[i] - incidence_paths_along_refractors[:, :, i]
            paths_list.append(curr_paths[:, i])
        paths_along_refractors = torch.column_stack(paths_list)  # (bs, n_ref)

        direct_traveltimes = self._estimate_direct_traveltimes(shots_coords, shots_elevations, shots_slownesses, shots_layers,
                                                               receivers_coords, receivers_elevations, receivers_slownesses, receivers_layers,
                                                               offsets, mean_slownesses)
        refracted_traveltimes = paths_along_refractors * mean_slownesses[:, 1:] + shots_incidence_times + receivers_incidence_times
        ignore_mask = ((paths_along_refractors < 0) |
                       (torch.maximum(shots_layers, receivers_layers).reshape(-1, 1) > torch.arange(n_refractors, device=self.device)))
        undefined_traveltime = torch.maximum(refracted_traveltimes.max(), direct_traveltimes.max()) + 1
        traveltimes = torch.column_stack([direct_traveltimes, torch.where(~ignore_mask, refracted_traveltimes, undefined_traveltime)])
        return traveltimes.min(axis=1)[0]

    def _get_params_by_indices(self, indices):
        slownesses = torch.column_stack([self.weathering_slowness_tensor[indices], self.slownesses_tensor[indices]])
        elevations = self.elevations_tensor[indices]
        return slownesses, elevations

    def _estimate_traveltimes_by_indices(self, shots_coords, receivers_coords, intermediate_indices):
        shots_slownesses, shots_elevations = self._get_params_by_indices(intermediate_indices[:, 0])
        receivers_slownesses, receivers_elevations = self._get_params_by_indices(intermediate_indices[:, -1])
        mean_slownesses = torch.column_stack([self.weathering_slowness_tensor[intermediate_indices].mean(axis=1),
                                              self.slownesses_tensor[intermediate_indices].mean(axis=1)])
        return self._estimate_traveltimes(shots_coords, shots_slownesses, shots_elevations, receivers_coords,
                                          receivers_slownesses, receivers_elevations, mean_slownesses)

    def _estimate_train_traveltimes(self, batch_size=1000000, bar=True):
        loader = TensorDataLoader(self.shots_coords, self.receivers_coords, self.intermediate_indices,
                                  batch_size=batch_size, shuffle=False, drop_last=False, device=self.device)
        tqdm_loader = tqdm(loader, desc="Traveltime estimation", disable=not bar)
        with torch.no_grad():
            tt = [self._estimate_traveltimes_by_indices(shots_coords, receivers_coords, intermediate_indices).cpu()
                  for shots_coords, receivers_coords, intermediate_indices in tqdm_loader]
        return torch.cat(tt)

    def _process_coords(self, coords):
        coords = np.array(coords)
        is_1d = coords.ndim == 1
        coords = np.atleast_2d(coords)
        if coords.ndim > 2 or coords.shape[1] not in {2, 3}:
            raise ValueError
        if coords.shape[1] == 2:
            coords = np.column_stack([coords, self.surface_elevation_interpolator(coords)])
        return coords, is_1d

    def estimate_traveltimes(self, shots, receivers):
        shots, is_1d_shots = self._process_coords(shots)
        receivers, is_1d_receivers = self._process_coords(receivers)
        shots, receivers = np.broadcast_arrays(shots, receivers)

        intermediate_indices = torch.tensor(self._get_intermediate_indices(shots[:, :2], receivers[:, :2]), device=self.device)
        shots = torch.tensor(shots, device=self.device)
        receivers = torch.tensor(receivers, device=self.device)
        with torch.no_grad():
            traveltimes = self._estimate_traveltimes_by_indices(shots, receivers, intermediate_indices).cpu().numpy()

        if is_1d_shots and is_1d_receivers:
            return traveltimes[0]
        return traveltimes

    def estimate_delays(self, coords, intermediate_datum=None, intermediate_datum_refractor=None, final_datum=None, replacement_velocity=None):
        coords, is_1d = self._process_coords(coords)
        indices = self.tree.query(coords[:, :2], workers=-1)[1]
        slownesses, elevations = self._get_params_by_indices(indices)
        coords_elevations = torch.tensor(coords[:, -1], dtype=torch.float32, device=self.device)
        if intermediate_datum is not None:
            intermediate_elevations = torch.tensor(intermediate_datum, dtype=torch.float32, device=self.device)
        else:
            intermediate_elevations = elevations[:, intermediate_datum_refractor - 1]
        sign = torch.sign(coords_elevations - intermediate_elevations)
        with torch.no_grad():
            delays = sign * self._get_vertical_traveltimes(coords_elevations, intermediate_elevations, elevations, slownesses)
        if final_datum is not None and replacement_velocity is not None:
            delays += 1000 * (intermediate_elevations - final_datum) / replacement_velocity
        delays = delays.detach().cpu().numpy()

        if is_1d:
            return delays[0]
        return delays

    def estimate_loss(self, batch_size=1000000, bar=True):
        pred_traveltimes = self._estimate_train_traveltimes(batch_size=batch_size, bar=bar)
        return torch.abs(pred_traveltimes - self.traveltimes).mean().item()

    def fit(self, batch_size=250000, n_epochs=5, elevations_reg_coef=0.1, slownesses_reg_coef=1, bar=True):
        elevations_reg_coef = torch.tensor(elevations_reg_coef, dtype=torch.float32, device=self.device)
        elevations_reg_coef = torch.broadcast_to(elevations_reg_coef, (self.n_refractors,))
        slownesses_reg_coef = torch.tensor(slownesses_reg_coef, dtype=torch.float32, device=self.device)
        slownesses_reg_coef = torch.broadcast_to(slownesses_reg_coef, (self.n_refractors + 1,))

        loader = TensorDataLoader(self.shots_coords, self.receivers_coords, self.intermediate_indices, self.traveltimes,
                                  batch_size=batch_size, shuffle=True, drop_last=True, device=self.device)
        with tqdm(total=n_epochs*loader.n_batches, desc="Iterations of model fitting", disable=not bar) as pbar:
            for _ in range(n_epochs):
                for shots_coords, receivers_coords, intermediate_indices, traveltimes in loader:
                    pred_traveltimes = self._estimate_traveltimes_by_indices(shots_coords, receivers_coords, intermediate_indices)
                    loss = torch.abs(pred_traveltimes - traveltimes).mean()

                    # Calc regularization
                    neighbors_indices = self.neighbors_indices[torch.unique(intermediate_indices[:, [0, -1]].ravel(), sorted=False)]
                    slownesses = torch.column_stack([self.weathering_slowness_tensor, self.slownesses_tensor])[neighbors_indices]
                    slownesses_reg = (torch.std(slownesses, axis=1) / torch.mean(slownesses, axis=1) * slownesses_reg_coef).mean()
                    elevations = self.elevations_tensor[neighbors_indices]
                    elevations_reg = (torch.std(elevations, axis=1) * elevations_reg_coef).mean()

                    total_loss = loss + slownesses_reg + elevations_reg
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()
                    self.scheduler.step(total_loss.item())

                    # Enforce model constraints
                    with torch.no_grad():
                        self.elevations_tensor.clamp_(max=self.surface_elevation_tensor.reshape(-1, 1))
                        self.elevations_tensor.data = torch.cummin(self.elevations_tensor, axis=1)[0]
                        self.slownesses_tensor.clamp_(min=0.01)
                        self.slownesses_tensor.data = torch.cummin(self.slownesses_tensor, axis=1)[0]
                        self.weathering_slowness_tensor.clamp_(min=self.slownesses_tensor[:, 0])

                    self.loss_hist.append(loss.item())
                    self.slownesses_reg_hist.append(slownesses_reg.item())
                    self.elevations_reg_hist.append(elevations_reg.item())
                    pbar.update(1)

    def qc(self, metrics=None, by="shot", bar=True):
        if metrics is None:
            metrics = TRAVELTIME_QC_METRICS

        coords_cols = {"shot": ["SourceX", "SourceY"], "receiver": ["GroupX", "GroupY"]}[by]
        qc_df = pd.DataFrame(np.column_stack([self.shots_coords[:, :2].numpy(), self.receivers_coords[:, :2].numpy()]),
                             columns=["SourceX", "SourceY", "GroupX", "GroupY"])
        qc_df["True"] = self.traveltimes.numpy()
        qc_df["Pred"] = self._estimate_train_traveltimes(bar=bar).numpy()
        coords_to_indices = qc_df.groupby(coords_cols).indices
        coords = np.stack(list(coords_to_indices.keys()))

        def calc_metrics(group_indices):
            group = qc_df.iloc[group_indices].to_numpy()
            shots_coords = group[:, :2]
            receivers_coords = group[:, 2:4]
            true_traveltimes = group[:, 4]
            pred_traveltimes = group[:, 5]
            return [metric.calc(shots_coords, receivers_coords, true_traveltimes, pred_traveltimes) for metric in metrics]

        metrics = [PartialMetric(metric, nsm=self, survey_list=self.survey_list, coords_cols=coords_cols)
                   for metric in metrics]
        results = [calc_metrics(indices) for indices in tqdm(coords_to_indices.values(), desc="Coordinates processed", disable=not bar)]
        metrics_maps = [metric.map_class(coords, metric_values, coords_cols=coords_cols, metric=metric)
                        for metric, metric_values in zip(metrics, zip(*results))]
        return metrics_maps

    def plot_loss(self):
        _, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 4), tight_layout=True)
        ax1.plot(self.loss_hist)
        ax1.set_title("Total loss")
        ax2.plot(self.elevations_reg_hist)
        ax2.set_title("Elevations reg")
        ax3.plot(self.slownesses_reg_hist)
        ax3.set_title("Slowness reg")

    def plot_profile(self, **kwargs):
        return ProfilePlot(self, **kwargs).plot()

    def plot_statics_correction(self, survey=None, **kwargs):
        survey_list = self.survey_list if survey is None else to_list(survey)
        return StaticsCorrectionPlot(self, survey_list, **kwargs).plot()
