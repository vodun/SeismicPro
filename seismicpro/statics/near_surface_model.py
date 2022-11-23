import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
from scipy.spatial import KDTree

from .data_loader import TensorDataLoader
from .interactive_plot import ProfilePlot
from ..const import HDR_FIRST_BREAK
from ..utils import to_list, IDWInterpolator


class NearSurfaceModel:
    def __init__(self, survey, refractor_velocity_field, first_breaks_col=HDR_FIRST_BREAK, is_uphole=None,
                 init_weathering_velocity=None, max_relative_weathering_velocity=0.8, n_intermediate_points=5,
                 n_smoothing_neighbors=32, device="cpu"):
        survey_list = to_list(survey)
        rvf_list = to_list(refractor_velocity_field)
        if len(survey_list) != len(rvf_list):
            raise ValueError
        if any(rvf_list[0].n_refractors != rvf.n_refractors for rvf in rvf_list):
            raise ValueError

        survey_data = [self._get_survey_data(sur, rvf, first_breaks_col=first_breaks_col, is_uphole=is_uphole)
                       for sur, rvf in zip(survey_list, rvf_list)]
        shots_coords_list, receivers_coords_list, traveltimes_list, field_params_list = zip(*survey_data)
        shots_coords = np.concatenate(shots_coords_list)
        receivers_coords = np.concatenate(receivers_coords_list)
        offsets = np.sqrt(np.sum((shots_coords[:, :2] - receivers_coords[:, :2])**2, axis=1))
        traveltimes = np.concatenate(traveltimes_list)
        field_params = pd.concat(field_params_list, ignore_index=True)
        field_params = field_params.groupby(by=["X", "Y"], as_index=False, sort=False).mean()
        unique_coords = field_params["X", "Y"].to_numpy()

        self.survey_list = survey_list
        self.field_params = field_params
        self.n_refractors = rvf_list[0].n_refractors

        # Define initial model
        field_params_array = field_params[rvf_list[0].param_names].to_numpy()
        slownesses = 1000 / field_params_array[:, self.n_refractors:]
        if init_weathering_velocity is not None:
            weathering_slowness = np.full(len(slownesses), 1000 / init_weathering_velocity)
        else:
            weathering_slowness = 2 * slownesses[:, 0]
        weathering_slowness = np.maximum(weathering_slowness, slownesses[:, 0] / max_relative_weathering_velocity)
        self.max_relative_weathering_velocity = max_relative_weathering_velocity

        intercepts = [field_params_array[:, 0]]
        for i in range(1, self.n_refractors):
            intercepts.append(intercepts[i - 1] + field_params_array[:, i] * (slownesses[:, i - 1] - slownesses[:, i]))

        thicknesses = []
        all_slownesses = np.column_stack([weathering_slowness, slownesses])
        for i in range(self.n_refractors):
            prev_delay = sum(thicknesses[j] * np.sqrt(all_slownesses[:, j]**2 - all_slownesses[:, i]**2)
                             for j in range(i))
            slowness_contrast = np.sqrt(all_slownesses[:, i]**2 - all_slownesses[:, i + 1]**2)
            thicknesses.append((intercepts[i] / 2 - prev_delay) / slowness_contrast)
        thicknesses = np.column_stack(thicknesses)

        # Fit nearest neighbors
        self.n_intermediate_points = n_intermediate_points
        self.elevation_interpolator = IDWInterpolator(unique_coords, field_params["Elevation"].to_numpy(), neighbors=8)
        self.tree = KDTree(unique_coords)
        neighbors_indices = self.tree.query(unique_coords, k=np.arange(1, n_smoothing_neighbors + 2), workers=-1)[1]
        intermediate_indices = self._get_intermediate_indices(shots_coords[:, :2], receivers_coords[:, :2])

        # Convert model parameters to torch tensors
        self.device = device
        self.weathering_slowness_tensor = torch.tensor(weathering_slowness, dtype=torch.float32, device=device)
        self.slownesses_tensor = torch.tensor(slownesses, dtype=torch.float32, device=device)
        self.thicknesses_tensor = torch.tensor(thicknesses, dtype=torch.float32, device=device)
        self.neighbors_indices = torch.tensor(neighbors_indices, device=device)

        # Convert dataset arrays to torch tensors but don't move them to the device
        self.intermediate_indices = torch.tensor(intermediate_indices)
        self.shots_depths = torch.tensor(shots_coords[:, -1], dtype=torch.float32)
        self.offsets = torch.tensor(offsets, dtype=torch.float32)
        self.traveltimes = torch.tensor(traveltimes, dtype=torch.float32)

        # Define optimization-related attributes
        self.weathering_slowness_optimizer = torch.optim.Adam([self.weathering_slowness_tensor], lr=0.1)
        self.weathering_slowness_scheduler = ReduceLROnPlateau(self.weathering_slowness_optimizer, mode="min",
                                                               factor=0.5, patience=100)
        self.thicknesses_optimizer = torch.optim.Adam([self.thicknesses_tensor], lr=1)
        self.thicknesses_scheduler = ReduceLROnPlateau(self.thicknesses_optimizer, mode="min", factor=0.5,
                                                       patience=100)
        self.slownesses_optimizer = torch.optim.Adam([self.slownesses_tensor], lr=0.001)
        self.slownesses_scheduler = ReduceLROnPlateau(self.slownesses_optimizer, mode="min", factor=0.5, patience=100)

        # Define history-related attributes
        self.loss_hist = []
        self.thicknesses_reg_hist = []
        self.weathering_slowness_reg_hist = []
        self.slownesses_reg_hist = []

    @staticmethod
    def _get_survey_data(survey, refractor_velocity_field, first_breaks_col=HDR_FIRST_BREAK, is_uphole=None):
        if is_uphole is None:
            loaded_headers = set(survey.headers.columns) | set(survey.headers.index.names)
            is_uphole = "SourceDepth" in loaded_headers

        shots_coords = survey[["SourceX", "SourceY", "SourceSurfaceElevation"]]
        shots_depths = survey["SourceDepth"] if is_uphole else np.zeros(len(shots_coords))
        shots_coords = np.column_stack([shots_coords[:, :2], shots_depths])
        receivers_coords = survey[["GroupX", "GroupY", "ReceiverGroupElevation"]]
        traveltimes = survey[first_breaks_col]

        shots_elevations = pd.DataFrame(shots_coords[:, :3], columns=["X", "Y", "Elevation"])
        shots_elevations = shots_elevations.groupby(by=["X", "Y"], as_index=False, sort=False).mean()
        receivers_elevations = pd.DataFrame(receivers_coords, columns=["X", "Y", "Elevation"])
        receivers_elevations = receivers_elevations.groupby(by=["X", "Y"], as_index=False, sort=False).mean()
        field_params = pd.concat([shots_elevations, receivers_elevations], ignore_index=True)
        field_params = field_params.groupby(by=["X", "Y"], as_index=False, sort=False).mean()
        rvf_params = refractor_velocity_field.interpolate(field_params["X", "Y"].to_numpy(), is_geographic=True)
        field_params[refractor_velocity_field.param_names] = rvf_params
        return shots_coords, receivers_coords, traveltimes, field_params

    def _get_intermediate_indices(self, shots, receivers):
        intermediate_coords = np.concatenate([(1 - alpha) * shots + alpha * receivers
                                              for alpha in np.linspace(0, 1, 2 + self.n_intermediate_points)])
        return self.tree.query(intermediate_coords, workers=-1)[1].reshape(2 + self.n_intermediate_points, -1).T

    def _estimate_traveltimes_by_indices(self, intermediate_indices, shots_depths, offsets):
        slownesses_tensor = torch.column_stack([self.weathering_slowness_tensor, self.slownesses_tensor])
        intermediate_slownesses = slownesses_tensor[intermediate_indices]
        mean_slownesses = intermediate_slownesses.mean(axis=1)

        # Get shot delay
        shot_ix = intermediate_indices[:, 0]
        shot_slownesses = intermediate_slownesses[:, 0]
        shot_thicknesses = self.thicknesses_tensor[shot_ix]
        shot_thicknesses[:, 0] = torch.clamp(shot_thicknesses[:, 0] - shots_depths, min=0)  # TODO: FIXME
        shot_contrasts = shot_slownesses[:, :-1, None]**2 - shot_slownesses[:, None]**2
        shot_sqrt_contrasts = torch.sqrt(torch.clamp(shot_contrasts, min=1e-8))
        shot_vertical_coefs = torch.where(shot_contrasts > 0, shot_sqrt_contrasts, 0)
        shot_horizontal_coefs = torch.where(shot_contrasts > 0, shot_slownesses[:, :-1, None] / shot_sqrt_contrasts, 0)
        shot_delays = (shot_thicknesses[:, :, None] * shot_vertical_coefs).sum(axis=1)
        shot_horizontal_passes = (shot_thicknesses[:, :, None] * shot_horizontal_coefs).sum(axis=1)

        # Get receiver delay
        receiver_ix = intermediate_indices[:, -1]
        receiver_slownesses = intermediate_slownesses[:, -1]
        receiver_thicknesses = self.thicknesses_tensor[receiver_ix]
        receiver_contrasts = receiver_slownesses[:, :-1, None]**2 - receiver_slownesses[:, None]**2
        receiver_sqrt_contrasts = torch.sqrt(torch.clamp(receiver_contrasts, min=1e-8))
        receiver_vertical_coefs = torch.where(receiver_contrasts > 0, receiver_sqrt_contrasts, 0)
        receiver_horizontal_coefs = torch.where(receiver_contrasts > 0,
                                                receiver_slownesses[:, :-1, None] / receiver_sqrt_contrasts, 0)
        receiver_delays = (receiver_thicknesses[:, :, None] * receiver_vertical_coefs).sum(axis=1)
        receiver_horizontal_passes = (receiver_thicknesses[:, :, None] * receiver_horizontal_coefs).sum(axis=1)

        critical_offsets = shot_horizontal_passes + receiver_horizontal_passes
        valid_mask = offsets.reshape(-1, 1) >= critical_offsets
        traveltimes = shot_delays + receiver_delays + mean_slownesses * offsets.reshape(-1, 1)
        traveltimes = torch.where(valid_mask, traveltimes, traveltimes.max() + 1)
        return traveltimes.min(axis=1)

    def estimate_traveltimes(self, shots, receivers, shots_depths=0, return_refractors_indices=False):
        shots = np.array(shots)
        is_1d = (shots.ndim == 1)
        shots = np.atleast_2d(shots)
        receivers = np.atleast_2d(receivers)
        shots, receivers = np.broadcast_arrays(shots, receivers)
        if shots.ndim > 2 or shots.shape[1] != 2:
            raise ValueError
        shots_depths = np.broadcast_to(shots_depths, len(shots))
        offsets = np.sqrt(np.sum((shots - receivers)**2, axis=1))

        with torch.no_grad():
            intermediate_indices = torch.tensor(self._get_intermediate_indices(shots, receivers), device=self.device)
            shots_depths = torch.tensor(shots_depths, dtype=torch.float32, device=self.device)
            offsets = torch.tensor(offsets, dtype=torch.float32, device=self.device)
            traveltimes, indices = self._estimate_traveltimes_by_indices(intermediate_indices, shots_depths, offsets)

        traveltimes = traveltimes.detach().cpu().numpy()
        indices = indices.detach().cpu().numpy()
        if is_1d:
            traveltimes = traveltimes[0]
            indices = indices[0]

        if return_refractors_indices:
            return traveltimes, indices
        return traveltimes

    def estimate_loss(self, batch_size=32768, bar=True):
        loader = TensorDataLoader(self.intermediate_indices, self.shots_depths, self.offsets, self.traveltimes,
                                  batch_size=batch_size, shuffle=False, drop_last=False, device=self.device)
        batch_sizes = [batch_size] * (len(loader) - 1) + [loader.n_items - batch_size * (len(loader) - 1)]
        with torch.no_grad():
            losses = [(self._estimate_traveltimes_by_indices(indices, depths, offsets)[0] - times).abs().mean().item()
                      for indices, depths, offsets, times in tqdm(loader, desc="Loss estimation", disable=not bar)]
        return np.average(losses, weights=batch_sizes)

    def fit(self, batch_size=32768, n_epochs=1, thicknesses_reg_coef=0, slownesses_reg_coef=0, bar=True):
        thicknesses_reg_coef = torch.tensor(thicknesses_reg_coef, dtype=torch.float32, device=self.device)
        thicknesses_reg_coef = torch.broadcast_to(thicknesses_reg_coef, (self.n_refractors,))
        slownesses_reg_coef = torch.tensor(slownesses_reg_coef, dtype=torch.float32, device=self.device)
        slownesses_reg_coef = torch.broadcast_to(slownesses_reg_coef, (self.n_refractors + 1,))

        self.weathering_slowness_tensor.requires_grad = True
        self.slownesses_tensor.requires_grad = True
        self.thicknesses_tensor.requires_grad = True

        loader = TensorDataLoader(self.intermediate_indices, self.shots_depths, self.offsets, self.traveltimes,
                                  batch_size=batch_size, shuffle=True, drop_last=True, device=self.device)
        with tqdm(total=n_epochs*loader.n_batches, desc="Iterations of model fitting", disable=not bar) as pbar:
            for _ in range(n_epochs):
                for intermediate_indices, depths, offsets, traveltimes in loader:
                    res = self._estimate_traveltimes_by_indices(intermediate_indices, depths, offsets)
                    pred_traveltimes, refractor_indices = res
                    loss = (pred_traveltimes - traveltimes).abs().mean()

                    # Calc thicknesses regularization
                    sensors_indices = torch.unique(torch.cat([intermediate_indices[:, 0], intermediate_indices[:, -1]]), sorted=False)
                    thicknesses = self.thicknesses_tensor[self.coords_neighbors[sensors_indices]]
                    thicknesses_cv = torch.std(thicknesses, axis=1) / torch.mean(thicknesses, axis=1)
                    thicknesses_reg = (thicknesses_cv * thicknesses_reg_coef).mean()

                    weathering_slowness = self.weathering_slowness_tensor[self.coords_neighbors[sensors_indices]]
                    weathering_slowness_cv = torch.std(weathering_slowness, axis=1) / torch.mean(weathering_slowness, axis=1)
                    weathering_slowness_reg = (weathering_slowness_cv * slownesses_reg_coef[0]).mean()

                    # Calc slownesses regularization
                    # slownesses = torch.column_stack([self.weathering_slowness_tensor, self.slownesses_tensor])
                    # slownesses = slownesses[self.coords_neighbors[intermediate_indices]] * slownesses_reg_coef
                    # slownesses = torch.take_along_dim(slownesses, refractor_indices.reshape(-1, 1, 1, 1), axis=-1)
                    # slownesses_reg = torch.abs(slownesses[:, :, 0] -  slownesses.mean(axis=2)).mean()

                    slownesses = torch.column_stack([self.weathering_slowness_tensor, self.slownesses_tensor])
                    slownesses = slownesses[self.coords_neighbors[intermediate_indices]]
                    slownesses_cv = torch.std(slownesses, axis=2) / torch.mean(slownesses, axis=2)
                    slownesses_reg = torch.take_along_dim(slownesses_cv * slownesses_reg_coef, refractor_indices.reshape(-1, 1, 1), axis=-1).mean()

                    self.loss_hist.append(loss.detach().cpu().numpy().item())
                    self.thicknesses_reg_hist.append(thicknesses_reg.detach().cpu().numpy().item())
                    self.weathering_slowness_reg_hist.append(weathering_slowness_reg.detach().cpu().numpy().item())
                    self.slownesses_reg_hist.append(slownesses_reg.detach().cpu().numpy().item())

                    self.weathering_slowness_optimizer.zero_grad()
                    self.slownesses_optimizer.zero_grad()
                    self.thicknesses_optimizer.zero_grad()

                    total_loss = loss + thicknesses_reg + weathering_slowness_reg + slownesses_reg
                    total_loss.backward()
                    loss_item = total_loss.detach().cpu().numpy().item()

                    self.weathering_slowness_optimizer.step()
                    self.slownesses_optimizer.step()
                    self.thicknesses_optimizer.step()

                    self.weathering_slowness_scheduler.step(loss_item)
                    self.slownesses_scheduler.step(loss_item)
                    self.thicknesses_scheduler.step(loss_item)

                    # Enforce physical constraints
                    with torch.no_grad():
                        self.thicknesses_tensor.clamp_(min=0)
                        self.slownesses_tensor.clamp_(min=0)
                        self.slownesses_tensor.data = torch.cummin(self.slownesses_tensor, axis=1)[0]
                        min_weathering_slowness = self.slownesses_tensor[:, 0] / self.max_relative_weathering_velocity
                        self.weathering_slowness_tensor.clamp_(min=min_weathering_slowness)

                    pbar.update(1)

        self.weathering_slowness_tensor.requires_grad = False
        self.slownesses_tensor.requires_grad = False
        self.thicknesses_tensor.requires_grad = False

    def plot_profile(self, **kwargs):
        return ProfilePlot(self, **kwargs).plot()
