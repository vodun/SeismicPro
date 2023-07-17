import torch
import numpy as np


class NearSurfaceModel:
    def __init__(self, grid, velocities, elevations=None, thicknesses=None, n_smoothing_neighbors=32, device="cpu"):
        # Process velocities and validate them
        velocities = self.broadcast_to_grid(velocities, grid)
        if (np.diff(velocities, axis=1) < 0).any():
            raise ValueError
        if (velocities <= 0).any():
            raise ValueError("Layer velocities must be positive")
        slownesses = 1000 / velocities

        # Process elevations or convert thicknesses to elevations
        if elevations is not None and thicknesses is not None:
            raise ValueError("Either elevations or thicknesses should be passed")
        if elevations is None and thicknesses is None:
            elevations = np.empty((len(grid), 0))
        elif elevations is not None:
            elevations = self.broadcast_to_grid(elevations, grid)
        else:
            thicknesses = self.broadcast_to_grid(thicknesses, grid)
            depths = np.cumsum(thicknesses, axis=1)
            elevations = grid.surface_elevations.reshape(-1, 1) - depths

        if (elevations > grid.surface_elevations.reshape(-1, 1)).any():
            raise ValueError
        if (np.diff(elevations, axis=1) > 0).any():
            raise ValueError

        if slownesses.shape[1] != elevations.shape[1] + 1:
            raise ValueError

        self.grid = grid

        # Convert model parameters to torch tensors
        self.device = device
        self.weathering_slowness_tensor = torch.tensor(slownesses[:, 0], dtype=torch.float32, requires_grad=True, device=device)
        self.slownesses_tensor = torch.tensor(slownesses[:, 1:], dtype=torch.float32, requires_grad=True, device=device)
        self.surface_elevation_tensor = torch.tensor(grid.surface_elevations, dtype=torch.float32, device=device)
        self.elevations_tensor = torch.tensor(elevations, dtype=torch.float32, requires_grad=True, device=device)

        # smoothing_interpolator = IDWInterpolator(self.coords, neighbors=n_smoothing_neighbors + 1)
        # neighbors_dist, neighbors_indices = self.coords_tree.query(self.coords, k=np.arange(2, n_smoothing_neighbors + 2), workers=-1)
        # neighbors_weights = smoothing_interpolator._distances_to_weights(neighbors_dist)
        # self.neighbors_indices = torch.tensor(neighbors_indices[:, 1:], dtype=torch.int32, device=device)
        # self.neighbors_weights = torch.tensor(neighbors_weights[:, 1:], dtype=torch.float32, device=device)
        
        # Fit nearest neighbors
        tree = KDTree(self.coords)
        # TODO: handle n_smoothing_neighbors == 0
        neighbors_dists, neighbors_indices = tree.query(self.coords, k=np.arange(2, n_smoothing_neighbors + 2), workers=-1)
        neighbors_dists **= 2
        zero_mask = np.isclose(neighbors_dists, 0)
        neighbors_dists[zero_mask] = 1  # suppress division by zero warning
        neighbors_weights = 1 / neighbors_dists
        neighbors_weights[zero_mask.any(axis=1)] = 0
        neighbors_weights[zero_mask] = 1
        neighbors_weights /= neighbors_weights.sum(axis=1, keepdims=True)
        self.neighbors_indices = torch.tensor(neighbors_indices, dtype=torch.int32, device=device)
        self.neighbors_weights = torch.tensor(neighbors_weights, dtype=torch.float32, device=device)

        # Define default optimization-related attributes
        self.optimizer = torch.optim.Adam([
            {"params": self.weathering_slowness_tensor, "lr": 0.1},
            {"params": self.slownesses_tensor, "lr": 0.001},
            {"params": self.elevations_tensor, "lr": 1},
        ])
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, threshold=0.01, patience=25)

        # Define history-related attributes
        self.loss_hist = []
        self.velocities_reg_hist = []
        self.elevations_reg_hist = []
        self.thicknesses_reg_hist = []

    @property
    def coords(self):
        return self.grid.coords

    @property
    def n_coords(self):
        return self.grid.n_coords
    
    @property
    def coords_tree(self):
        return self.grid.coords_tree

    @property
    def n_refractors(self):
        return self.slownesses_tensor.shape[1]

    @property
    def n_layers(self):
        return self.n_refractors + 1

    @staticmethod
    def broadcast_to_grid(arr, grid):
        arr = np.atleast_1d(arr)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError("Arrays must be 2-dimensional at most")
        return np.broadcast_to(arr, (len(grid), arr.shape[1]))

    @classmethod
    def _init_model(cls, grid, refractor_velocity_field, init_weathering_velocity=None):
        n_refractors = refractor_velocity_field.n_refractors
        rvf_params = refractor_velocity_field.interpolate(grid.coords, is_geographic=True)

        # Initialize weathering velocity
        velocities = rvf_params[:, n_refractors:]
        if init_weathering_velocity is None:
            weathering_velocity = velocities[:, 0] / 2
        elif callable(init_weathering_velocity):
            weathering_velocity = init_weathering_velocity(grid.coords)
        else:
            weathering_velocity = init_weathering_velocity
        weathering_velocity = np.minimum(weathering_velocity, velocities[:, 0])
        velocities = np.column_stack([weathering_velocity, velocities])
        if (velocities <= 0).any():
            raise ValueError("Layer velocities must be positive")

        # Estimate initial layer thicknesses and convert them to layer elevations
        slownesses = 1000 / velocities
        intercept_deltas = np.cumsum(-rvf_params[:, 1:n_refractors:] * np.diff(slownesses[:, 1:], axis=1), axis=1)
        intercepts = np.column_stack([rvf_params[:, 0], rvf_params[:, :1] + intercept_deltas])
        thicknesses = []
        for i in range(n_refractors):
            prev_delay = sum(thicknesses[j] * np.sqrt(slownesses[:, j]**2 - slownesses[:, i]**2)
                             for j in range(i))
            slowness_contrast = np.maximum(np.sqrt(slownesses[:, i]**2 - slownesses[:, i + 1]**2), 0.01)
            thicknesses.append(np.maximum((intercepts[:, i] / 2 - prev_delay) / slowness_contrast, 0.01))
        thicknesses = np.column_stack(thicknesses)
        elevations = grid.surface_elevations.reshape(-1, 1) - thicknesses.cumsum(axis=1)
        return velocities, elevations

    @classmethod
    def from_refractor_velocity_field(cls, grid, refractor_velocity_field, init_weathering_velocity=None, device="cpu"):
        velocities, elevations = cls._init_model(grid, refractor_velocity_field, init_weathering_velocity=init_weathering_velocity)
        return cls(grid, velocities, elevations, device=device)

#     @classmethod
#     def from_file(cls, path, device="cpu", encoding="UTF-8"):
#         params_df = load_dataframe(path, has_header=True, encoding=encoding)
#         n_layers = (len(params_df.columns) - 2) // 2
#         coords_cols = ["x", "y"]
#         elevation_cols = [f"e{i}" for i in range(n_layers)]
#         velocity_cols = [f"v{i}" for i in range(n_layers)]
#         expected_cols = coords_cols + elevation_cols + velocity_cols
#         if set(expected_cols) != set(params_df.columns):
#             raise ValueError

#         coords = params_df[coords_cols].to_numpy()
#         elevations = params_df[elevation_cols].to_numpy()
#         velocities = params_df[velocity_cols].to_numpy()
#         nsm = cls(coords, elevations, velocities, device=device)
#         return nsm

#     def dump(self, path, encoding="UTF-8"):
#         coords_cols = ["x", "y"]
#         elevation_cols = [f"e{i}" for i in range(self.n_layers)]
#         velocity_cols = [f"v{i}" for i in range(self.n_layers)]
#         params_df = pd.DataFrame(self.coords, columns=coords_cols)
#         params_df[elevation_cols[0]] = self.surface_elevation_tensor.detach().cpu().numpy()
#         params_df[elevation_cols[1:]] = self.elevations_tensor.detach().cpu().numpy()
#         params_df[velocity_cols[0]] = 1000 / self.weathering_slowness_tensor.detach().cpu().numpy()
#         params_df[velocity_cols[1:]] = 1000 / self.slownesses_tensor.detach().cpu().numpy()
#         dump_dataframe(params_df, path, has_header=True, encoding=encoding)

    # Traveltime estimation - any survey

    # Training and validation - survey with fb -> dataset

    # QC - survey with fb -> dataset

    # Statics - full survey
    
    def create_dataset(self, survey, first_breaks_header=HDR_FIRST_BREAK, uphole_correction_method="auto",
                       velocity_cell_size=250):
        return self.grid.create_dataset(survey, first_breaks_header, uphole_correction_method, velocity_cell_size)
    
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
        n_refractors = shots_slownesses.shape[-1] - 1

        offsets = torch.sqrt(torch.sum((shots_coords[:, :2] - receivers_coords[:, :2])**2, axis=1))
        dips_tan = (receivers_elevations - shots_elevations) / torch.clamp(offsets, min=0.01).reshape(-1, 1)
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

    @staticmethod
    def _weight_tensor(tensor, indices, weights):
        indexed_shape = indices.shape + tensor.shape[1:]
        indexed = tensor.index_select(0, indices.ravel()).reshape(indexed_shape)
        weights_shape = weights.shape + (1,) * (tensor.ndim - 1)
        return (indexed * weights.reshape(weights_shape)).sum(axis=1)

    def _get_params_by_indices(self, indices, weights):
        slownesses = torch.column_stack([self._weight_tensor(self.weathering_slowness_tensor, indices, weights),
                                         self._weight_tensor(self.slownesses_tensor, indices, weights)])
        elevations = self._weight_tensor(self.elevations_tensor, indices, weights)
        return slownesses, elevations

    def _estimate_traveltimes_by_indices(self, source_coords, source_indices, source_weights,
                                         receiver_coords, receiver_indices, receiver_weights,
                                         grid_indices, grid_weights):
        source_slownesses, source_elevations = self._get_params_by_indices(source_indices, source_weights)
        receiver_slownesses, receiver_elevations = self._get_params_by_indices(receiver_indices, receiver_weights)
        mean_slownesses = torch.column_stack([self._weight_tensor(self.weathering_slowness_tensor, grid_indices, grid_weights),
                                              self._weight_tensor(self.slownesses_tensor, grid_indices, grid_weights)])
        return self._estimate_traveltimes(source_coords, source_slownesses, source_elevations,
                                          receiver_coords, receiver_slownesses, receiver_elevations, mean_slownesses)

    def _estimate_traveltimes_by_loader(self, loader, bar=True):
        with torch.no_grad():
            tt = [self._estimate_traveltimes_by_indices(*batch[:-1]).cpu() for batch in loader]
        return torch.cat(tt)

    def estimate_loss(self, dataset, batch_size=1000000, bar=True):
        loader = dataset.create_train_loader(batch_size=batch_size, n_epochs=1, shuffle=False, drop_last=False,
                                             device=self.device, bar=bar)
        pred_traveltimes = self._estimate_traveltimes_by_loader(loader)
        return torch.abs(pred_traveltimes - dataset.traveltimes).mean().item()

    def fit(self, dataset, batch_size=250000, n_epochs=5, elevations_reg_coef=0.5, thicknesses_reg_coef=0.5,
            velocities_reg_coef=1, bar=True):
        elevations_reg_coef = torch.tensor(elevations_reg_coef, dtype=torch.float32, device=self.device)
        elevations_reg_coef = torch.broadcast_to(elevations_reg_coef, (self.n_refractors,))
        thicknesses_reg_coef = torch.tensor(thicknesses_reg_coef, dtype=torch.float32, device=self.device)
        thicknesses_reg_coef = torch.broadcast_to(thicknesses_reg_coef, (self.n_refractors,))
        velocities_reg_coef = torch.tensor(velocities_reg_coef, dtype=torch.float32, device=self.device)
        velocities_reg_coef = torch.broadcast_to(velocities_reg_coef, (self.n_refractors + 1,))

        loader = dataset.create_train_loader(batch_size=batch_size, n_epochs=n_epochs, shuffle=True, drop_last=True,
                                             device=self.device, bar=bar)
        for batch in loader:
            source_coords, source_indices, source_weights = batch[:3]
            receiver_coords, receiver_indices, receiver_weights = batch[3:6]
            grid_indices, grid_weights, traveltimes = batch[6:]

            pred_traveltimes = self._estimate_traveltimes_by_indices(source_coords, source_indices, source_weights,
                                                                     receiver_coords, receiver_indices, receiver_weights,
                                                                     grid_indices, grid_weights)
            loss = torch.abs(pred_traveltimes - traveltimes).mean()

            # Calc regularization
            valid_source_indices = source_indices[source_weights > 0.1]
            valid_receiver_indices = receiver_indices[receiver_weights > 0.1]
            unique_batch_indices = torch.unique(torch.cat([valid_source_indices, valid_receiver_indices]), sorted=False)
            neighbors_indices = self.neighbors_indices[unique_batch_indices]
            neighbors_weights = self.neighbors_weights[unique_batch_indices][..., None]

            velocities = 1000 / torch.column_stack([self.weathering_slowness_tensor, self.slownesses_tensor])
            batch_velocities = velocities[unique_batch_indices]
            interp_velocities = (neighbors_weights * velocities[neighbors_indices]).sum(axis=1)
            velocities_err = torch.abs(batch_velocities - interp_velocities) / velocities.mean(axis=0)
            velocities_reg = (velocities_err * velocities_reg_coef).mean()

            batch_elevations = self.elevations_tensor[unique_batch_indices]
            interp_elevations = (neighbors_weights * self.elevations_tensor[neighbors_indices]).sum(axis=1)
            elevations_err = torch.abs(batch_elevations - interp_elevations)
            elevations_reg = (elevations_err * elevations_reg_coef).mean()

            thicknesses = -torch.diff(torch.column_stack([self.surface_elevation_tensor, self.elevations_tensor]), axis=1)
            batch_thicknesses = thicknesses[unique_batch_indices]
            interp_thicknesses = (neighbors_weights * thicknesses[neighbors_indices]).sum(axis=1)
            thicknesses_err = torch.abs(batch_thicknesses - interp_thicknesses)
            thicknesses_reg = (thicknesses_err * thicknesses_reg_coef).mean()

            total_loss = loss + velocities_reg + elevations_reg + thicknesses_reg
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
            self.velocities_reg_hist.append(velocities_reg.item())
            self.elevations_reg_hist.append(elevations_reg.item())
            self.thicknesses_reg_hist.append(thicknesses_reg.item())

    # Vis

    def plot_loss(self):
        _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2, figsize=(12, 8), tight_layout=True)
        ax1.plot(self.loss_hist)
        ax1.set_title("Traveltime MAE")
        ax2.plot(self.velocities_reg_hist)
        ax2.set_title("Weighted velocities interpolation MAPE")
        ax3.plot(self.elevations_reg_hist)
        ax3.set_title("Weighted elevations interpolation MAE")
        ax4.plot(self.thicknesses_reg_hist)
        ax4.set_title("Weighted thicknesses interpolation MAE")

    def plot_profile(self, **kwargs):
        return ProfilePlot(self, **kwargs).plot()
