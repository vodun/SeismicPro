import math
from tqdm.auto import tqdm
from numba import njit, prange


class TravelTimeDataset:
    def __init__(self, survey, grid, first_breaks_header=HDR_FIRST_BREAK, uphole_correction_method="auto",
                 velocity_cell_size=250):
        if uphole_correction_method not in {"auto", "time", "depth", None}:
            raise ValueError

        survey_list = to_list(survey)
        traveltime_data = [self._get_traveltime_data(survey, first_breaks_header, uphole_correction_method)
                           for survey in survey_list]
        source_coords_list, receiver_coords_list, traveltimes_list = zip(*traveltime_data)
        source_coords = np.concatenate(source_coords_list)
        receiver_coords = np.concatenate(receiver_coords_list)
        traveltimes = np.concatenate(traveltimes_list)

        source_indices, source_weights = grid.get_interpolation_params(source_coords[:, :2])
        receiver_indices, receiver_weights = grid.get_interpolation_params(receiver_coords[:, :2])

        coords, indices, weights = self.get_velocity_grid(source_coords, receiver_coords, velocity_cell_size)
        coords_grid_indices, coords_grid_weights = grid.get_interpolation_params(coords)
        grid_indices, grid_weights = self.get_grid_weights(coords_grid_indices, coords_grid_weights, indices, weights)

        # Convert dataset arrays to torch tensors but don't move them to the device as they can be large
        self.source_coords = torch.tensor(source_coords, dtype=torch.float32)
        self.source_indices = torch.tensor(source_indices, dtype=torch.int32)
        self.source_weights = torch.tensor(source_weights, dtype=torch.float32)
        self.receiver_coords = torch.tensor(receiver_coords, dtype=torch.float32)
        self.receiver_indices = torch.tensor(receiver_indices, dtype=torch.int32)
        self.receiver_weights = torch.tensor(receiver_weights, dtype=torch.float32)
        self.grid_indices = torch.tensor(grid_indices, dtype=torch.int32)
        self.grid_weights = torch.tensor(grid_weights, dtype=torch.float32)
        self.traveltimes = torch.tensor(traveltimes, dtype=torch.float32)

    @staticmethod
    @njit(nogil=True, parallel=True)
    def get_velocity_grid(source_coords, receiver_coords, velocity_cell_size):
        min_sx = source_coords[:, 0].min()
        min_sy = source_coords[:, 1].min()
        max_sx = source_coords[:, 0].max()
        max_sy = source_coords[:, 1].max()
        min_rx = receiver_coords[:, 0].min()
        min_ry = receiver_coords[:, 1].min()
        max_rx = receiver_coords[:, 0].max()
        max_ry = receiver_coords[:, 1].max()

        grid_origin_x = min(min_sx, min_rx)
        grid_origin_y = min(min_sy, min_ry)
        grid_size_x = 1 + math.ceil((max(max_sx, max_rx) - grid_origin_x) / velocity_cell_size)
        grid_size_y = 1 + math.ceil((max(max_sy, max_ry) - grid_origin_y) / velocity_cell_size)
        grid_coords_x = grid_origin_x + velocity_cell_size * np.arange(grid_size_x)
        grid_coords_y = grid_origin_y + velocity_cell_size * np.arange(grid_size_y)

        coords = np.empty((grid_size_x * grid_size_y, 2))
        for i in prange(grid_size_x):
            for j in range(grid_size_y):
                coords_ix = i * grid_size_y + j
                coords[coords_ix, 0] = grid_coords_x[i]
                coords[coords_ix, 1] = grid_coords_y[j]

        n_traces = len(source_coords)
        trace_n_cells = np.empty(n_traces, dtype=np.int32)
        for i in prange(n_traces):
            offset = np.sqrt(np.sum((receiver_coords[i] - source_coords[i])**2))
            trace_n_cells[i] = 1 + math.ceil(offset / velocity_cell_size)
        max_n_cells = trace_n_cells.max()

        keep_coords_mask = np.zeros(n_traces, dtype=np.bool_)
        indices = np.zeros((n_traces, max_n_cells), dtype=np.int32)
        weights = np.zeros((n_traces, max_n_cells), dtype=np.float32)

        for i in prange(n_traces):
            n_cells = trace_n_cells[i]

            coords_x = np.linspace(source_coords[i, 0], receiver_coords[i, 0], n_cells)
            indices_x = np.round((coords_x - grid_origin_x) / velocity_cell_size).astype(np.int32)
            indices_x = np.clip(indices_x, 0, grid_size_x - 1)

            coords_y = np.linspace(source_coords[i, 1], receiver_coords[i, 1], n_cells)
            indices_y = np.round((coords_y - grid_origin_y) / velocity_cell_size).astype(np.int32)
            indices_y = np.clip(indices_y, 0, grid_size_y - 1)

            intermediate_indices = indices_x * grid_size_y + indices_y
            keep_coords_mask[intermediate_indices] = True
            indices[i, :n_cells] = intermediate_indices
            weights[i, :n_cells] = 1 / n_cells

        indices_shift = np.cumsum(~keep_coords_mask)
        for i in prange(n_traces):
            n_cells = trace_n_cells[i]
            indices[i, :n_cells] -= indices_shift[indices[i, :n_cells]]

        return coords[keep_coords_mask], indices, weights

    @staticmethod
    @njit(nogil=True, parallel=True)
    def get_grid_weights(coords_grid_indices, coords_grid_weights, indices, weights):
        n_traces = len(indices)
        grid_indices = np.empty((n_traces, indices.shape[1] * coords_grid_indices.shape[1]), dtype=np.int32)
        grid_weights = np.empty((n_traces, indices.shape[1] * coords_grid_indices.shape[1]), dtype=np.float32)

        for i in prange(n_traces):
            grid_indices[i] = coords_grid_indices[indices[i]].ravel()
            grid_weights[i] = (weights[i].reshape(-1, 1) * coords_grid_weights[indices[i]]).ravel()

        return grid_indices, grid_weights

    @staticmethod
    def _get_uphole_correction_method(survey, uphole_correction_method):
        if uphole_correction_method != "auto":
            return uphole_correction_method
        if not survey.is_uphole:
            return None
        return "time" if "SourceUpholeTime" in survey.available_headers else "depth"

    @classmethod
    def _get_traveltime_data(cls, survey, first_breaks_header, uphole_correction_method):
        source_coords = survey[["SourceX", "SourceY", "SourceSurfaceElevation"]]
        receiver_coords = survey[["GroupX", "GroupY", "ReceiverGroupElevation"]]
        traveltimes = survey[first_breaks_header]

        uphole_correction_method = cls._get_uphole_correction_method(survey, uphole_correction_method)
        if uphole_correction_method == "time":
            traveltimes = traveltimes + survey["SourceUpholeTime"]
        elif uphole_correction_method == "depth":
            source_coords[:, -1] -= survey["SourceDepth"]
        return source_coords, receiver_coords, traveltimes

    # def _get_predict_traveltime_data(self, container, uphole_correction_method):
    #     source_coords = container[["SourceX", "SourceY", "SourceSurfaceElevation"]]
    #     receiver_coords = container[["GroupX", "GroupY", "ReceiverGroupElevation"]]
    #     if uphole_correction_method == "depth":
    #         source_coords[:, -1] -= container["SourceDepth"]
    #     if uphole_correction_method == "time":
    #         traveltime_correction = container["SourceUpholeTime"]
    #     else:
    #         traveltime_correction = np.zeros(container.n_traces, dtype=np.float32)
    #     return source_coords, receiver_coords, traveltime_correction

    @property
    def train_tensors(self):
        return (self.source_coords, self.source_indices, self.source_weights,
                self.receiver_coords, self.receiver_indices, self.receiver_weights,
                self.grid_indices, self.grid_weights, self.traveltimes)

    def create_train_loader(self, batch_size=1, n_epochs=1, shuffle=True, drop_last=True, device=None, bar=True):
        loader = TensorDataLoader(*self.train_tensors, batch_size=batch_size, n_epochs=n_epochs,
                                  shuffle=shuffle, drop_last=drop_last, device=device)
        return tqdm(loader, desc="Iterations of model fitting", disable=not bar)

#     def create_predict_loader(self, ...):
#         pass
