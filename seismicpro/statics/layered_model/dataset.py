import math

import torch
import numpy as np
from numba import njit, prange
from tqdm.auto import tqdm

from .dataloader import TensorDataLoader
from ..utils import get_uphole_correction_method
from ...utils import to_list, align_args
from ...const import HDR_FIRST_BREAK


class TravelTimeDataset:
    def __init__(self, survey, grid, first_breaks_header=HDR_FIRST_BREAK, uphole_correction_method="auto",
                 velocity_cell_size=250):
        self.grid = grid
        self.survey_list = to_list(survey)

        source_coords = np.concatenate([survey[["SourceX", "SourceY"]] for survey in self.survey_list])
        source_indices, source_weights = self.grid.get_interpolation_params(source_coords)
        self.source_coords = torch.tensor(source_coords, dtype=torch.float32)
        self.source_indices = torch.tensor(source_indices, dtype=torch.int32)
        self.source_weights = torch.tensor(source_weights, dtype=torch.float32)

        receiver_coords = np.concatenate([survey[["GroupX", "GroupY"]] for survey in self.survey_list])
        receiver_indices, receiver_weights = self.grid.get_interpolation_params(receiver_coords)
        receiver_elevations = np.concatenate([survey["ReceiverGroupElevation"] for survey in self.survey_list])
        self.receiver_coords = torch.tensor(receiver_coords, dtype=torch.float32)
        self.receiver_elevations = torch.tensor(receiver_elevations, dtype=torch.float32)
        self.receiver_indices = torch.tensor(receiver_indices, dtype=torch.int32)
        self.receiver_weights = torch.tensor(receiver_weights, dtype=torch.float32)

        # Process uphole correction method: set proper source elevation and target traveltime
        self.first_breaks_header = first_breaks_header
        self.uphole_correction_method_list = None
        self.source_elevations = None
        self.true_traveltimes = None
        self.pred_traveltimes = None
        self.target_traveltimes = None
        self.traveltime_corrections = None
        self.set_uphole_correction_method(uphole_correction_method)

        # Construct an interpolation grid for mean slowness estimation
        coords, indices, weights = self.get_velocity_grid(source_coords, receiver_coords, velocity_cell_size)
        coords_grid_indices, coords_grid_weights = self.grid.get_interpolation_params(coords)
        grid_indices, grid_weights = self.get_grid_weights(coords_grid_indices, coords_grid_weights, indices, weights)
        self.grid_indices = torch.tensor(grid_indices, dtype=torch.int32)
        self.grid_weights = torch.tensor(grid_weights, dtype=torch.float32)

    @property
    def has_predictions(self):
        return self.pred_traveltimes is not None

    def _process_survey_uphole_correction_method(self, survey, uphole_correction_method):
        source_elevations = survey["SourceSurfaceElevation"]
        true_traveltimes = survey[self.first_breaks_header]
        target_traveltimes = true_traveltimes
        uphole_correction_method = get_uphole_correction_method(survey, uphole_correction_method)
        if uphole_correction_method == "time":
            traveltime_corrections = survey["SourceUpholeTime"]
            target_traveltimes = target_traveltimes + traveltime_corrections
        elif uphole_correction_method == "depth":
            source_elevations = source_elevations - survey["SourceDepth"]
            traveltime_corrections = np.zeros_like(target_traveltimes)
        else:
            traveltime_corrections = np.zeros_like(target_traveltimes)
        return (source_elevations, true_traveltimes, target_traveltimes,
                traveltime_corrections, uphole_correction_method)

    def set_uphole_correction_method(self, uphole_correction_method):
        _, uphole_correction_method_list = align_args(self.survey_list, uphole_correction_method)
        res = [self._process_survey_uphole_correction_method(survey, uphole_correction_method)
               for survey, uphole_correction_method in zip(self.survey_list, uphole_correction_method_list)]
        res = list(zip(*res))

        self.source_elevations = torch.tensor(np.concatenate(res[0]), dtype=torch.float32)
        self.true_traveltimes = torch.tensor(np.concatenate(res[1]), dtype=torch.float32)
        self.target_traveltimes = torch.tensor(np.concatenate(res[2]), dtype=torch.float32)
        self.traveltime_corrections = torch.tensor(np.concatenate(res[3]), dtype=torch.float32)
        self.uphole_correction_method_list = list(res[4])

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

    # Loader creation

    def create_train_loader(self, batch_size, n_epochs, shuffle=True, drop_last=True, device=None, bar=True):
        train_tensors = [self.source_coords, self.source_elevations, self.source_indices, self.source_weights,
                         self.receiver_coords, self.receiver_elevations, self.receiver_indices, self.receiver_weights,
                         self.grid_indices, self.grid_weights, self.target_traveltimes]
        loader = TensorDataLoader(*train_tensors, batch_size=batch_size, n_epochs=n_epochs,
                                  shuffle=shuffle, drop_last=drop_last, device=device)
        return tqdm(loader, desc="Iterations of model fitting", disable=not bar)

    def create_predict_loader(self, batch_size, n_epochs=1, shuffle=False, drop_last=False, device=None, bar=True):
        pred_tensors = [self.source_coords, self.source_elevations, self.source_indices, self.source_weights,
                        self.receiver_coords, self.receiver_elevations, self.receiver_indices, self.receiver_weights,
                        self.grid_indices, self.grid_weights, self.traveltime_corrections]
        loader = TensorDataLoader(*pred_tensors, batch_size=batch_size, n_epochs=n_epochs,
                                  shuffle=shuffle, drop_last=drop_last, device=device)
        return tqdm(loader, desc="Iterations of model inference", disable=not bar)

    # Evaluation of predictions

    def evaluate(self):
        if not self.has_predictions:
            raise ValueError
        return torch.abs(self.pred_traveltimes - self.true_traveltimes).mean().item()
