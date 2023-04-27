import math

import numpy as np
import rustworkx as rx
from numba import njit, prange

from .stacking_velocity import StackingVelocity
from ..utils import to_list, interpolate
from ..utils.interpolation.univariate import _times_to_indices


@njit(nogil=True)
def get_path_sum(spectrum_data, times_ix, vels_ix):
    res = 0
    for time_ix, vel_ix in zip(times_ix, vels_ix):
        prev_vel_ix = math.floor(vel_ix)
        next_vel_ix = math.ceil(vel_ix)
        weight = next_vel_ix - vel_ix
        res += spectrum_data[time_ix, prev_vel_ix] * weight + spectrum_data[time_ix, next_vel_ix] * (1 - weight)
    return res


@njit(nogil=True)
def create_edges_between_layers(i, j, spectrum_data, spectrum_times, spectrum_velocities, node_times_ix,
                                node_velocities_ix, node_biases, acceleration_bounds, max_spectrum):
    start_time_ix = node_times_ix[i]
    end_time_ix = node_times_ix[j]
    times_ix = np.arange(start_time_ix, end_time_ix)
    n_times = len(times_ix)
    dt = (spectrum_times[end_time_ix] - spectrum_times[start_time_ix]) / 1000

    start_velocities_ix = node_velocities_ix[i]
    end_velocities_ix = node_velocities_ix[j]
    start_velocities = interpolate(start_velocities_ix, np.arange(len(spectrum_velocities)), spectrum_velocities, 0, 0)
    end_velocities = interpolate(end_velocities_ix, np.arange(len(spectrum_velocities)), spectrum_velocities, 0, 0)

    edges = []
    for start_vel_pos, (start_vel, start_vel_ix) in enumerate(zip(start_velocities, start_velocities_ix)):
        for end_vel_pos, (end_vel, end_vel_ix) in enumerate(zip(end_velocities, end_velocities_ix)):
            acceleration = (end_vel - start_vel) / dt
            if (acceleration < acceleration_bounds[0]) | (acceleration > acceleration_bounds[1]):
                continue
            vels_ix = np.linspace(start_vel_ix, end_vel_ix, n_times + 1)[:-1]
            weight = n_times * max_spectrum - get_path_sum(spectrum_data, times_ix, vels_ix)
            edges.append((node_biases[i] + start_vel_pos, node_biases[j] + end_vel_pos, weight))
    return edges


@njit(nogil=True, parallel=True)
def create_edges(spectrum_data, spectrum_times, spectrum_velocities, node_times_ix, node_velocities_ix,
                 node_biases, max_n_skips, acceleration_bounds):
    max_spectrum = spectrum_data.max()
    edges = [[(1, 1, -1.0)] for _ in range((max_n_skips + 1) * (len(node_times_ix) - 1))]
    for ix in prange(len(edges)):
        i = ix // (max_n_skips + 1)
        j = i + 1 + ix % (max_n_skips + 1)
        if j >= len(node_times_ix):
            continue
        edges[ix] = create_edges_between_layers(i, j, spectrum_data, spectrum_times, spectrum_velocities,
                                                node_times_ix, node_velocities_ix, node_biases,
                                                acceleration_bounds, max_spectrum)
    edges = [layer_edges for layer_edges in edges if (len(layer_edges) > 1) or (layer_edges[0][-1] > 0)]
    edges.append([(0, i + 1, 0.0) for i in range(len(node_velocities_ix[0]))])
    edges.append([(node_biases[-1] + i, node_biases[-1] + len(node_velocities_ix[-1]), 0.0)
                  for i in range(len(node_velocities_ix[-1]))])
    return edges


def calculate_stacking_velocity(spectrum, init=None, bounds=None, relative_margin=0.2, acceleration_bounds="auto",
                                times_step=100, max_offset=5000, hodograph_correction_step=10, max_n_skips=2):
    spectrum_data = spectrum.velocity_spectrum
    spectrum_times = np.asarray(spectrum.times, dtype=np.float32)
    spectrum_velocities = np.asarray(spectrum.velocities, dtype=np.float32)

    node_times = np.arange(spectrum_times[0], spectrum_times[-1], times_step)
    node_times_ix = _times_to_indices(node_times, spectrum_times, round=True).astype(np.int32)
    node_times_ix[-1] = len(spectrum_times) - 1
    node_times = spectrum_times[node_times_ix]

    if bounds is not None:
        bounds = to_list(bounds)
        if len(bounds) != 2 or not all(isinstance(bound, StackingVelocity) for bound in bounds):
            raise ValueError
        min_vel_bound = bounds[0](node_times)
        max_vel_bound = bounds[1](node_times)
    else:
        if not isinstance(init, StackingVelocity):
            raise ValueError
        center_vel = init(node_times)
        min_vel_bound = center_vel * (1 - relative_margin)
        max_vel_bound = center_vel * (1 + relative_margin)

    min_spectrum_velocity = spectrum_velocities.min()
    max_spectrum_velocity = spectrum_velocities.max()
    min_vel_bound = np.clip(min_vel_bound, min_spectrum_velocity, max_spectrum_velocity)
    max_vel_bound = np.clip(max_vel_bound, min_spectrum_velocity, max_spectrum_velocity)
    max_vel_bound = np.maximum(min_vel_bound, max_vel_bound)

    if acceleration_bounds is None:
        acceleration_bounds = [0, np.inf]
    elif acceleration_bounds == "auto":
        min_vel_accelerations = 1000 * np.diff(min_vel_bound) / np.diff(node_times)
        max_vel_accelerations = 1000 * np.diff(min_vel_bound) / np.diff(node_times)
        min_acceleration = min(min_vel_accelerations.min(), max_vel_accelerations.min())
        max_acceleration = min(min_vel_accelerations.max(), max_vel_accelerations.max())
        acceleration_bounds = [min_acceleration / 2, max_acceleration * 2]
    acceleration_bounds = np.array(acceleration_bounds, dtype=np.float32)
    if len(acceleration_bounds) != 2:
        raise ValueError

    final_hodograph_time_max = np.sqrt(node_times**2 + (max_offset * 1000 / min_vel_bound)**2)
    final_hodograph_time_min = np.sqrt(node_times**2 + (max_offset * 1000 / max_vel_bound)**2)
    node_velocities = []
    node_velocities_ix = []
    for time, start, end in zip(node_times, final_hodograph_time_min, final_hodograph_time_max):
        n_vels = int((end - start) // hodograph_correction_step) + 1
        vels = np.sqrt(max_offset**2 * 1000**2 / (np.linspace(start, end, n_vels)[::-1]**2 - time**2))
        node_velocities.append(vels)
        node_velocities_ix.append(_times_to_indices(vels, spectrum_velocities, round=False))
    node_velocities_ix = tuple(node_velocities_ix)

    # Create a graph and find paths with maximal velocity spectrum sum along them to all reachable nodes
    n_nodes = 2 + sum(len(node_vels) for node_vels in node_velocities)
    node_biases = np.cumsum([1] + [len(node_vels) for node_vels in node_velocities[:-1]])
    edges = create_edges(spectrum_data, spectrum_times, spectrum_velocities, node_times_ix, node_velocities_ix,
                         node_biases, max_n_skips, acceleration_bounds)
    graph = rx.PyDiGraph()
    for layer_edges in edges:
        graph.extend_from_weighted_edge_list(layer_edges)
    paths_dict = rx.dijkstra_shortest_paths(graph, 0, n_nodes - 1, weight_fn=float)
    path = np.array(paths_dict[n_nodes - 1], dtype=np.int32)[1:-1]
    times_ix = np.searchsorted(node_biases, path, side="right") - 1
    vels_ix = path - node_biases[times_ix]
    sv_times = node_times[times_ix]
    sv_velocities = np.array([node_velocities[tix][vix] for tix, vix in zip(times_ix, vels_ix)])
    return StackingVelocity(sv_times, sv_velocities, coords=spectrum.coords)
