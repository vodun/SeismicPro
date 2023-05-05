import math

import numpy as np
import rustworkx as rx
from numba import njit, prange


@njit(nogil=True)
def get_path_sum(spectrum_data, start_time_ix, start_vel_ix, end_time_ix, end_vel_ix):
    n_times = end_time_ix - start_time_ix + 1
    vels_ix = np.linspace(start_vel_ix, end_vel_ix, n_times)[:-1]
    res = 0
    for i, vel_ix in enumerate(vels_ix):
        time_ix = start_time_ix + i
        prev_vel_ix = math.floor(vel_ix)
        next_vel_ix = math.ceil(vel_ix)
        weight = next_vel_ix - vel_ix
        res += spectrum_data[time_ix, prev_vel_ix] * weight + spectrum_data[time_ix, next_vel_ix] * (1 - weight)
    return res


@njit(nogil=True)  # pylint: disable-next=too-many-function-args
def create_edges_between_layers(spectrum_data, start_time, start_time_ix, start_bias, end_time, end_time_ix, end_bias,
                                start_velocities, start_velocities_ix, end_velocities, end_velocities_ix,
                                acceleration_bounds):
    dt = (end_time - start_time) / 1000
    edges = []
    for start_vel_pos, (start_vel, start_vel_ix) in enumerate(zip(start_velocities, start_velocities_ix)):
        for end_vel_pos, (end_vel, end_vel_ix) in enumerate(zip(end_velocities, end_velocities_ix)):
            acceleration = (end_vel - start_vel) / dt
            if (acceleration < acceleration_bounds[0]) | (acceleration > acceleration_bounds[1]):
                continue
            weight = get_path_sum(spectrum_data, start_time_ix, start_vel_ix, end_time_ix, end_vel_ix)
            edges.append((start_bias + start_vel_pos, end_bias + end_vel_pos, weight))
    return edges


@njit(nogil=True, parallel=True)
def create_edges(spectrum_data, spectrum_times, spectrum_velocities, node_times_ix, node_velocities_ix, node_biases,
                 max_n_skips, acceleration_bounds):
    edges = [[(1, 1, -1.0)] for _ in range((max_n_skips + 1) * (len(node_times_ix) - 1))]
    for ix in prange(len(edges)):  # pylint: disable=not-an-iterable
        i = ix // (max_n_skips + 1)
        j = i + 1 + ix % (max_n_skips + 1)
        if j >= len(node_times_ix):
            continue
        edges[ix] = create_edges_between_layers(i, j, spectrum_data, spectrum_times, spectrum_velocities,
                                                node_times_ix, node_velocities_ix, node_biases, acceleration_bounds)
    edges = [layer_edges for layer_edges in edges if (len(layer_edges) > 1) or (layer_edges[0][-1] > 0)]
    edges.append([(0, i + 1, 0.0) for i in range(len(node_velocities_ix[0]))])
    edges.append([(node_biases[-1] + i, node_biases[-1] + len(node_velocities_ix[-1]), 0.0)
                  for i in range(len(node_velocities_ix[-1]))])
    return edges


# pylint: disable-next=too-many-statements
def calculate_stacking_velocity(spectrum, init=None, bounds=None, relative_margin=0.2, acceleration_bounds="auto",
                                times_step=100, max_offset=5000, hodograph_correction_step=10, max_n_skips=2):
    spectrum_data = spectrum.velocity_spectrum.max() - spectrum.velocity_spectrum
    spectrum_times = np.asarray(spectrum.times, dtype=np.float32)
    spectrum_velocities = np.asarray(spectrum.velocities, dtype=np.float32)

    # Calculate times of graph nodes
    times_step_samples = int(times_step // spectrum.sample_interval)
    node_times_ix = np.arange(0, len(spectrum_times), times_step_samples)
    node_times_ix[-1] = len(spectrum_times) - 1
    node_times = spectrum_times[node_times_ix]

    # Estimate velocity bounds for each time where nodes are placed
    if bounds is not None:
        min_vel_bound = bounds[0](node_times)
        max_vel_bound = bounds[1](node_times)
    else:
        center_vel = init(node_times)
        min_vel_bound = center_vel * (1 - relative_margin)
        max_vel_bound = center_vel * (1 + relative_margin)
    min_spectrum_velocity = spectrum_velocities.min()
    max_spectrum_velocity = spectrum_velocities.max()
    min_vel_bound = np.clip(min_vel_bound, min_spectrum_velocity, max_spectrum_velocity)
    max_vel_bound = np.clip(max_vel_bound, min_spectrum_velocity, max_spectrum_velocity)
    max_vel_bound = np.maximum(min_vel_bound, max_vel_bound)

    # Estimate node velocities for each time
    final_hodograph_time_max = np.sqrt(node_times**2 + (max_offset * 1000 / min_vel_bound)**2)  # ms
    final_hodograph_time_min = np.sqrt(node_times**2 + (max_offset * 1000 / max_vel_bound)**2)  # ms
    spectrum_velocity_indices = np.arange(len(spectrum_velocities))
    node_velocities = []
    node_velocities_ix = []
    for time, start, end in zip(node_times, final_hodograph_time_min, final_hodograph_time_max):
        n_vels = int((end - start) // hodograph_correction_step) + 1
        vels = np.sqrt(max_offset**2 * 1000**2 / (np.linspace(start, end, n_vels)[::-1]**2 - time**2))
        vels_ix = np.interp(vels, spectrum_velocities, spectrum_velocity_indices)
        node_velocities.append(vels)
        node_velocities_ix.append(vels_ix)
    node_velocities = tuple(node_velocities)
    node_velocities_ix = tuple(node_velocities_ix)

    # Calculate allowed acceleration bounds
    if acceleration_bounds is None:
        acceleration_bounds = [0, np.inf]
    elif acceleration_bounds == "auto":
        dt = np.diff(node_times) / 1000
        min_vel_accelerations = np.diff(min_vel_bound) / dt
        max_vel_accelerations = np.diff(max_vel_bound) / dt
        min_acceleration = min(min_vel_accelerations.min(), max_vel_accelerations.min())
        max_acceleration = max(min_vel_accelerations.max(), max_vel_accelerations.max())
        acceleration_bounds = [min_acceleration / 2, max_acceleration * 2]
    acceleration_bounds = np.array(acceleration_bounds, dtype=np.float32)
    if len(acceleration_bounds) != 2:
        raise ValueError("acceleration_bounds must be an array-like with 2 elements")
    if acceleration_bounds[1] <= acceleration_bounds[0]:
        raise ValueError("Upper acceleration bound must greater than the lower one")

    # Create a graph and find the path with maximal velocity spectrum sum along it
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
    times = node_times[times_ix]
    velocities = np.array([node_velocities[tix][vix] for tix, vix in zip(times_ix, vels_ix)])
    return times, velocities
