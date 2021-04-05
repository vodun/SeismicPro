import numpy as np
import networkx as nx
from numba import njit


@njit(nogil=True)
def get_index_by_val(val, array):
    return np.array([np.argmin(np.abs(v - array)) for v in val])


@njit(nogil=True)
def interpolate_indices(x0, y0, x1, y1, x):
    return (y1 * (x - x0) + y0 * (x1 - x)) // (x1 - x0)


@njit(nogil=True)
def create_edges(semblance, times, velocities, v0_range, vn_range, n_times, n_vels, gap):
    start_nodes = []
    end_nodes = []
    weights = []

    times_ix = np.linspace(0, len(times) - 1, n_times).astype(np.int64)
    v0_min_ix, v0_max_ix = get_index_by_val(v0_range, velocities)
    vn_min_ix, vn_max_ix = get_index_by_val(vn_range, velocities)
    v_start_ix = interpolate_indices(times_ix[0], v0_min_ix, times_ix[-1], vn_min_ix, times_ix)
    v_end_ix = interpolate_indices(times_ix[0], v0_max_ix, times_ix[-1], vn_max_ix, times_ix)

    start_node = (-1, 0)
    prev_nodes = [start_node]
    for t, v_start, v_end in zip(times_ix, v_start_ix, v_end_ix):
        curr_nodes = [(t, v) for v in np.unique(np.linspace(v_start, v_end, n_vels).astype(np.int64))]
        for prev_time_ix, prev_vel_ix in prev_nodes:
            for curr_time_ix, curr_vel_ix in curr_nodes:
                if prev_time_ix == -1:
                    start_nodes.append((prev_time_ix, prev_vel_ix))
                    end_nodes.append((curr_time_ix, curr_vel_ix))
                    weights.append(-semblance[curr_time_ix, curr_vel_ix])
                elif (curr_vel_ix >= prev_vel_ix) and (curr_vel_ix <= prev_vel_ix + gap):
                    times_indices = np.arange(prev_time_ix + 1, curr_time_ix + 1)
                    velocity_indices = interpolate_indices(prev_time_ix, prev_vel_ix, curr_time_ix, curr_vel_ix, times_indices)
                    weight = 0
                    for ti, vi in zip(times_indices, velocity_indices):
                        weight -= semblance[ti, vi]
                    start_nodes.append((prev_time_ix, prev_vel_ix))
                    end_nodes.append((curr_time_ix, curr_vel_ix))
                    weights.append(weight)
        prev_nodes = curr_nodes
    weights = np.array(weights, dtype=np.float64)
    min_weight = weights.min() - 1
    weights -= min_weight
    return start_nodes, end_nodes, weights, min_weight, start_node, curr_nodes


def create_graph(semblance, times, velocities, v0_range, vn_range, n_times, n_vels, gap):
    res = create_edges(semblance, times, velocities, np.array(v0_range), np.array(vn_range), n_times, n_vels, gap)
    *edges, min_weight, start_node, end_nodes = res
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(zip(*edges))
    return graph, min_weight, start_node, end_nodes


def calc_velocity_model(semblance, times, velocities, v0_range=(1400, 1800),
                        vn_range=(2500, 5000), n_times=25, n_vels=25, max_acc=None):
    total_time = (times[-1] - times[0]) / 1000
    if max_acc is None:
        max_acc = 2 * (np.mean(vn_range) - np.mean(v0_range)) / total_time
    gap = np.ceil((max_acc * total_time / n_times) / np.mean(velocities[1:] - velocities[:-1]))
    graph, min_weight, start_node, end_nodes = create_graph(semblance, times, velocities, v0_range,
                                                            vn_range, n_times, n_vels, gap)
    paths = nx.shortest_path(graph, source=start_node, weight="weight")

    path_weights = [(paths[end_node], nx.path_weight(graph, paths[end_node], weight="weight"))
                    for end_node in end_nodes if end_node in paths]
    if not path_weights:
        raise ValueError("No path was found for given parameters")
    path, metric = min(path_weights, key=lambda x: x[1])
    path = np.array(path)[1:]
    path = np.column_stack([times[path[:, 0]], velocities[path[:, 1]]]).tolist()
    metric = -(metric + n_times * min_weight) / (len(times) - 1)
    return path, metric
