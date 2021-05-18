import numpy as np
import networkx as nx
from numba import njit


@njit(nogil=True)
def get_closest_index_by_val(val, array):
    return np.array([np.argmin(np.abs(v - array)) for v in val])


@njit(nogil=True)
def interpolate_indices(x0, y0, x1, y1, x):
    return (y1 * (x - x0) + y0 * (x1 - x)) // (x1 - x0)


@njit(nogil=True)
def create_edges(semblance, times, velocities, start_velocity_range, end_velocity_range, max_vel_step,
                 n_times, n_velocities):
    start_nodes = []
    end_nodes = []
    weights = []

    # Switch from time and velocity values to their indices in semblance
    # to further use them as node identifiers in the graph
    times_ix = np.linspace(0, len(times) - 1, n_times).astype(np.int64)
    start_vel_min_ix, start_vel_max_ix = get_closest_index_by_val(start_velocity_range, velocities)
    end_vel_min_ix, end_vel_max_ix = get_closest_index_by_val(end_velocity_range, velocities)
    start_vels_ix = interpolate_indices(times_ix[0], start_vel_min_ix, times_ix[-1], end_vel_min_ix, times_ix)
    end_vels_ix = interpolate_indices(times_ix[0], start_vel_max_ix, times_ix[-1], end_vel_max_ix, times_ix)

    # Instead of running the path search for each of starting nodes iteratively, create an auxiliary node,
    # connected to all of them, and run the search from it
    start_node = (-1, 0)
    prev_nodes = [start_node]
    for time_ix, start_vel_ix, end_vel_ix in zip(times_ix, start_vels_ix, end_vels_ix):
        curr_vels_ix = np.unique(np.linspace(start_vel_ix, end_vel_ix, n_velocities).astype(np.int64))
        curr_nodes = [(time_ix, vel_ix) for vel_ix in curr_vels_ix]
        for prev_time_ix, prev_vel_ix in prev_nodes:
            for curr_time_ix, curr_vel_ix in curr_nodes:
                # Connect two nodes only if:
                # 1. they are a starting node and an auxilliary one
                # 2. current velocity is no less then the previous one, but also does not exceed it by more than
                #    max_vel_step, determined by max_acceleration provided
                if not ((prev_time_ix == -1) or (prev_vel_ix <= curr_vel_ix <= prev_vel_ix + max_vel_step)):
                    continue

                # Calculate the edge weight: sum of (1 - semblance_value) for each value along the path between nodes
                times_indices = np.arange(prev_time_ix + 1, curr_time_ix + 1)
                velocity_indices = interpolate_indices(prev_time_ix, prev_vel_ix, curr_time_ix, curr_vel_ix,
                                                       times_indices)
                weight = len(times_indices)
                for ti, vi in zip(times_indices, velocity_indices):
                    weight -= semblance[ti, vi]

                start_nodes.append((prev_time_ix, prev_vel_ix))
                end_nodes.append((curr_time_ix, curr_vel_ix))
                weights.append(weight)
        prev_nodes = curr_nodes

    edges = (start_nodes, end_nodes, weights)
    return edges, start_node, curr_nodes


def create_graph(semblance, times, velocities, start_velocity_range, end_velocity_range, max_vel_step,
                 n_times, n_velocities):
    edges, start_node, end_nodes = create_edges(semblance, times, velocities, start_velocity_range,
                                                end_velocity_range, max_vel_step, n_times, n_velocities)
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(zip(*edges))
    return graph, start_node, end_nodes


def calculate_stacking_velocity(semblance, times, velocities, start_velocity_range, end_velocity_range,
                                max_acceleration=None, n_times=25, n_velocities=25):
    start_velocity_range = np.array(start_velocity_range)
    end_velocity_range = np.array(end_velocity_range)

    # Calculate maximal velocity growth (in samples) between two adjacent timestamps,
    # for which graph nodes are created
    total_time = (times[-1] - times[0]) / 1000  # from ms to s
    if max_acceleration is None:
        max_acceleration = 2 * (np.mean(end_velocity_range) - np.mean(start_velocity_range)) / total_time
    max_vel_step = np.ceil((max_acceleration * total_time / n_times) / np.mean(velocities[1:] - velocities[:-1]))

    # Create a graph and find paths with maximal semblance sum along them to all reachable nodes
    graph, start_node, end_nodes = create_graph(semblance, times, velocities, start_velocity_range,
                                                end_velocity_range, max_vel_step, n_times, n_velocities)
    paths = nx.shortest_path(graph, source=start_node, weight="weight")

    # Select only paths to the nodes at the last timestamp and choose the optimal one
    path_weights = [(paths[end_node], nx.path_weight(graph, paths[end_node], weight="weight"))
                    for end_node in end_nodes if end_node in paths]
    if not path_weights:
        raise ValueError("No path was found for given parameters")
    path, metric = min(path_weights, key=lambda x: x[1])

    # Remove the first auxiliary node from the path and calculate mean semblance value along it
    path = np.array(path)[1:]
    metric = 1 - metric / len(times)
    return times[path[:, 0]], velocities[path[:, 1]], metric
