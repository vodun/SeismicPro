"""Implements an algorithm for stacking velocity computation"""

import numpy as np
import networkx as nx
from numba import njit


@njit(nogil=True)
def get_closest_index_by_val(val, array):
    """Return indices of the array elements, closest to the values from `val`."""
    return np.array([np.argmin(np.abs(v - array)) for v in val])


@njit(nogil=True)
def interpolate_indices(x0, y0, x1, y1, x):
    """Linearly interpolate an int-valued function between points (`x0`, `y0`) and (`x1`, `y1`) and calculate its
    values at `x`."""
    return (y1 * (x - x0) + y0 * (x1 - x)) // (x1 - x0)


@njit(nogil=True)
def create_edges(semblance, times, velocities, start_velocity_range, end_velocity_range, max_vel_step,
                 n_times, n_velocities):
    """Return edges of the graph for stacking velocity computation with their weights.

    Parameters
    ----------
    semblance : 2d np.ndarray
        An array with calculated vertical velocity semblance values.
    times : 1d np.ndarray
        Recording time for each seismic trace value for which semblance was calculated. Measured in milliseconds.
    velocities : 1d np.ndarray
        Range of velocity values for which semblance was calculated. Measured in meters/seconds.
    start_velocity_range : 1d np.ndarray with 2 elements
        Valid range for stacking velocity for the first timestamp. Both velocities are measured in meters/seconds.
    end_velocity_range : 1d np.ndarray with 2 elements
        Valid range for stacking velocity for the last timestamp. Both velocities are measured in meters/seconds.
    max_vel_step : int
        Maximal allowed velocity increase for nodes with adjacent times. Measured in samples.
    n_times : int
        The number of evenly spaced points to split time range into to generate graph vertices.
    n_velocities : int
        The number of evenly spaced points to split velocity range into for each time to generate graph vertices.

    Returns
    -------
    edges : tuple with 3 elements: start_nodes, end_nodes, weights
        start_nodes : list of tuples with 2 elements
            Identifiers of start nodes of the edges.
        end_nodes : list of tuples with 2 elements
            Identifiers of end nodes of the edges, corresponding to `start_nodes`. Matches `start_nodes` length.
        weights : list of floats
            Weights of corresponding edges. Matches `start_nodes` length.
    start_node : tuple with 2 elements
        An identifier of an auxiliary node, connected to all of the actual starting nodes to run the path search from.
    end_nodes : list of tuples with 2 elements
        Identifiers of possible end nodes of the path.
    """
    start_nodes = []
    end_nodes = []
    weights = []

    # Switch from time and velocity values to their indices in semblance
    # to further use them as node identifiers in the graph
    times_ix = np.linspace(0, len(times) - 1, n_times).astype(np.int32)
    start_vel_min_ix, start_vel_max_ix = get_closest_index_by_val(start_velocity_range, velocities)
    end_vel_min_ix, end_vel_max_ix = get_closest_index_by_val(end_velocity_range, velocities)
    start_vels_ix = interpolate_indices(times_ix[0], start_vel_min_ix, times_ix[-1], end_vel_min_ix, times_ix)
    end_vels_ix = interpolate_indices(times_ix[0], start_vel_max_ix, times_ix[-1], end_vel_max_ix, times_ix)

    # Instead of running the path search for each of starting nodes iteratively, create an auxiliary node,
    # connected to all of them, and run the search from it
    start_node = (np.int32(-1), np.int32(0))
    prev_nodes = [start_node]
    semb_max = semblance.max() 
    for time_ix, start_vel_ix, end_vel_ix in zip(times_ix, start_vels_ix, end_vels_ix):
        curr_vels_ix = np.unique(np.linspace(start_vel_ix, end_vel_ix, n_velocities).astype(np.int32))
        curr_nodes = [(time_ix, vel_ix) for vel_ix in curr_vels_ix]
        for prev_time_ix, prev_vel_ix in prev_nodes:
            for curr_time_ix, curr_vel_ix in curr_nodes:
                # Connect two nodes only if:
                # 1. they are a starting node and an auxiliary one
                # 2. current velocity is no less then the previous one, but also does not exceed it by more than
                #    max_vel_step, determined by max_acceleration provided
                if not ((prev_time_ix == -1) or (prev_vel_ix <= curr_vel_ix <= prev_vel_ix + max_vel_step)):
                    continue

                # Calculate the edge weight: sum of (1 - semblance_value) for each value along the path between nodes
                times_indices = np.arange(prev_time_ix + 1, curr_time_ix + 1, dtype=np.int32)
                velocity_indices = interpolate_indices(prev_time_ix, prev_vel_ix, curr_time_ix, curr_vel_ix,
                                                       times_indices)
                weight = len(times_indices) * semb_max
                for ti, vi in zip(times_indices, velocity_indices):
                    weight -= semblance[ti, vi]

                start_nodes.append((prev_time_ix, prev_vel_ix))
                end_nodes.append((curr_time_ix, curr_vel_ix))
                weights.append(weight)
        prev_nodes = curr_nodes

    edges = (start_nodes, end_nodes, weights)
    return edges, start_node, curr_nodes


def calculate_stacking_velocity(semblance, times, velocities, start_velocity_range, end_velocity_range,
                                max_acceleration=None, n_times=25, n_velocities=25):
    """Calculate stacking velocity by given semblance.

    Stacking velocity is the value of the seismic velocity obtained from the best fit of the traveltime curve by a
    hyperbola for each timestamp. It is used to correct the arrival times of reflection events in the traces for their
    varying offsets prior to stacking.

    If calculated by semblance, stacking velocity must meet the following conditions:
    1. It should be monotonically increasing
    2. Its gradient should be bounded above to avoid gather stretching after NMO correction
    3. It should pass through local energy maxima on the semblance

    In order for these conditions to be satisfied, the following algorithm is proposed:
    1. Stacking velocity is being found inside a trapezoid whose vertices at first and last time are defined by
       `start_velocity_range` and `end_velocity_range` respectively.
    2. An auxiliary directed graph is constructed so that:
        1. `n_times` evenly spaced points are generated to cover the whole semblance time range. For each of these
           points `n_velocities` evenly spaced points are generated to cover the whole range of velocities inside the
           trapezoid from its left to right border. All these points form a set of vertices of the graph.
        2. An edge from a vertex A to a vertex B exists only if:
            1. Vertex B is located at the very next timestamp after vertex A,
            2. Velocity at vertex B is no less than at A,
            3. Velocity at vertex B does not exceed that of A by a value determined by `max_acceleration` provided.
        3. Edge weight is defined as sum of semblance values along its path.
    3. A path with maximal semblance sum along it between any of starting and ending nodes is found using Dijkstra
       algorithm and is considered to be the required stacking velocity.

    Parameters
    ----------
    semblance : 2d np.ndarray
        An array with calculated vertical velocity semblance values.
    times : 1d np.ndarray
        Recording time for each seismic trace value for which semblance was calculated. Measured in milliseconds.
    velocities : 1d np.ndarray
        Range of velocity values for which semblance was calculated. Measured in meters/seconds.
    start_velocity_range : tuple with 2 elements
        Valid range for stacking velocity for the first timestamp. Both velocities are measured in meters/seconds.
    end_velocity_range : tuple with 2 elements
        Valid range for stacking velocity for the last timestamp. Both velocities are measured in meters/seconds.
    max_acceleration : None or float, defaults to None
        Maximal acceleration allowed for the stacking velocity function. If `None`, equals to
        2 * (mean(end_velocity_range) - mean(start_velocity_range)) / total_time. Measured in meters/seconds^2.
    n_times : int, defaults to 25
        The number of evenly spaced points to split time range into to generate graph vertices.
    n_velocities : int, defaults to 25
        The number of evenly spaced points to split velocity range into for each time to generate graph vertices.

    Returns
    -------
    stacking_times : 1d np.ndarray
        Times for which stacking velocities were picked. Measured in milliseconds.
    stacking_velocities : 1d np.ndarray
        Picked stacking velocities. Matches the length of `stacking_times`. Measured in meters/seconds.
    metric : float
        Sum of semblance values along the stacking velocity path.

    Raises
    ------
    ValueError
        If no path was found for given parameters.
    """
    times = np.asarray(times, dtype=np.float32)
    velocities = np.asarray(velocities, dtype=np.float32)

    start_velocity_range = np.array(start_velocity_range, dtype=np.float32)
    end_velocity_range = np.array(end_velocity_range, dtype=np.float32)

    # Calculate maximal velocity growth (in samples) between two adjacent timestamps,
    # for which graph nodes are created
    total_time = (times[-1] - times[0]) / 1000  # from ms to s
    if max_acceleration is None:
        max_acceleration = 2 * (np.mean(end_velocity_range) - np.mean(start_velocity_range)) / total_time
    max_vel_step = np.ceil((max_acceleration * total_time / n_times) / np.mean(velocities[1:] - velocities[:-1]))
    max_vel_step = np.int32(max_vel_step)

    # Create a graph and find paths with maximal semblance sum along them to all reachable nodes
    edges, start_node, end_nodes = create_edges(semblance, times, velocities, start_velocity_range,
                                                end_velocity_range, max_vel_step, n_times, n_velocities)
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(zip(*edges))
    paths = nx.shortest_path(graph, source=start_node, weight="weight")  # pylint: disable=unexpected-keyword-arg

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
