# Standard modules
import collections
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.optimize as opt
import time

import noise
import probabilistic

BIG_NUMBER = 1e7

# Allocation cost given precomputed route lengths
def get_allocation_cost(route_lengths, vehicle_node_ind, passenger_node_ind):
    assert isinstance(route_lengths, np.ndarray) and len(route_lengths.shape) == 2, 'This function requires a contiguous route length matrix. Use the graph_util.normalize() function.'

    num_vehicles = len(vehicle_node_ind)
    num_passengers = len(passenger_node_ind)
    allocation_cost = np.zeros((num_vehicles, num_passengers))
    for i in range(num_vehicles):
        start = vehicle_node_ind[i]
        for j in range(num_passengers):
            end = passenger_node_ind[j]
            # Compute cost of shortest path for all possible allocations
            allocation_cost[i, j] = route_lengths[start, end]

    return allocation_cost

# Assign vehicles to passengers
def get_routing_assignment(allocation_cost):
    row_ind, col_ind = opt.linear_sum_assignment(allocation_cost)
    cost = allocation_cost[row_ind, col_ind]

    return cost, row_ind, col_ind

# Random assignment
def get_rand_routing_assignment(allocation_cost):
    num_vehicles = allocation_cost.shape[0]
    num_passengers = allocation_cost.shape[1]
    max_allocations = min(num_vehicles, num_passengers)
    row_ind = np.random.choice(np.arange(num_vehicles), size=max_allocations, replace=False)
    col_ind = np.random.choice(np.arange(num_passengers), size=max_allocations, replace=False)
    cost = allocation_cost[row_ind, col_ind]

    return cost, row_ind, col_ind

def get_updated_allocation_cost(vehicle_available, passenger_vehicles, vehicle_distances):
    P = np.array(passenger_vehicles, dtype=np.int32)
    V = np.expand_dims(np.array(vehicle_available, dtype=np.int32), axis=1)
    P = np.broadcast_arrays(np.ones((len(vehicle_available), 1, 1)), P)[1]
    V = np.expand_dims(np.broadcast_arrays(np.ones((1, len(passenger_vehicles))), V)[1], axis=-1)
    Wi = np.expand_dims(np.arange(len(passenger_vehicles)), axis=-1)
    W = np.broadcast_arrays(np.ones((len(vehicle_available), 1, 1)), Wi)[1]
    PV = np.concatenate([P, V], axis=-1)
    distances = vehicle_distances[PV, W]
    allocation_cost = np.mean(np.min(distances, axis=-2), axis=-1)
    return allocation_cost

def get_repeated_routing_assignment(route_lengths, vehicle_pos_noisy, passenger_node_ind, epsilon, noise_model, nearest_neighbor_searcher, graph, repeat=1,
                                    previous_row_ind=None, previous_col_ind=None, previous_repeat=None, previous_vehicle_distances=None):
    assert isinstance(route_lengths, np.ndarray) and len(route_lengths.shape) == 2, 'This function requires a contiguous route length matrix. Use the graph_util.normalize() function.'

    if len(vehicle_pos_noisy) == 0:
        return 0., [], [], None

    # Precompute all vehicles random positions and the distance from every sample to every passenger.
    if previous_vehicle_distances is None:
        num_samples = 100
        vehicle_sample_distances = []
        for p in vehicle_pos_noisy:
            point_locations = np.ones((num_samples, 2)) * p
            vehicle_node_ind_noisy, _ = noise.add_noise(point_locations, nearest_neighbor_searcher, epsilon, noise_model)
            V = np.broadcast_arrays(np.ones((passenger_node_ind.shape[0], 1)), vehicle_node_ind_noisy)[1]
            P = np.broadcast_arrays(np.ones((vehicle_node_ind_noisy.shape[0], 1)), passenger_node_ind)[1]
            vehicle_sample_distances.append(route_lengths[V, P.T])
        vehicle_sample_distances = np.array(vehicle_sample_distances)
    else:
        vehicle_sample_distances = previous_vehicle_distances

    if len(passenger_node_ind) == 0:
        return 0., [], [], vehicle_sample_distances

    # Holds the mapping from passenger to allocated vehicles.
    already_allocated_vehicles = []
    for _ in passenger_node_ind:
        already_allocated_vehicles.append([])
    if previous_row_ind and previous_col_ind and previous_repeat is not None:
        for v, p in zip(previous_row_ind, previous_col_ind):
            already_allocated_vehicles[p].append(v)
        vehicle_indices = list(set(range(len(vehicle_pos_noisy))) - set(previous_row_ind))
        assert previous_repeat < repeat
    else:
        # Holds the indices of vehicles that are still available.
        vehicle_indices = range(len(vehicle_pos_noisy))
        previous_repeat = -1

    for _ in range(repeat - previous_repeat):
        # Compute cost matrix.
        allocation_cost = get_updated_allocation_cost(vehicle_indices, already_allocated_vehicles, vehicle_sample_distances)
        cost, row_ind, col_ind = get_routing_assignment(allocation_cost)
        # Update lists.
        for v, p in zip(row_ind, col_ind):
            already_allocated_vehicles[p].append(vehicle_indices[v])
        newly_allocated_vehicles = set(vehicle_indices[i] for i in row_ind)
        vehicle_indices = list(set(vehicle_indices) - newly_allocated_vehicles)
        if not vehicle_indices:
            break

    # Report final allocation.
    row_ind = []
    col_ind = []
    for p, vehicles in enumerate(already_allocated_vehicles):
        for v in vehicles:
            row_ind.append(v)
            col_ind.append(p)
    return cost, row_ind, col_ind, vehicle_sample_distances

def compute_waiting_times(route_lengths, vehicle_node_ind, passenger_node_ind, row_ind, col_ind):
    waiting_times = collections.defaultdict(lambda: np.inf)
    for v, p in zip(row_ind, col_ind):
        waiting_times[p] = min(waiting_times[p], route_lengths[vehicle_node_ind[v]][passenger_node_ind[p]])
    w = []
    for p in range(len(passenger_node_ind)):
        w.append(float(waiting_times[p]))
    return w
