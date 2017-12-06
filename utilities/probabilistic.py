# Standard modules
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.optimize as opt
# Import my modules
import noise as util_noise
import graph as util_graph

BIG_NUMBER = 1e7


# Allocation cost given precomputed route lengths and noisy vehicle positions
# In: route_lengths is ndarray (normalized)
def get_allocation_cost_noisy(route_lengths, vehicle_pos_noisy, passenger_node_ind, epsilon, noise_model, nearest_neighbor_searcher, graph):
    assert isinstance(route_lengths, np.ndarray) and len(route_lengths.shape) == 2, (
        'This function requires a contiguous route length matrix. Use the graph_util.normalize() function.')

    num_vehicles = vehicle_pos_noisy.shape[0]
    num_passengers = len(passenger_node_ind)
    allocation_cost = np.zeros((num_vehicles, num_passengers))
    
    #allocation_cost = np.ones((num_vehicles, num_passengers)) 

    for i in range(num_vehicles): 
        # Compute nearest nodes and corresponding probabilities for noisy position
        vehicle_node_ind, node_prob = compute_nearest_nodes(vehicle_pos_noisy[i,:], epsilon, noise_model, nearest_neighbor_searcher, graph)

        for j in range(num_passengers):
            # end node
            end = passenger_node_ind[j] 
            # for all nodes in feasible set
            for k in range(len(vehicle_node_ind)):
                start = vehicle_node_ind[k]
                allocation_cost[i, j] += route_lengths[start, end] * node_prob[k]
                    


    return allocation_cost

def compute_nearest_nodes(vehicle_pos_noisy, epsilon, noise_model, nearest_neighbor_searcher, graph):
    
    num_samples = 200
    point_locations = np.ones((num_samples,2)) * vehicle_pos_noisy

    # Find nearest nodes around noisy point, given noise model and scale parameter
    vehicle_node_ind_noisy, vehicle_node_pos_noisy = util_noise.add_noise(point_locations, nearest_neighbor_searcher, epsilon, noise_model)

    # Count occurences of graph nodes
    count = dict()
    for v in vehicle_node_ind_noisy:
        if v in count:
            count[v] += 1
        else:
            count[v] = 1
    key_node, count_node = zip(*count.items())
    # Compute probability
    vehicle_node_ind_noisy_unique = np.array(key_node)
    probabilities = np.array(count_node) / float(sum(count_node))

    # Compute probabilities of those nodes
    # node_xys = util_graph.GetNodePositions(graph, vehicle_node_ind_noisy)
    # vec = node_xys - np.ones(node_xys.shape)*vehicle_pos_noisy
    # dist_vec = np.linalg.norm(vec, axis=1)
    # probabilities = 1./2. * epsilon * np.exp(-dist_vec * epsilon)
    # normalization_const = 1. / np.sum(probabilities)

    return vehicle_node_ind_noisy_unique, probabilities








