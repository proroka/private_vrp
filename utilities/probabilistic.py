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
def get_allocation_cost_noisy(route_lengths, vehicle_pos_noisy, passenger_node_ind, ):
    num_vehicles = len(vehicle_node_ind)
    num_passengers = len(passenger_node_ind)
    allocation_cost = np.zeros((num_vehicles, num_passengers))
    
    for i in range(num_vehicles): 
        start = vehicle_node_ind[i]  
        # Compute node set for noisy position
        vehicle_node_ind, node_prob, const = compute_nearest_nodes(vehicle_pos_noisy, epsilon)

        for j in range(num_passengers):
            # end node
            end = passenger_node_ind[j] 

            # for all nodes in heuristic set
            for start_n in vehicle_node_ind:

                # compute probability of that node given noisy position
                prob_ni = node_prob[start_n]

                # Compute cost of shortest path for all possible allocations
                if start not in route_lengths or end not in route_lengths[start]:
                    cost_nj = BIG_NUMBER
                else:
                    cost_nj = route_lengths[start_n][end] * prob_ni

                allocation_cost[i,j] += cost_nj

            # Normalize the allocation cost
            allocation_cost[i,j] /= const 
    return allocation_cost

def compute_nearest_nodes(vehicle_pos_noisy, epsilon, nearest_neighbor_searcher, graph):
    
    num_samples = 4
    point_locations = np.ones((num_samples,2)) * vehicle_pos_noisy

    # Find nearest nodes around noisy point, given Laplace(epsilon)
    vehicle_node_ind_noisy, vehicle_node_pos_noisy = util_noise.add_noise(point_locations, nearest_neighbor_searcher, epsilon)

    #print vehicle_node_ind_noisy
    #print vehicle_pos_noisy

    # Compute probabilities of those nodes
    probabilities = np.zeros(vehicle_node_ind_noisy.shape[0])
    for i in range(vehicle_node_ind_noisy.shape[0]):
        node_xy = util_graph.GetNodePosition(graph, vehicle_node_ind_noisy[i])

        dist = np.linalg.norm([node_xy - vehicle_pos_noisy])
        print 'dist', dist
        probabilities[i] = 1./(2.*epsilon) * np.exp(-dist / epsilon)

    normalization_const = 1. / np.sum(probabilities)

    return vehicle_node_ind_noisy, probabilities, normalization_const








