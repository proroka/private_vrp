# Standard modules
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.optimize as opt

BIG_NUMBER = 1e7

# Allocation cost given precomputed route lengths
def get_allocation_cost(route_lengths, vehicle_node_ind, passenger_node_ind):
    num_vehicles = len(vehicle_node_ind)
    num_passengers = len(passenger_node_ind)
    allocation_cost = np.zeros((num_vehicles, num_passengers))
    for i in range(num_vehicles): 
        start = vehicle_node_ind[i]   
        for j in range(num_passengers):
            end = passenger_node_ind[j]
            # Compute cost of shortest path for all possible allocations
            if start not in route_lengths or end not in route_lengths[start]:
                allocation_cost[i,j] = BIG_NUMBER
            else:
                allocation_cost[i,j] = route_lengths[start][end]

    return allocation_cost

# Weighted allocation cost
def get_allocation_cost_graph(graph, num_vehicles, num_passengers, vehicle_node_init, passenger_node_init):
    allocation_cost = np.zeros((num_vehicles, num_passengers))
    for i in range(num_vehicles):
        all_paths = nx.shortest_path_length(graph, source=graph.nodes()[vehicle_node_init[i]], weight='weights')
        
        for j in range(num_passengers):
            # Compute cost of shortest path for all possible allocations
            allocation_cost[i,j] = all_paths[graph.nodes()[passenger_node_init[j]]]

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
    row_ind = np.random.choice(np.arange(num_vehicles), size=num_vehicles, replace=False)
    col_ind = np.random.choice(np.arange(num_passengers), size=num_passengers, replace=False)
    cost = allocation_cost[row_ind, col_ind]

    return cost, row_ind, col_ind

