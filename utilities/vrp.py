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

def run_vrp_allocation(route_lengths, vehicle_node_ind, passenger_node_ind):
    allocation_cost = get_allocation_cost(route_lengths, vehicle_node_ind, passenger_node_ind)
    row_ind, col_ind = opt.linear_sum_assignment(allocation_cost)
    final_cost = allocation_cost[row_ind, col_ind].sum()
    vehicle_node_final = col_ind
    waiting_time = allocation_cost[row_ind, col_ind]
    #print "Total allocation cost, opt: ", final_cost
    return waiting_time

def run_rand_allocation(graph, vehicle_node_ind, passenger_node_ind):
    num_vehicles = len(vehicle_node_ind)
    num_passengers = len(passenger_node_ind)
    allocation_cost = get_allocation_cost(graph, num_vehicles, num_passengers, vehicle_node_ind, passenger_node_ind)
    row_ind_rand = np.random.choice(np.arange(num_vehicles), size=num_vehicles, replace=False)
    col_ind_rand = np.random.choice(np.arange(num_vehicles), size=num_vehicles, replace=False)
    vehicle_node_final_rand = col_ind_rand
    final_cost_rand = allocation_cost[row_ind_rand, col_ind_rand].sum()
    #print "Total allocation cost, rand: ", final_cost_rand
    waiting_time = allocation_cost[row_ind_rand, col_ind_rand]
    return waiting_time