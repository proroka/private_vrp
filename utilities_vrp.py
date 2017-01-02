# Standard modules
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.optimize as opt
import utilities


def run_vrp_allocation(graph, vehicle_node_ind, passenger_node_ind):
    num_vehicles = len(vehicle_node_ind)
    num_passengers = len(passenger_node_ind)
    allocation_cost = utilities.get_allocation_cost(graph, num_vehicles, num_passengers, vehicle_node_ind, passenger_node_ind)
    row_ind, col_ind = opt.linear_sum_assignment(allocation_cost)
    final_cost = allocation_cost[row_ind, col_ind].sum()
    vehicle_node_final = col_ind
    waiting_time = allocation_cost[row_ind, col_ind]
    #print "Total allocation cost, opt: ", final_cost
    return waiting_time

def run_rand_allocation(graph, vehicle_node_ind, passenger_node_ind):
    num_vehicles = len(vehicle_node_ind)
    num_passengers = len(passenger_node_ind)
    allocation_cost = utilities.get_allocation_cost(graph, num_vehicles, num_passengers, vehicle_node_ind, passenger_node_ind)
    row_ind_rand = np.random.choice(np.arange(num_vehicles), size=num_vehicles, replace=False)
    col_ind_rand = np.random.choice(np.arange(num_vehicles), size=num_vehicles, replace=False)
    vehicle_node_final_rand = col_ind_rand
    final_cost_rand = allocation_cost[row_ind_rand, col_ind_rand].sum()
    #print "Total allocation cost, rand: ", final_cost_rand
    waiting_time = allocation_cost[row_ind_rand, col_ind_rand]
    return waiting_time