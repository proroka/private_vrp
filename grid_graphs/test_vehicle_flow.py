# Standard modules
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.optimize as opt
import scipy.special as spc
# My modules
import utilities
import utilities_field

set_seed = True
save_plots = False
plot_on = False

# Global settings
grid_size = 5
cell_size = 10 # meters
num_nodes = grid_size**2
epsilon = .2
vehicle_density = 0.1
num_vehicles = int(num_nodes * vehicle_density)
num_passengers = num_vehicles

# Load graph
graph = nx.grid_2d_graph(grid_size, grid_size, periodic=False, create_using=None)

print nx.info(graph)
print "Number of vehicles: ", num_vehicles

# Initialization
if set_seed: 
    np.random.seed(1234)
vehicle_node_ind = np.random.choice(np.arange(num_nodes), size=num_vehicles, replace=False)
passenger_node_ind = np.random.choice(np.arange(num_nodes), size=num_passengers, replace=False)
vehicle_nodes = utilities.index_to_location(graph, vehicle_node_ind)
passenger_nodes = utilities.index_to_location(graph, passenger_node_ind)


# Initialize probability field
field = utilities_field.create_probability_field(graph, passenger_node_ind)
max_force = utilities_field.max_field_force(field)
if plot_on:
    utilities_field.plot_field(field, max_force, graph, cell_size, passenger_node_ind)


def run_vehicle_flow(vehicle_nodes, passenger_nodes, graph, field):
    steps = 0
    waiting_time = []
    occupied_vehicle = np.zeros(num_vehicles)
    passenger_nodes_set = set(tuple(row) for row in passenger_nodes)
    # While there are still waiting passengers
    while passenger_nodes_set: 
    #for i in range(10):
        print "Waiting passengers: ", len(passenger_nodes_set)
        steps += 1 # time
        # Each vehicle takes 1 step
        for i, vnode in enumerate(vehicle_nodes):
            # Don't do anything for occupied vehicles
            if occupied_vehicle[i]:
                continue
            pvalues = field[tuple(vnode)]
            heading = np.random.choice(range(4), p=pvalues)
            # Add the offset direction to vehicle pos
            vehicle_nodes[i,:] = vnode + utilities_field.direction_offset[heading]
            vnode_t = tuple(vehicle_nodes[i,:])
            # Check if passenger is picked up
            if vnode_t in passenger_nodes_set:
                print "Vehicle %d occupied" % i
                # Vehicle i is occupied, store pickup time, remove passenger
                occupied_vehicle[i] = True
                passenger_nodes_set.remove(vnode_t)
                waiting_time.append(steps)
                # Update field
                passenger_node_ind = utilities.location_to_index(graph, passenger_nodes_set)
                field = utilities_field.create_probability_field(graph, passenger_node_ind)
                max_force = utilities_field.max_field_force(field)
                if passenger_nodes_set and plot_on:
                    utilities_field.plot_field(field, max_force, graph, cell_size, passenger_node_ind)

    return waiting_time


waiting_time = run_vehicle_flow(vehicle_nodes, passenger_nodes, graph, field) 
print waiting_time

    





