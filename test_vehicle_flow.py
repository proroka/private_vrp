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


# Global settings
grid_size = 5
cell_size = 10 # meters
num_nodes = grid_size**2
epsilon = .2
vehicle_density = 0.1
num_vehicles = int(num_nodes * vehicle_density)
num_passengers = num_vehicles
num_waiting_passengers = num_passengers

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

while num_waiting_passengers > 0:
    # Each vehicle takes 1 step
    for vnode in vehicle_nodes:
        pvalues = field[tuple(vnode)]
        direction = np.random.choice(range(4), p=pvalues)
        print direction
        # add the offset direction to vehicle pos

        # update vehicle nodes array
        
    break

    # Check if passenger is picked up

    # If yes, update passengers and field




