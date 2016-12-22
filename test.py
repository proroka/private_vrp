

# Standard modules
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.optimize as opt
import utilities

verbose = True
plot_on = False
set_seed = True



# Global settings
vehicle_density = 0.2
grid_size = 3
num_nodes = grid_size**2
num_vehicles = num_nodes #int(num_nodes * vehicle_density)


# Load graph
graph = nx.grid_2d_graph(grid_size, grid_size, periodic=False, create_using=None)
 

# Initialization
if set_seed: 
    np.random.seed(1234)
vehicle_node_init = np.random.choice(np.arange(num_nodes), size=num_vehicles, replace=False)
vehicle_node_init_b = np.arange(num_nodes)

noise = 2.0

# Add noise to vehicle initial node positions
vehicle_node_init_noisy = utilities.add_noise(graph, vehicle_node_init, noise, grid_size)



print "Original nodes:\n", vehicle_node_init
print "Original nodes:\n", vehicle_node_init_b

print "Noisy nodes:\n", vehicle_node_init_noisy





