

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
grid_size = 5
num_nodes = grid_size**2
num_vehicles = int(num_nodes * vehicle_density)


# Load graph
graph = nx.grid_2d_graph(grid_size, grid_size, periodic=False, create_using=None)
 

# Initialization
if set_seed: 
    np.random.seed(1234)
vehicle_node_init = np.random.choice(np.arange(num_nodes), size=num_vehicles, replace=False)


noise = 2.0

# Add noise to vehicle initial node positions
vehicle_node_init_noisy = utilities.add_noise(graph, vehicle_node_init, noise, grid_size)


print vehicle_node_init
print vehicle_node_init_noisy

