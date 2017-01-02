# Standard modules
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.optimize as opt
import scipy.special as spc
# My modules
import utilities
import utilities_field
import utilities_vrp

set_seed = True
save_plots = False
plot_on = False
save_fig = True

# Global settings
grid_size = 30
cell_size = 10 # meters
num_nodes = grid_size**2
epsilon = .2
vehicle_density = 0.2
num_vehicles = int(num_nodes * vehicle_density)
num_passengers = num_vehicles


# Load graph
graph = nx.grid_2d_graph(grid_size, grid_size, periodic=False, create_using=None)
print nx.info(graph)
print "Number of vehicles: ", num_vehicles

if set_seed: 
    np.random.seed(1234)

# Initialization
vehicle_node_ind = np.random.choice(np.arange(num_nodes), size=num_vehicles, replace=False)
passenger_node_ind = np.random.choice(np.arange(num_nodes), size=num_passengers, replace=False)
vehicle_nodes = utilities.index_to_location(graph, vehicle_node_ind)
passenger_nodes = utilities.index_to_location(graph, passenger_node_ind)

# Run vehicles on flow field
alpha = 10. # the higher, the more deterministic (greedy) the flow
field = utilities_field.create_probability_field(graph, passenger_node_ind, alpha)
max_force = utilities_field.max_field_force(field)
if plot_on:
    utilities_field.plot_field(field, max_force, graph, cell_size, passenger_node_ind)
waiting_time_flow = utilities_field.run_vehicle_flow(vehicle_nodes, passenger_nodes, graph, field, alpha) 


# Run VRP
vehicle_node_ind_noisy = utilities.add_noise(graph, vehicle_node_ind, epsilon, grid_size, cell_size)
waiting_time_vrpopt = utilities_vrp.run_vrp_allocation(graph, vehicle_node_ind, passenger_node_ind)
waiting_times_vrpsub = utilities_vrp.run_vrp_allocation(graph, vehicle_node_ind_noisy, passenger_node_ind)
waiting_time_vrprand = utilities_vrp.run_rand_allocation(graph, vehicle_node_ind, passenger_node_ind)


# Plot
max_value = np.max(np.stack([waiting_time_flow, waiting_time_vrpopt, waiting_times_vrpsub, waiting_time_vrprand]))

#max_value = np.max(np.stack([waiting_time_flow]))

num_bins = 30
bins = np.linspace(-0.5, max_value+0.5, num_bins+1)
a = 0.5
plt.figure()
plt.hist(waiting_time_flow, bins=bins, color='cyan', alpha=a)
plt.xlim([-1, max_value+1])
if save_fig:
    plt.savefig('figures/times_flow.eps',format='eps')
plt.figure()
plt.hist(waiting_time_vrpopt, bins=bins, color='blue', alpha=a)
plt.xlim([-1, max_value+1])
if save_fig:
    plt.savefig('figures/times_vrpopt.eps',format='eps')

plt.figure()
plt.hist(waiting_times_vrpsub, bins=bins, color='green', alpha=a)
plt.xlim([-1, max_value+1])
if save_fig:
    plt.savefig('figures/times_vrpsub.eps',format='eps')

plt.figure()
plt.hist(waiting_time_vrprand, bins=bins, color='red', alpha=a)
plt.xlim([-1, max_value+1])
if save_fig:
    plt.savefig('figures/times_vrprand.eps',format='eps')

plt.show()






