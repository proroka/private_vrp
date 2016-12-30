

# Standard modules
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.optimize as opt
import utilities

verbose = False
plot_on = True
set_seed = True


# Global settings
vehicle_density = 0.2
grid_size = 20
cell_size = 10 # meters
num_nodes = grid_size**2
num_vehicles = int(num_nodes * vehicle_density)
num_passengers = num_vehicles
epsilon = 0.8


# Load graph
graph = nx.grid_2d_graph(grid_size, grid_size, periodic=False, create_using=None)
#graph = nx.watts_strogatz_graph(num_nodes,3,0.1)

print nx.info(graph)
print "Number of vehicles: ", num_vehicles


if plot_on:
    pos = dict(zip(graph.nodes(), np.array(graph.nodes())*cell_size)) # Only works for grid graph
    nx.draw_networkx(graph, pos=pos, with_labels=False, node_size=250, node_color='lightblue')
    plt.show()


# Initialization
if set_seed: 
    np.random.seed(1234)
vehicle_node_init = np.random.choice(np.arange(num_nodes), size=num_vehicles, replace=False)
passenger_node_init = np.random.choice(np.arange(num_nodes), size=num_passengers, replace=False)

# Add noise to vehicle initial node positions
vehicle_node_init_noisy = utilities.add_noise(graph, vehicle_node_init, epsilon, grid_size, cell_size)

if verbose:
    print "Vehicles, init: ", vehicle_node_init
    print "Passengers:     ", passenger_node_init



# Compute optimal allocation
allocation_cost = utilities.get_allocation_cost(graph, num_vehicles, num_passengers, vehicle_node_init, passenger_node_init)
row_ind, col_ind = opt.linear_sum_assignment(allocation_cost)
final_cost = allocation_cost[row_ind, col_ind].sum()
vehicle_node_final = col_ind
print "Total allocation cost, opt: ", final_cost

# Compute noisy allocation
allocation_cost_noisy = utilities.get_allocation_cost(graph, num_vehicles, num_passengers, vehicle_node_init_noisy, passenger_node_init)
# Find sub-optimal allocation (Munkres algorithm)
row_ind_noisy, col_ind_noisy = opt.linear_sum_assignment(allocation_cost_noisy)
# Cost of noisy allocation (should be hgher than opt.)
final_cost_noisy = allocation_cost[row_ind_noisy, col_ind_noisy].sum()
vehicle_node_final_noisy = col_ind_noisy
print "Total allocation cost, noisy: ", final_cost_noisy

# Random allocation
row_ind_rand = np.random.choice(np.arange(num_vehicles), size=num_vehicles, replace=False)
col_ind_rand = np.random.choice(np.arange(num_vehicles), size=num_vehicles, replace=False)
vehicle_node_final_rand = col_ind_rand
final_cost_rand = allocation_cost[row_ind_rand, col_ind_rand].sum()
print "Total allocation cost, rand: ", final_cost_rand

# Distribution of waiting times
rand_waiting_times = allocation_cost[row_ind_rand, col_ind_rand]
opt_waiting_times = allocation_cost[row_ind, col_ind]
subopt_waiting_times = allocation_cost[row_ind_noisy, col_ind_noisy]

max_value = np.max(np.stack([rand_waiting_times, opt_waiting_times, subopt_waiting_times]))

num_bins = 30
bins = np.linspace(-0.5, max_value+0.5, num_bins+1)
a = 0.5
plt.figure()

plt.hist(rand_waiting_times, bins=bins, color='blue', alpha=a)
plt.hist(opt_waiting_times, bins=bins, color='green', alpha=a)
plt.hist(subopt_waiting_times, bins=bins, color='red', alpha=a)

plt.xlim([-1, max_value+1])
plt.show()
















