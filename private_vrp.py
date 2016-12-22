

# Standard modules
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.optimize as opt
import utilities

verbose = False
plot_on = False
set_seed = True


# Global settings
vehicle_density = 0.2
grid_size = 80
num_nodes = grid_size**2
num_vehicles = int(num_nodes * vehicle_density)
num_passengers = num_vehicles
noise = 2.0


# Load graph
graph = nx.grid_2d_graph(grid_size, grid_size, periodic=False, create_using=None)
#graph = nx.watts_strogatz_graph(num_nodes,3,0.1)

print nx.info(graph)
print "Number of vehicles: ", num_vehicles


if plot_on:
    pos = dict(zip(graph.nodes(), graph.nodes())) # Only works for grid graph
    nx.draw_networkx(graph, pos=pos, with_labels=False, node_size=250, node_color='lightblue')
    plt.show()


# Initialization
if set_seed: 
    np.random.seed(1234)
vehicle_node_init = np.random.choice(np.arange(num_nodes), size=num_vehicles, replace=False)
passenger_node_init = np.random.choice(np.arange(num_nodes), size=num_passengers, replace=False)

# Add noise to vehicle initial node positions
vehicle_node_init_noisy = utilities.add_noise(graph, vehicle_node_init, noise, grid_size)

if verbose:
    print "Vehicles, init: ", vehicle_node_init
    print "Passengers:     ", passenger_node_init



# Compute cost matrix (vehicle to passenger pairings)
allocation_cost = np.zeros((num_vehicles, num_passengers))
for i in range(num_vehicles):
    all_paths = nx.shortest_path_length(graph, source=graph.nodes()[vehicle_node_init[i]], weight=None)
    for j in range(num_passengers):
        # Compute cost of shortest path for all possible allocations
        allocation_cost[i,j] = all_paths[graph.nodes()[passenger_node_init[j]]]
         
# Find optimal allocation (Munkres algorithm)
row_ind, col_ind = opt.linear_sum_assignment(allocation_cost)

if verbose:
    print "row: ", row_ind
    print "col: ", col_ind
final_cost = allocation_cost[row_ind, col_ind].sum()
vehicle_node_final = col_ind
print "Total allocation cost: ", final_cost


# Compute noisy cost matrix
allocation_cost_noisy = np.zeros((num_vehicles, num_passengers))
for i in range(num_vehicles):
    all_paths = nx.shortest_path_length(graph, source=graph.nodes()[vehicle_node_init_noisy[i]], weight=None)
    for j in range(num_passengers):
        # Compute cost of shortest path for all possible allocations
        allocation_cost_noisy[i,j] = all_paths[graph.nodes()[passenger_node_init[j]]]
# Find sub-optimal allocation (Munkres algorithm)
row_ind_noisy, col_ind_noisy = opt.linear_sum_assignment(allocation_cost_noisy)
# Cost of noisy allocation (should be hgher than opt.)
final_cost = allocation_cost[row_ind_noisy, col_ind_noisy].sum()
vehicle_node_final_noisy = col_ind_noisy
print "Total allocation cost, noisy: ", final_cost


# Distribution of waiting times
opt_waiting_times = allocation_cost[row_ind, col_ind]
plt.figure()
plt.hist(opt_waiting_times, bins=20, color='green')

subopt_waiting_times = allocation_cost[row_ind_noisy, col_ind_noisy]
plt.figure()
plt.hist(subopt_waiting_times, bins=30, color='red')

plt.show()
















