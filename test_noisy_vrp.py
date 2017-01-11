# Standard modules
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.optimize as opt
import scipy.special as spc
import pickle

# My modules
import utilities.graph as util_graph
import utilities.noise as util_noise
import utilities.vrp as util_vrp
import utilities.probabilistic as util_prob
import utilities.plot as util_plot
import manhattan.data as manh_data

#-------------------------------------
# Global settings
vehicle_density = 0.4
passenger_density = 0.4
# Noise
epsilon = 0.08

set_seed = False

if set_seed: 
    np.random.seed(1234) #1019

# ---------------------------------------------------
# Load grid graph 

graph = util_graph.create_grid_map(grid_size=20, edge_length=100.)
nearest_neighbor_searcher = util_graph.NearestNeighborSearcher(graph)
route_lengths = manh_data.LoadShortestPathData(graph, must_recompute=True)


# Initialize 
num_nodes = len(graph.nodes())
num_vehicles = int(num_nodes * vehicle_density)
num_passengers = int(num_nodes * passenger_density)

print 'Num vehicles:', num_vehicles
print 'Num passengers:', num_passengers
print 'Num nodes:', num_nodes

vehicle_node_ind = np.random.choice(graph.nodes(), size=num_vehicles, replace=False)
passenger_node_ind = np.random.choice(graph.nodes(), size=num_passengers, replace=False)

# Generate noisy vehicle positions
vehicle_node_pos = util_graph.GetNodePositions(graph, vehicle_node_ind)
vehicle_node_ind_noisy, vehicle_pos_noisy = util_noise.add_noise(vehicle_node_pos, nearest_neighbor_searcher, epsilon)

# True allocation cost
true_allocation_cost = util_vrp.get_allocation_cost(route_lengths, vehicle_node_ind, passenger_node_ind)

# Run VRP versions
waiting_time = dict()
runs = 0

# Optimal
print 'Computing VRP...'
allocation_cost = util_vrp.get_allocation_cost(route_lengths, vehicle_node_ind, passenger_node_ind)
cost, row_ind, col_ind = util_vrp.get_routing_assignment(allocation_cost)
waiting_time[runs] = true_allocation_cost[row_ind, col_ind]
runs += 1

# Noisy with naive cost function
print 'Computing noisy VRP, naive...'
allocation_cost = util_vrp.get_allocation_cost(route_lengths, vehicle_node_ind_noisy, passenger_node_ind)
cost, row_ind, col_ind = util_vrp.get_routing_assignment(allocation_cost)
waiting_time[runs] = true_allocation_cost[row_ind, col_ind]
runs += 1

# Noisy with expected cost function
print 'Computing noisy VRP, probabilistic...'
allocation_cost = util_prob.get_allocation_cost_noisy(route_lengths, vehicle_pos_noisy, passenger_node_ind, epsilon, nearest_neighbor_searcher, graph)
cost, row_ind, col_ind = util_vrp.get_routing_assignment(allocation_cost)
waiting_time[runs] =  true_allocation_cost[row_ind, col_ind]
runs += 1


# Plot
print 'Plotting...'

for i in range(runs):
    fig = plt.figure(figsize=(6,6), frameon=False)
    max_value = np.max(waiting_time[i])
    num_bins = 25
    bins = np.linspace(-0.5, max_value+0.5, num_bins+1)
    perc = [np.percentile(waiting_time[i], 50), np.percentile(waiting_time[i], 95)]
    print 'Mean:', np.mean(waiting_time[i])
    util_plot.plot_waiting_time_distr(waiting_time[0], perc, bins, fig=fig, filename=None, max_value=max_value)


plt.show(block=False)
input('Hit ENTER to close figure')

plt.close()



