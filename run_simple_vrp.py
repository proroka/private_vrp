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
import utilities.plot as util_plot
import utilities.probabilistic as util_prob
import manhattan.data as manh_data

#-------------------------------------
# Global settings
vehicle_density = 0.3
passenger_density = 0.3
# Noise
epsilon = 0.02

set_seed = True

if set_seed: 
    np.random.seed(1019)

# ---------------------------------------------------
# Load small manhattan and initialize

use_small_graph = True
graph  = manh_data.LoadMapData(use_small_graph=use_small_graph)
nearest_neighbor_searcher = util_graph.NearestNeighborSearcher(graph)
route_lengths = manh_data.LoadShortestPathData(graph, use_small_graph=use_small_graph)
taxi_data = manh_data.LoadTaxiData(graph, route_lengths, use_small_graph=use_small_graph)

# Initialize 
num_nodes = len(graph.nodes())
num_vehicles = int(num_nodes * vehicle_density)
num_passengers = int(num_nodes * passenger_density)

print 'Num vehicles:', num_vehicles
print 'Num passengers:', num_passengers
print 'Num nodes:', num_nodes

vehicle_node_ind = np.random.choice(graph.nodes(), size=num_vehicles, replace=False)
passenger_node_ind = np.random.choice(graph.nodes(), size=num_passengers, replace=False)


# Run VRP versions

# Optimal
print 'Computing VRP...'
waiting_time_vrpopt = util_vrp.run_vrp_allocation(route_lengths, vehicle_node_ind, passenger_node_ind)


# Generate noisy vehicle positions
vehicle_node_pos = util_graph.GetNodePositions(graph, vehicle_node_ind)
vehicle_node_ind_noisy, vehicle_pos_noisy = util_noise.add_noise(vehicle_node_pos, nearest_neighbor_searcher, epsilon)

# Noisy with naive cost function
print 'Computing noisy VRP, naive...'
waiting_time_vrpsubopt = util_vrp.run_vrp_allocation(route_lengths, vehicle_node_ind_noisy, passenger_node_ind)

# Noisy with expected cost function
print 'Computing noisy VRP, probabilistic...'
allocation_cost = util_prob.get_allocation_cost_noisy(route_lengths, vehicle_pos_noisy, passenger_node_ind, epsilon,
                                                      nearest_neighbor_searcher, graph)
waiting_time_vrpprob = util_vrp.run_vrp(allocation_cost)



# Plot
print 'Plotting...'


fig1 = plt.figure(figsize=(6,6), frameon=False)
max_value = np.max(waiting_time_vrpopt)
num_bins = 25
bins = np.linspace(-0.5, max_value+0.5, num_bins+1)
perc_vrpopt = [np.percentile(waiting_time_vrpopt, 50), np.percentile(waiting_time_vrpopt, 95)]
util_plot.plot_waiting_time_distr(waiting_time_vrpopt, perc_vrpopt, bins, fig=fig1, filename=None, max_value=max_value)

fig2 = plt.figure(figsize=(6,6), frameon=False)
max_value = np.max(waiting_time_vrpsubopt)
num_bins = 25
bins = np.linspace(-0.5, max_value+0.5, num_bins+1)
perc_vrpsubopt = [np.percentile(waiting_time_vrpsubopt, 50), np.percentile(waiting_time_vrpsubopt, 95)]
util_plot.plot_waiting_time_distr(waiting_time_vrpsubopt, perc_vrpsubopt, bins, fig=fig2, filename=None, max_value=max_value)

fig3 = plt.figure(figsize=(6,6), frameon=False)
max_value = np.max(waiting_time_vrpprob)
num_bins = 25
bins = np.linspace(-0.5, max_value+0.5, num_bins+1) 
perc_vrpprob = [np.percentile(waiting_time_vrpprob, 50), np.percentile(waiting_time_vrpprob, 95)]
util_plot.plot_waiting_time_distr(waiting_time_vrpprob, perc_vrpprob, bins, fig=fig3, filename=None, max_value=max_value)


plt.show(block=False)
input("Hit Enter To Close")
plt.close()
#plt.show()




