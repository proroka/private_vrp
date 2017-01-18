# Standard modules
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.optimize as opt
import scipy.special as spc
import pickle
import collections
from collections import Counter
import msgpack

# My modules
import utilities.graph as util_graph
import utilities.noise as util_noise
import utilities.vrp as util_vrp
import utilities.plot as util_plot
import utilities.probabilistic as util_prob
import manhattan.data as manh_data

#-------------------------------------
# Global settings
use_manhattan = True
num_vehicles = 500
num_passengers = 250

use_real_taxi_data = True
must_recompute = False

# Noise for privacy mechanism
epsilons = [0.02, 0.01]

ploton = False
set_seed = False
if set_seed:
    np.random.seed(1019)

# Simulation
filename = 'data/vrp_batch_real_distrib_s4.dat'
# Iterations
num_iter = 100

# ---------------------------------------------------
# Load small manhattan and initialize


if use_manhattan:
    use_small_graph = False
    graph  = manh_data.LoadMapData(use_small_graph=use_small_graph)
    nearest_neighbor_searcher = util_graph.NearestNeighborSearcher(graph)

    taxi_data = manh_data.LoadTaxiData(graph, synthetic_rides=not use_real_taxi_data, must_recompute=must_recompute,
                                   num_synthetic_rides=1000, max_rides=1000000)


    # Use empirical edge costs for 'time'
    manh_data.UpdateEdgeTime(graph, taxi_data, nearest_neighbor_searcher, must_recompute=False)
    # Compute route lengths based on 'time' attribute of graph
    route_lengths = manh_data.LoadShortestPathData(graph, must_recompute=False)

else:
    graph = util_graph.create_grid_map(grid_size=20, edge_length=100.)
    nearest_neighbor_searcher = util_graph.NearestNeighborSearcher(graph)
    route_lengths = manh_data.LoadShortestPathData(graph, must_recompute=True)


num_nodes = len(graph.nodes())
print 'Num nodes:', num_nodes
print 'Num vehicles:', num_vehicles
print 'Num passengers:', num_passengers

# Compute occurence of pick-up and drop-off locations 
nearest_pickup_nodes, dist = nearest_neighbor_searcher.Search(taxi_data['pickup_xy'])
nearest_dropoff_nodes, dist = nearest_neighbor_searcher.Search(taxi_data['dropoff_xy'])
# Count probabilities of dropoff nodes
key_node, count_node = zip(*Counter(nearest_dropoff_nodes).items())
vehicle_node_ind_unique = np.array(key_node)
vehicle_node_ind_unique_prob = np.array(count_node) / float(sum(count_node))
# Count probabilities of pickup nodes
key_node, count_node = zip(*Counter(nearest_pickup_nodes).items())
passenger_node_ind_unique = np.array(key_node)
passenger_node_ind_unique_prob = np.array(count_node) / float(sum(count_node))



# Indeces
OPT = 'optimal'
RAND = 'random'

# Run VRP versions
waiting_time = collections.defaultdict(lambda: [])

for it in range(num_iter):

    # Random initialization of vehicle/passenger nodes
    if not use_real_taxi_data:
        vehicle_node_ind = np.random.choice(graph.nodes(), size=num_vehicles, replace=True)
        passenger_node_ind = np.random.choice(graph.nodes(), size=num_passengers, replace=True)
    else:
        vehicle_node_ind = np.random.choice(vehicle_node_ind_unique, size=num_vehicles, replace=True, p=vehicle_node_ind_unique_prob)
        passenger_node_ind = np.random.choice(passenger_node_ind_unique, size=num_passengers, replace=True, p=passenger_node_ind_unique_prob)

    # True allocation cost
    true_allocation_cost = util_vrp.get_allocation_cost(route_lengths, vehicle_node_ind, passenger_node_ind)

    # Optimal
    print 'Computing optimal VRP...'
    allocation_cost = true_allocation_cost 
    cost, row_ind, col_ind = util_vrp.get_routing_assignment(allocation_cost)
    waiting_time[OPT].extend(true_allocation_cost[row_ind, col_ind].flatten().tolist())

    # Random
    print 'Computing random VRP...'
    allocation_cost = true_allocation_cost 
    cost, row_ind, col_ind = util_vrp.get_rand_routing_assignment(allocation_cost)
    waiting_time[RAND].extend(true_allocation_cost[row_ind, col_ind].flatten().tolist())


    # Noisy with expected cost function
    for epsilon in epsilons:
        # Generate noisy vehicle positions
        vehicle_node_pos = util_graph.GetNodePositions(graph, vehicle_node_ind)
        vehicle_node_ind_noisy, vehicle_pos_noisy = util_noise.add_noise(vehicle_node_pos, nearest_neighbor_searcher, epsilon)

        print 'Computing suboptimal VRP, using expected cost (epsilon = %g)...' % epsilon
        allocation_cost = util_prob.get_allocation_cost_noisy(route_lengths, vehicle_pos_noisy, passenger_node_ind, epsilon, nearest_neighbor_searcher, graph)
        cost, row_ind, col_ind = util_vrp.get_routing_assignment(allocation_cost)
        waiting_time['subopt_%g' % epsilon].extend(true_allocation_cost[row_ind, col_ind].flatten().tolist())


with open(filename, 'wb') as fp:
    fp.write(msgpack.packb({'waiting_time': waiting_time, 'epsilons': epsilons, 'num_vehicles': num_vehicles, 'num_passengers': num_passengers, 'num_iter': num_iter}))


# Plot

if plot_on:
    print 'Plotting...'

    set_x_lim = 500
    set_y_lim = 0.25
    max_value = max(np.max(w) for i, w in waiting_time.iteritems() if i != RAND)
    num_bins = 100
    for i, w in waiting_time.iteritems():
        print 'Mean, %s: %g' % (i, np.mean(w))
        if i == RAND:
                continue
        fig = plt.figure(figsize=(6,6), frameon=False)
        bins = np.linspace(-0.5, max_value+0.5, num_bins+1)
        stats = [np.mean(w)]
        fig_filename = 'figures/results_batch_real_distrib_%s.eps' % i
        util_plot.plot_waiting_time_distr(w, stats, bins, fig=fig, filename=fig_filename, max_value=max_value, set_x_max=set_x_lim, set_y_max=set_y_lim)


    plt.show(block=False)
    raw_input('Hit ENTER to close figure')

    plt.close()
