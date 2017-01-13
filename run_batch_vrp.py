# Standard modules
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.optimize as opt
import scipy.special as spc
import pickle
import collections

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
vehicle_density = 0.1
passenger_density = 0.1

# Noise for privacy mechanism
epsilons = [0.02] #, 0.01]

set_seed = False
if set_seed:
    np.random.seed(1019)

# ---------------------------------------------------
# Load small manhattan and initialize

if use_manhattan:
    use_small_graph = False
    graph  = manh_data.LoadMapData(use_small_graph=use_small_graph)
    nearest_neighbor_searcher = util_graph.NearestNeighborSearcher(graph)
    route_lengths = manh_data.LoadShortestPathData(graph, must_recompute=False)
else:
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

# Iterations
num_iter = 20
# Indeces
OPT = 'optimal'
RAND = 'random'

# Run VRP versions
waiting_time = collections.defaultdict(lambda: [])

for it in range(num_iter):

    # Random initialization of vehicle/passenger nodes
    vehicle_node_ind = np.random.choice(graph.nodes(), size=num_vehicles, replace=False)
    passenger_node_ind = np.random.choice(graph.nodes(), size=num_passengers, replace=False)

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



# Plot
print 'Plotting...'

set_x_lim = 1000
max_value = max(np.max(w) for i, w in waiting_time.iteritems() if i != RAND)
num_bins = 100
for i, w in waiting_time.iteritems():
    print 'Mean, %s: %g' % (i, np.mean(w))
    if i == RAND:
            continue
    fig = plt.figure(figsize=(6,6), frameon=False)
    bins = np.linspace(-0.5, max_value+0.5, num_bins+1)
    stats = [np.mean(w)]
    filename = 'figures/results_batch_%s.eps' % i
    util_plot.plot_waiting_time_distr(w, stats, bins, fig=fig, filename=filename, max_value=max_value, set_max=set_x_lim)


plt.show(block=False)
raw_input('Hit ENTER to close figure')

plt.close()
