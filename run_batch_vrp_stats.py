# Standard modules
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import collections
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
vehicle_density = np.array([0.05, 0.1, 0.15, 0.2])

# Noise for privacy mechanism
num_epsilon = 5
epsilons = (2 * np.logspace(-4, -1, num_epsilon)).tolist()
print 'Epsilons:', epsilons

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

num_nodes = len(graph.nodes())
print 'Num nodes:', num_nodes
num_vehicles_list = ((num_nodes * vehicle_density).astype(np.int32)).tolist()

# Iterations
num_iter = 10

# Indeces
OPT = 'optimal'
RAND = 'random'

# Create a dict of dict of lists
waiting_time = collections.defaultdict(lambda: collections.defaultdict(lambda: []))

# Run VRP versions
for it in range(num_iter):

    for num_vehicles in num_vehicles_list:

        # Random initialization of vehicle/passenger nodes
        vehicle_node_ind = np.random.choice(graph.nodes(), size=num_vehicles, replace=False)
        passenger_node_ind = np.random.choice(graph.nodes(), size=num_vehicles, replace=False)

        # True allocation cost
        true_allocation_cost = util_vrp.get_allocation_cost(route_lengths, vehicle_node_ind, passenger_node_ind)

        # Optimal
        print 'Computing optimal VRP...'
        allocation_cost = true_allocation_cost 
        cost, row_ind, col_ind = util_vrp.get_routing_assignment(allocation_cost)
        waiting_time[num_vehicles][OPT].extend(true_allocation_cost[row_ind, col_ind].flatten().tolist())


        # Noisy with expected cost function
        for epsilon in epsilons:
            # Generate noisy vehicle positions
            vehicle_node_pos = util_graph.GetNodePositions(graph, vehicle_node_ind)
            vehicle_node_ind_noisy, vehicle_pos_noisy = util_noise.add_noise(vehicle_node_pos, nearest_neighbor_searcher, epsilon)

            print 'Computing suboptimal VRP, using expected cost (epsilon = %g)...' % epsilon
            allocation_cost = util_prob.get_allocation_cost_noisy(route_lengths, vehicle_pos_noisy, passenger_node_ind, epsilon, nearest_neighbor_searcher, graph)
            cost, row_ind, col_ind = util_vrp.get_routing_assignment(allocation_cost)
            waiting_time[num_vehicles]['subopt_%g' % epsilon].extend(true_allocation_cost[row_ind, col_ind].flatten().tolist())


# We need to convert all defaultdict to dict before saving.
waiting_time = dict(waiting_time)
for k, v in waiting_time.iteritems():
    waiting_time[k] = dict((m, n) for m, n in v.iteritems())

filename = 'data/vrp_batch_s1.dat'
with open(filename, 'wb') as fp:
    fp.write(msgpack.packb({'waiting_time': waiting_time, 'epsilons': epsilons, 'num_vehicles_list': num_vehicles_list,'num_iter': num_iter}))



