# Standard modules
import numpy as np
import collections
import msgpack

# My modules
import utilities.graph as util_graph
import utilities.noise as util_noise
import utilities.vrp as util_vrp
import manhattan.data as manh_data

#-------------------------------------
# Global settings
use_manhattan = True
num_passengers = 250
num_vehicles_list = [250, 500, 1000]
repeats = [1, 2, 3]  # Allocating 1, 2, 3, or 4 vehicles.

use_real_taxi_data = True
must_recompute = False

# Noise for privacy mechanism
num_epsilon = 10
epsilons = (2 * np.logspace(-4, -1, num_epsilon)).tolist()
print 'Epsilons:', epsilons

set_seed = False
if set_seed:
    np.random.seed(1019)

# Simulation
num_iter = 20
filename = 'data/vrp_batch_real_repeats.dat'

# ---------------------------------------------------
# Load small manhattan and initialize

if use_manhattan:
    use_small_graph = False
    graph = manh_data.LoadMapData(use_small_graph=use_small_graph)
    nearest_neighbor_searcher = util_graph.NearestNeighborSearcher(graph)
    taxi_data = manh_data.LoadTaxiData(graph, synthetic_rides=not use_real_taxi_data, must_recompute=False,
                                       num_synthetic_rides=1000, max_rides=1000000)
    # Use empirical edge costs for 'time'
    manh_data.UpdateEdgeTime(graph, taxi_data, nearest_neighbor_searcher, must_recompute=False)
    # Compute route lengths based on 'time' attribute of graph
    route_lengths = manh_data.LoadShortestPathData(graph, must_recompute=False)
else:
    graph = util_graph.create_grid_map(grid_size=20, edge_length=100., default_speed=10.)
    taxi_data = manh_data.LoadTaxiData(graph, synthetic_rides=True, must_recompute=True, num_synthetic_rides=1000)
    nearest_neighbor_searcher = util_graph.NearestNeighborSearcher(graph)
    route_lengths = manh_data.LoadShortestPathData(graph, must_recompute=True)

graph, route_lengths, nearest_neighbor_searcher = util_graph.normalize(graph, route_lengths)

num_nodes = len(graph.nodes())
print 'Num nodes:', num_nodes
print 'Num vehicles:', num_vehicles_list
print 'Num passengers:', num_passengers

# Compute occurence of pick-up and drop-off locations
nearest_pickup_nodes, dist = nearest_neighbor_searcher.Search(taxi_data['pickup_xy'])
nearest_dropoff_nodes, dist = nearest_neighbor_searcher.Search(taxi_data['dropoff_xy'])
# Count probabilities of dropoff nodes
key_node, count_node = zip(*collections.Counter(nearest_dropoff_nodes).items())
vehicle_node_ind_unique = np.array(key_node)
vehicle_node_ind_unique_prob = np.array(count_node) / float(sum(count_node))
# Count probabilities of pickup nodes
key_node, count_node = zip(*collections.Counter(nearest_pickup_nodes).items())
passenger_node_ind_unique = np.array(key_node)
passenger_node_ind_unique_prob = np.array(count_node) / float(sum(count_node))

# Indeces
OPT = 'optimal'
RAND = 'random'

# Create a dict of dict of lists
waiting_time = collections.defaultdict(lambda: collections.defaultdict(lambda: []))

# Run VRP versions
for it in range(num_iter):

    for num_vehicles in num_vehicles_list:
        print 'Iteration %d with %d vehicles' % (it, num_vehicles)

        # Random initialization of vehicle/passenger nodes
        if not use_real_taxi_data:
            vehicle_node_ind = np.random.choice(graph.nodes(), size=num_vehicles, replace=True)
            passenger_node_ind = np.random.choice(graph.nodes(), size=num_passengers, replace=True)
        else:
            vehicle_node_ind = np.random.choice(vehicle_node_ind_unique, size=num_vehicles, replace=True, p=vehicle_node_ind_unique_prob)
            passenger_node_ind = np.random.choice(passenger_node_ind_unique, size=num_passengers, replace=True, p=passenger_node_ind_unique_prob)

        # Optimal
        print 'Computing optimal VRP...'
        allocation_cost = util_vrp.get_allocation_cost(route_lengths, vehicle_node_ind, passenger_node_ind)
        cost, row_ind, col_ind = util_vrp.get_routing_assignment(allocation_cost)
        waiting_time[num_vehicles][OPT].extend(util_vrp.compute_waiting_times(route_lengths, vehicle_node_ind, passenger_node_ind, row_ind, col_ind))

        # Random
        print 'Computing random VRP...'
        cost, row_ind, col_ind = util_vrp.get_rand_routing_assignment(allocation_cost)
        waiting_time[num_vehicles][RAND].extend(util_vrp.compute_waiting_times(route_lengths, vehicle_node_ind, passenger_node_ind, row_ind, col_ind))

        # Noisy with expected cost function
        for epsilon in epsilons:
            # Generate noisy vehicle positions
            vehicle_node_pos = util_graph.GetNodePositions(graph, vehicle_node_ind)
            _, vehicle_pos_noisy = util_noise.add_noise(vehicle_node_pos, nearest_neighbor_searcher, epsilon)

            print 'Computing suboptimal VRP, using expected cost (epsilon = %g)...' % epsilon
            cost, row_ind, col_ind, vd = util_vrp.get_repeated_routing_assignment(route_lengths, vehicle_pos_noisy, passenger_node_ind, epsilon, nearest_neighbor_searcher, graph, repeat=0)
            waiting_time[num_vehicles]['subopt_%g_0' % epsilon].extend(util_vrp.compute_waiting_times(route_lengths, vehicle_node_ind, passenger_node_ind, row_ind, col_ind))
            previous_repeat = 0
            for repeat in repeats:
                # Do not compute for cases where vehicles cannot be multi-allocated.
                if num_vehicles < num_passengers * (repeat + 1):
                    break
                cost, row_ind, col_ind, _ = util_vrp.get_repeated_routing_assignment(route_lengths, vehicle_pos_noisy, passenger_node_ind, epsilon, nearest_neighbor_searcher, graph, repeat=repeat,
                                                                                     previous_row_ind=row_ind, previous_col_ind=col_ind, previous_repeat=previous_repeat, previous_vehicle_distances=vd)
                waiting_time[num_vehicles]['subopt_%g_%d' % (epsilon, repeat)].extend(util_vrp.compute_waiting_times(route_lengths, vehicle_node_ind, passenger_node_ind, row_ind, col_ind))
                previous_repeat = repeat

# We need to convert all defaultdict to dict before saving.
waiting_time = dict(waiting_time)
for k, v in waiting_time.iteritems():
    waiting_time[k] = dict((m, n) for m, n in v.iteritems())

with open(filename, 'wb') as fp:
    fp.write(msgpack.packb({'waiting_time': waiting_time, 'epsilons': epsilons, 'num_vehicles_list': num_vehicles_list, 'num_iter': num_iter, 'repeats': repeats}))
