# Standard modules
import numpy as np
import matplotlib.pyplot as plt
import collections
import time
import msgpack

# My modules
import utilities.graph as util_graph
import utilities.noise as util_noise
import utilities.vrp as util_vrp
import utilities.plot as util_plot
import manhattan.data as manh_data

#-------------------------------------
# Global settings
use_manhattan = False
num_vehicles = 500
num_passengers = 250

use_real_taxi_data = False
must_recompute = False

# Uncertainty on locations
noise_model = 'gauss' # {'gauss', 'laplace'}
# Set noise parameter: scale 
if noise_model == 'laplace': epsilons = [0.02] 
elif noise_model == 'gauss': epsilons =  [20.0, 60.0]  

# Set-greedy 
repeats = [] # [1, 2, 3]  # Start at 1 (0 is always tested).

plot_on = True
set_seed = False
if set_seed:
    np.random.seed(1019)

# Save simulation data and figures
filename = 'data/rich-vrp_batch_s0.dat'
fig_fn_base = 'figures/rich-vrp_batch_s0'

# Iterations
num_iter = 1

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
    taxi_data = manh_data.LoadTaxiData(graph, synthetic_rides=True, must_recompute=must_recompute, num_synthetic_rides=1000)
    nearest_neighbor_searcher = util_graph.NearestNeighborSearcher(graph)
    route_lengths = manh_data.LoadShortestPathData(graph, must_recompute=must_recompute)

graph, route_lengths, nearest_neighbor_searcher = util_graph.normalize(graph, route_lengths)

num_nodes = len(graph.nodes())
print 'Num nodes:', num_nodes
print 'Num vehicles:', num_vehicles
print 'Num passengers:', num_passengers

# Compute occurence of pick-up and drop-off locations -> used to sample realistic pick-up and drop-off locations
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

    # Optimal
    print 'Computing optimal VRP...'
    s = time.time()
    true_allocation_cost = util_vrp.get_allocation_cost(route_lengths, vehicle_node_ind, passenger_node_ind)
    allocation_cost = true_allocation_cost
    cost, row_ind, col_ind = util_vrp.get_routing_assignment(allocation_cost)
    print 'Optimal allocation took %.2fms' % ((time.time() - s) * 1000.)
    waiting_time[OPT].extend(util_vrp.compute_waiting_times(route_lengths, vehicle_node_ind, passenger_node_ind, row_ind, col_ind))

    # For each noise value epsilon, generate set-greedy allocation.
    for epsilon in epsilons:
        # Generate noisy vehicle positions
        vehicle_node_pos = util_graph.GetNodePositions(graph, vehicle_node_ind)
        _, vehicle_pos_noisy = util_noise.add_noise(vehicle_node_pos, nearest_neighbor_searcher, epsilon, noise_model)
        # Generate noisy passenger positions
        passenger_node_pos = util_graph.GetNodePositions(graph, passenger_node_ind)
        _, passenger_pos_noisy = util_noise.add_noise(passenger_node_pos, nearest_neighbor_searcher, epsilon, noise_model)

        print 'Computing suboptimal VRP, using expected cost (epsilon = %g)...' % epsilon
        s = time.time()

        # Set-greedy allocation strategy
        cost, row_ind, col_ind, vd = util_vrp.get_repeated_routing_assignment(route_lengths, vehicle_pos_noisy, passenger_node_ind, epsilon, noise_model, nearest_neighbor_searcher, graph, repeat=0)
        print 'Suboptimal allocation took %.2fms' % ((time.time() - s) * 1000.)
        waiting_time['subopt_%g_0' % epsilon].extend(util_vrp.compute_waiting_times(route_lengths, vehicle_node_ind, passenger_node_ind, row_ind, col_ind))
        previous_repeat = 0
        for repeat in repeats:
            s = time.time()
            cost, row_ind, col_ind, _ = util_vrp.get_repeated_routing_assignment(route_lengths, vehicle_pos_noisy, passenger_node_ind, epsilon, noise_model, nearest_neighbor_searcher, graph, repeat=repeat,
                                                                                 previous_row_ind=row_ind, previous_col_ind=col_ind, previous_repeat=previous_repeat, previous_vehicle_distances=vd)
            print 'Extra suboptimal allocation took %.2fms' % ((time.time() - s) * 1000.)
            waiting_time['subopt_%g_%d' % (epsilon, repeat)].extend(util_vrp.compute_waiting_times(route_lengths, vehicle_node_ind, passenger_node_ind, row_ind, col_ind))
            previous_repeat = repeat

    # Random
    print 'Computing random VRP...'
    cost, row_ind, col_ind = util_vrp.get_rand_routing_assignment(true_allocation_cost)
    waiting_time[RAND].extend(util_vrp.compute_waiting_times(route_lengths, vehicle_node_ind, passenger_node_ind, row_ind, col_ind))

with open(filename, 'wb') as fp:
    fp.write(msgpack.packb({'waiting_time': waiting_time, 'epsilons': epsilons, 'num_vehicles': num_vehicles, 'num_passengers': num_passengers, 'num_iter': num_iter}))

# Plot
if plot_on:
    print 'Plotting...'
    set_x_lim = None #500
    set_y_lim = None #0.25
    max_value = max(np.max(w) for i, w in waiting_time.iteritems() if i != RAND)
    num_bins = 30
    for i, w in waiting_time.iteritems():
        print 'Mean, %s: %g' % (i, np.mean(w))
        if i == RAND:
            continue
        fig = plt.figure(figsize=(6, 6), frameon=False)
        bins = np.linspace(-0.5, max_value + 0.5, num_bins+1)
        stats = [np.mean(w)]
        fig_filename = fig_fn_base + '_hist_%s.eps' % i
        util_plot.plot_waiting_time_distr(w, stats, bins, fig=fig, filename=fig_filename, max_value=max_value, set_x_max=set_x_lim, set_y_max=set_y_lim)

    plt.show(block=False)
    raw_input('Hit ENTER to close figure')

    plt.close()
