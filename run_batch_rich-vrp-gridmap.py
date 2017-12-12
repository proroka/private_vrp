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


BOUND_INF = 0
BOUND_HUNGARIAN = 1


#-------------------------------------
# Global settings

# Total number of cars and passengers
num_vehicles_list = [4, 8, 12, 16]
num_passengers = 4

grid_size = 10
edge_length = 100.
speed = 10.
std = 2.0 # sigma on speeds

# Set-greedy settings
#repeats = [1] # Start at 1 (0 is always tested).

# Uncertainty on locations
noise_model = 'gauss' # {'gauss', 'laplace'}
# Set noise parameter: scale 
if noise_model == 'laplace': epsilons = [0.02] 
elif noise_model == 'gauss': epsilons =  [100.0]  

plot_on = True
set_seed = False
if set_seed:
    np.random.seed(1019)

# Iterations over vehicle/passenger distributions
num_iter = 1000

# Save simulation data and figures
filename = 'data/rich-vrp_batch_s3.dat'
fig_fn_base = 'figures/rich-vrp_batch_s3'




#-------------------------------------
# Load structures

# Load graph
graph = util_graph.create_grid_map(grid_size=grid_size, edge_length=edge_length, default_speed=speed, std_dev=std)
nearest_neighbor_searcher = util_graph.NearestNeighborSearcher(graph)
route_lengths = util_graph.grid_map_route_lengths(graph)  #manh_data.LoadShortestPathData(graph, must_recompute=must_recompute)

# Transform route_lengths from dict to ndarray
graph, route_lengths, nearest_neighbor_searcher = util_graph.normalize(graph, route_lengths)

plot_distr = False
if plot_distr:
    rl_flat = route_lengths.flatten()
    rl_bins = 50
    hdata, hbins = np.histogram(rl_flat, bins=rl_bins)
    plt.figure()
    plt.bar(hbins[:-1], hdata, hbins[1]-hbins[0], bottom=None, hold=None, data=None)


num_nodes = len(graph.nodes())
print 'Num nodes:', num_nodes
print 'Num vehicles:', num_vehicles_list
print 'Num passengers:', num_passengers


# Run VRP algorithms
TRUE = 'true'
OPT = 'optimal'
SG = 'set-greedy'
EG = 'element-greedy'
HUN = 'hungarian'
RAND = 'random'
#waiting_time = collections.defaultdict(lambda: [])
waiting_time = collections.defaultdict(lambda: collections.defaultdict(list))

#-------------------------------------
# Run algorithms


for num_vehicles in num_vehicles_list:
    # Settings for set-greedy
    repeats = range(1, num_vehicles / num_passengers)

    for it in range(num_iter):

        s = time.time()
        print '********  Iteration %d  ********' % it
        # Compute vehicle and passenger pickup and dropoff locations
        vehicle_node_ind = np.random.choice(graph.nodes(), size=num_vehicles, replace=True)
        passenger_node_ind = np.random.choice(graph.nodes(), size=num_passengers, replace=True)

        if plot_distr:
            rl_flat = route_lengths[vehicle_node_ind][passenger_node_ind].flatten()
            print len(rl_flat)
            rl_bins = 50
            hdata, hbins = np.histogram(rl_flat, bins=rl_bins)
            plt.figure()
            plt.bar(hbins[:-1], hdata, hbins[1]-hbins[0], bottom=None, hold=None, data=None)

        # Non-noisy (true) allocation
        true_allocation_cost = util_vrp.get_allocation_cost(route_lengths, vehicle_node_ind, passenger_node_ind)
        allocation_cost = true_allocation_cost
        cost, row_ind, col_ind = util_vrp.get_routing_assignment(allocation_cost)
        waiting_time[TRUE][num_vehicles].append(util_vrp.compute_waiting_times(route_lengths, vehicle_node_ind, passenger_node_ind, row_ind, col_ind))

        for epsilon in epsilons:
            # Generate noisy vehicle positions
            vehicle_node_pos = util_graph.GetNodePositions(graph, vehicle_node_ind)
            _, vehicle_pos_noisy = util_noise.add_noise(vehicle_node_pos, nearest_neighbor_searcher, epsilon, noise_model)        
            # Generate noisy passenger positions
            passenger_node_pos = util_graph.GetNodePositions(graph, passenger_node_ind)
            _, passenger_pos_noisy = util_noise.add_noise(passenger_node_pos, nearest_neighbor_searcher, epsilon, noise_model)

            num_samples = 200
            route_length_samples = util_vrp.get_vehicle_sample_route_lengths(route_lengths, num_samples, vehicle_pos_noisy, passenger_node_ind, nearest_neighbor_searcher, epsilon, noise_model)
            
            # Compute optimal allocation
            topt = time.time()
            print 'Computing optimal allocation, using expected cost (epsilon = %g)...' % epsilon
            cost, row_ind, col_ind = util_vrp.get_optimal_assignment(route_length_samples, vehicle_pos_noisy, passenger_node_ind, nearest_neighbor_searcher, epsilon, noise_model,
                                use_initial_hungarian=True, use_bound=True, refined_bound=True, bound_initialization=BOUND_HUNGARIAN)
            #print 'Time for opt in BATCH: ', time.time() - topt
            waiting_time[OPT+'_%g_0' % epsilon][num_vehicles].append(util_vrp.compute_waiting_times(route_lengths, vehicle_node_ind, passenger_node_ind, row_ind, col_ind))
            #print cost

            # Compute element-greedy allocation
            print 'Computing element-greedy allocation, using expected cost (epsilon = %g)...' % epsilon
            cost, row_ind, col_ind = util_vrp.get_greedy_assignment(route_length_samples, vehicle_pos_noisy, passenger_node_ind, epsilon, noise_model, nearest_neighbor_searcher, graph)
            waiting_time[EG+'_%g_0' % epsilon][num_vehicles].append(util_vrp.compute_waiting_times(route_lengths, vehicle_node_ind, passenger_node_ind, row_ind, col_ind))
            #print cost

            # Compute set-greedy allocation
            print 'Computing set-greedy allocation, using expected cost (epsilon = %g)...' % epsilon
            cost, row_ind, col_ind = util_vrp.get_set_greedy_assignment(route_length_samples, vehicle_pos_noisy, passenger_node_ind, epsilon, noise_model, nearest_neighbor_searcher, graph, repeats)
            waiting_time[SG+'_%g_0' % epsilon][num_vehicles].append(util_vrp.compute_waiting_times(route_lengths, vehicle_node_ind, passenger_node_ind, row_ind, col_ind))
            #print cost

            # Compute Hungarian allocation, redundant vehicles remain unused
            print 'Computing Hungarian allocation, using expected cost (epsilon = %g)...' % epsilon
            cost, row_ind, col_ind = util_vrp.get_Hungarian_assignment(route_length_samples) #, vehicle_pos_noisy, passenger_node_ind, epsilon, noise_model, nearest_neighbor_searcher, graph)
            waiting_time[HUN+'_%g_0' % epsilon][num_vehicles].append(util_vrp.compute_waiting_times(route_lengths, vehicle_node_ind, passenger_node_ind, row_ind, col_ind))
            #print cost

        # Random
        print 'Computing random VRP...'
        cost, row_ind, col_ind = util_vrp.get_rand_routing_assignment(true_allocation_cost)
        waiting_time[RAND][num_vehicles].append(util_vrp.compute_waiting_times(route_lengths, vehicle_node_ind, passenger_node_ind, row_ind, col_ind))

        e = time.time()
        print 'Time per iteration: ', e-s

    with open(filename, 'wb') as fp:
        fp.write(msgpack.packb({'waiting_time': waiting_time, 'epsilons': epsilons, 'num_vehicles': num_vehicles, 'num_passengers': num_passengers, 'num_iter': num_iter}))



#-------------------------------------
# Plot results

# Plot
plot_hist = False
if plot_hist:
    print 'Plotting...'
    set_x_lim = None #500
    set_y_lim = None #0.25
    max_value = max(np.max(w) for i, w in waiting_time.iteritems() if i != RAND)
    num_bins = 30
    for i, w in waiting_time.iteritems():
        print 'Mean, %s:\t %g ' % (i, np.mean(w))
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

plot_curve = True
if plot_curve:
    
    col = ['b','r','g','m','c','y']
    fig = plt.figure(figsize=(6, 6), frameon=False)


    ind = 0
    for algo, w_dict in waiting_time.iteritems():
        if algo == RAND:
            continue
        m_values = np.zeros((len(num_vehicles_list),))
        s_values = np.zeros((len(num_vehicles_list),))
        
        i_list = []
        for num_vehicles, w in w_dict.iteritems():
        
            index = np.where(np.array(num_vehicles_list)==num_vehicles)
            i_list.append(index)
            m_values[index] = np.mean(w)
            s_values[index] = np.std(w)
            #print '%d : %f' % (num_vehicles, np.mean(w))

        print 'Algorithm: %s, : %s' %(algo, m_values)
        #print i_list
        plt.plot(num_vehicles_list, m_values, color=col[ind], marker='o', label=algo)
        ind += 1

    plt. legend()
    fig_filename = fig_fn_base + '_curve.eps'
    plt.show(block=False)
    raw_input('Hit ENTER to close figure')

    plt.close()



