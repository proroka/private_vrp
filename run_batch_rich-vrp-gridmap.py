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

run = 30

# Uncertainty on locations
noise_model = 'gauss' # {'gauss', 'laplace', 'uniform'}
compute_slice = True

# Iterations over vehicle/passenger distributions
num_iter = 100
compute_optimal = True
include_set_greedy = False
use_initial_hungarian = True

# Save simulation data and figures
filename = 'data/rich-vrp_batch_s' + str(run) + '.dat'
fig_fn_base = 'figures/rich-vrp_batch_s' + str(run)

# Total number of cars and passengers
if compute_optimal:
    max_assignable_vehicles_list = [4, 6, 8, 10, 12, 14, 16]
else:
    max_assignable_vehicles_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
num_vehicles = max_assignable_vehicles_list[-1]
num_passengers = max_assignable_vehicles_list[0]

if compute_slice:
    max_assignable_vehicles_list = [10]
    num_vehicles = 16
    num_passengers = 4


grid_size = 16
edge_length = 50.
speed = 10.
std = 2.0 # sigma on speeds

# To get a standard deviation of 100:
# For Gaussian: epsilon = 100             (i.e., sigma)
# For Laplace: epsilon = np.sqrt(3) / 100 (i.e., scale parameter)
# For Uniform: epsilon = 2 * 100          (i.e., radius)
if noise_model == 'laplace': epsilons = [np.sqrt(3) / 100.]
elif noise_model == 'gauss': epsilons =  [25., 50., 75., 100.0]
elif noise_model == 'uniform': epsilons = [2. * 100.]
if compute_slice:
    # epsilons = range(20,200,10)
    epsilons = [10, 20, 40, 60, 80, 120, 160, 240, 320]


plot_on = True
set_seed = False
if set_seed:
    np.random.seed(1019)


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
print 'Num assignable vehicles:', max_assignable_vehicles_list
print 'Num total vehicles:', num_vehicles
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
costs = collections.defaultdict(lambda: collections.defaultdict(list))

#-------------------------------------
# Run algorithms

for max_assignable_vehicles in max_assignable_vehicles_list:
    # Settings for set-greedy
    repeats = range(1, max_assignable_vehicles / num_passengers)

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
        waiting_time[TRUE][max_assignable_vehicles].append(util_vrp.compute_waiting_times(route_lengths, vehicle_node_ind, passenger_node_ind, row_ind, col_ind))

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
            if compute_optimal:
                print 'Computing optimal allocation, using expected cost (epsilon = %g)...' % epsilon
                _, row_ind, col_ind = util_vrp.get_optimal_assignment(route_length_samples, vehicle_pos_noisy, passenger_node_ind, nearest_neighbor_searcher, epsilon, noise_model,
                                    use_initial_hungarian=use_initial_hungarian, use_bound=False, refined_bound=True, bound_initialization=BOUND_HUNGARIAN,
                                    max_assignable_vehicles=max_assignable_vehicles)
                #print 'Time for opt in BATCH: ', time.time() - topt
                waiting_time[OPT+'_%g' % epsilon][max_assignable_vehicles].append(util_vrp.compute_waiting_times(route_lengths, vehicle_node_ind, passenger_node_ind, row_ind, col_ind))
                costs[OPT+'_%g' % epsilon][max_assignable_vehicles].append(util_vrp.compute_sampled_cost(route_length_samples, row_ind, col_ind))

            # Compute element-greedy allocation
            print 'Computing element-greedy allocation, using expected cost (epsilon = %g)...' % epsilon
            _, row_ind, col_ind = util_vrp.get_greedy_assignment(route_length_samples, vehicle_pos_noisy, passenger_node_ind, max_assignable_vehicles, epsilon, noise_model, nearest_neighbor_searcher, graph)
            waiting_time[EG+'_%g' % epsilon][max_assignable_vehicles].append(util_vrp.compute_waiting_times(route_lengths, vehicle_node_ind, passenger_node_ind, row_ind, col_ind))
            costs[EG+'_%g' % epsilon][max_assignable_vehicles].append(util_vrp.compute_sampled_cost(route_length_samples, row_ind, col_ind))

            # Compute set-greedy allocation
            if include_set_greedy:
                print 'Computing set-greedy allocation, using expected cost (epsilon = %g)...' % epsilon
                _, row_ind, col_ind = util_vrp.get_set_greedy_assignment(route_length_samples, vehicle_pos_noisy, passenger_node_ind, epsilon, noise_model, nearest_neighbor_searcher, graph, repeats)
                waiting_time[SG+'_%g' % epsilon][max_assignable_vehicles].append(util_vrp.compute_waiting_times(route_lengths, vehicle_node_ind, passenger_node_ind, row_ind, col_ind))
                costs[SG+'_%g' % epsilon][max_assignable_vehicles].append(util_vrp.compute_sampled_cost(route_length_samples, row_ind, col_ind))

            # Compute Hungarian allocation, redundant vehicles remain unused
            print 'Computing Hungarian allocation, using expected cost (epsilon = %g)...' % epsilon
            _, row_ind, col_ind = util_vrp.get_Hungarian_assignment(route_length_samples) #, vehicle_pos_noisy, passenger_node_ind, epsilon, noise_model, nearest_neighbor_searcher, graph)
            waiting_time[HUN+'_%g' % epsilon][max_assignable_vehicles].append(util_vrp.compute_waiting_times(route_lengths, vehicle_node_ind, passenger_node_ind, row_ind, col_ind))
            costs[HUN+'_%g' % epsilon][max_assignable_vehicles].append(util_vrp.compute_sampled_cost(route_length_samples, row_ind, col_ind))

            # Random assignment of redundant vehicles, first set assigned based on Hungarian
            print 'Computing random VRP...'
            _, row_ind, col_ind = util_vrp.get_rand_routing_assignment(route_length_samples, max_assignable_vehicles)
            waiting_time[RAND+'_%g' % epsilon][max_assignable_vehicles].append(util_vrp.compute_waiting_times(route_lengths, vehicle_node_ind, passenger_node_ind, row_ind, col_ind))
            costs[RAND+'_%g' % epsilon][max_assignable_vehicles].append(util_vrp.compute_sampled_cost(route_length_samples, row_ind, col_ind))

        e = time.time()
        print 'Time per iteration: ', e-s

    with open(filename, 'wb') as fp:
        fp.write(msgpack.packb({'waiting_time': waiting_time, 'epsilons': epsilons, 'max_assignable_vehicles_list': max_assignable_vehicles_list, 'num_vehicles': num_vehicles, 'num_passengers': num_passengers, 'num_iter': num_iter,
                                'sampled_cost': costs}))



#-------------------------------------
# Plot results



# Plot histograms
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

# Plot performance vs num vehicles
plot_curve = True
if plot_curve and not compute_slice:

    col = ['b','r','g','m','c','y']
    fig = plt.figure(figsize=(6, 6), frameon=False)

    ind = 0
    for algo, w_dict in waiting_time.iteritems():
        #if algo == RAND:
        #    continue
        m_values = np.zeros((len(max_assignable_vehicles_list),))
        s_values = np.zeros((len(max_assignable_vehicles_list),))

        for num_vehicles, w in w_dict.iteritems():
            # means over passengers per iter
            means = np.zeros(num_iter)
            for j in range(num_iter):
                means[j] = np.mean(w[j])

            index = np.where(np.array(max_assignable_vehicles_list)==num_vehicles)
            m_values[index] = np.mean(w)
            s_values[index] = np.std(means)
            #print '%d : %f' % (num_vehicles, np.mean(w))

        print 'Mean Waiting Time'
        print '%s, : %s' %(algo, m_values)
        print '%s, : %s' %(algo, s_values)
        plt.plot(max_assignable_vehicles_list, m_values, color=col[ind], marker='o', label=algo)
        plt.errorbar(max_assignable_vehicles_list, m_values, s_values, color=col[ind])


        plt.title('Mean Waiting Time')
        ind += 1

    plt. legend()
    fig_filename = fig_fn_base + '_curve.eps'
    plt.show(block=False)

# Plot cost vs num vehicles
plot_sampled_curve = False
if plot_sampled_curve:

    col = ['b','r','g','m','c','y']
    fig = plt.figure(figsize=(6, 6), frameon=False)


    ind = 0
    for algo, w_dict in costs.iteritems():
        # if algo == RAND or algo == TRUE:
        #     continue
        m_values = np.zeros((len(max_assignable_vehicles_list),))
        s_values = np.zeros((len(max_assignable_vehicles_list),))

        i_list = []
        for num_vehicles, w in w_dict.iteritems():
            index = np.where(np.array(max_assignable_vehicles_list)==num_vehicles)
            i_list.append(index)
            m_values[index] = np.mean(w)
            s_values[index] = np.std(w)
            #print '%d : %f' % (num_vehicles, np.mean(w))

        print 'Sampled Cost'
        print '%s, : %s' %(algo, m_values)
        #print i_list
        plt.plot(max_assignable_vehicles_list, m_values, color=col[ind], marker='o', label=algo)

        plt.title('Sampled cost')
        ind += 1

    plt. legend()
    fig_filename = fig_fn_base + '_curve.eps'
    plt.show(block=False)

raw_input('Hit ENTER to close figure')
plt.close()



