# Standard modules
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.optimize as opt
import scipy.special as spc
import pickle
# My modules
import utilities
import utilities_field
import utilities_vrp

set_seed = True
save_fig = True
save_data = False
plot_field = False
run_flow = True
run_vrp = True

# Global settings
grid_size = 5 # larger than 50 takes long to compute on mac
cell_size = 100. # meters
vehicle_speed = 10.
num_nodes = grid_size**2
vehicle_density = 0.25
num_vehicles = int(num_nodes * vehicle_density)
num_passengers = num_vehicles
# VRP
epsilon = .001
epsilon2 = .0001
# Flow
deterministic_flow = True # choose edge leading to nearest passenger (greedy strategy)
alpha_flow = 2.0 # the higher, the more deterministic (greedy) the flow

pickle.dump(num_nodes, open('data/num_nodes.p', 'wb'))


# ---------------------------------------------------
# Load graph
graph = nx.grid_2d_graph(grid_size, grid_size, periodic=False, create_using=None)
print nx.info(graph)
print "Number of vehicles: ", num_vehicles

if set_seed: 
    np.random.seed(1234)

# Initialization
vehicle_node_ind = np.random.choice(np.arange(num_nodes), size=num_vehicles, replace=False)
passenger_node_ind = np.random.choice(np.arange(num_nodes), size=num_passengers, replace=False)
vehicle_nodes = utilities.index_to_location(graph, vehicle_node_ind)
passenger_nodes = utilities.index_to_location(graph, passenger_node_ind)

# Run vehicles on flow field
if run_flow:
    field = utilities_field.create_probability_field(graph, passenger_node_ind, alpha_flow)
    max_force = utilities_field.max_field_force(field)
    if plot_field:
        utilities_field.plot_field(field, max_force, graph, cell_size, passenger_node_ind)
    waiting_time_flow = utilities_field.run_vehicle_flow(vehicle_nodes, 
        passenger_nodes, graph, field, alpha_flow, deterministic_flow) * cell_size / vehicle_speed
    perc_flow = [np.percentile(waiting_time_flow, 50), np.percentile(waiting_time_flow, 95)]
    if save_data:
        pickle.dump(waiting_time_flow, open('data/waiting_time_flow.p','wb'))

# Run VRP
vehicle_node_ind_noisy = utilities.add_noise(graph, vehicle_node_ind, epsilon, grid_size, cell_size)
vehicle_node_ind_noisy2 = utilities.add_noise(graph, vehicle_node_ind, epsilon2, grid_size, cell_size)

if run_vrp:
    print 'Computing optimal VRP...'
    waiting_time_vrpopt = utilities_vrp.run_vrp_allocation(graph, vehicle_node_ind, passenger_node_ind) * cell_size / vehicle_speed
    print 'Computing subopt. VRP...'
    waiting_time_vrpsub = utilities_vrp.run_vrp_allocation(graph, vehicle_node_ind_noisy, passenger_node_ind) * cell_size / vehicle_speed
    print 'Computing subopt. VRP no. 2...'
    waiting_time_vrpsub2 = utilities_vrp.run_vrp_allocation(graph, vehicle_node_ind_noisy2, passenger_node_ind) * cell_size / vehicle_speed
    print 'Computing random VRP...'
    waiting_time_vrprand = utilities_vrp.run_rand_allocation(graph, vehicle_node_ind, passenger_node_ind) * cell_size / vehicle_speed

    perc_vrpopt = [np.percentile(waiting_time_vrpopt, 50), np.percentile(waiting_time_vrpopt, 95)]
    perc_vrpsub = [np.percentile(waiting_time_vrpsub, 50), np.percentile(waiting_time_vrpsub, 95)]
    perc_vrpsub2 = [np.percentile(waiting_time_vrpsub2, 50), np.percentile(waiting_time_vrpsub2, 95)]
    perc_vrprand = [np.percentile(waiting_time_vrprand, 50), np.percentile(waiting_time_vrprand, 95)]

    if save_data:
        pickle.dump(waiting_time_vrpopt, open('data/waiting_time_vrpopt.p','wb'))
        pickle.dump(waiting_time_vrpsub, open('data/waiting_time_vrpsub.p','wb'))
        pickle.dump(waiting_time_vrpsub2, open('data/waiting_time_vrpsub2.p','wb'))
        pickle.dump(waiting_time_vrprand, open('data/waiting_time_vrprand.p','wb'))



# ---------------------------------------------------
# Plot
if run_flow and run_vrp:
    max_value = np.max(np.stack([waiting_time_flow, waiting_time_vrpopt, waiting_time_vrpsub, waiting_time_vrprand]))
elif run_flow:
    max_value = np.max(np.stack([waiting_time_flow]))
elif run_vrp:
    max_value = np.max(np.stack([waiting_time_vrpopt, waiting_time_vrpsub, waiting_time_vrprand]))

def plot_waiting_time_distr(waiting_time, percentile, bins, max_value, filename, save_fig=False):
    a = 0.5
    fig = plt.figure(figsize=(6,6), frameon=False)
    hdata, hbins = np.histogram(waiting_time, bins=bins)
    hdata_max = np.max(hdata)
    plt.hist(waiting_time, bins=bins, color='blue', alpha=a)
    for i in range(len(percentile)):
        plt.plot([percentile[i], percentile[i]],[0, hdata_max], 'k--')
    plt.xlim([-1, max_value+1])
    plt.ylim([0, hdata_max])
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency')
    if save_fig:
        plt.savefig(filename, format='png', transparent=True)


num_bins = 25
bins = np.linspace(-0.5, max_value+0.5, num_bins+1)


# Flow
if run_flow:
    plot_waiting_time_distr(waiting_time_flow, perc_flow, bins, max_value, 'figures/times_flow.png', save_fig=save_fig)

if run_vrp:
    # Optimal VRP
    plot_waiting_time_distr(waiting_time_vrpopt, perc_vrpopt, bins, max_value, 'figures/times_vrpopt.png', save_fig=save_fig)

    # Subopt VRP
    plot_waiting_time_distr(waiting_time_vrpsub, perc_vrpsub, bins, max_value, 'figures/times_vrpsub.png', save_fig=save_fig)
    
    # Subopt 2 VRP
    plot_waiting_time_distr(waiting_time_vrpsub2, perc_vrpsub2, bins, max_value, 'figures/times_vrpsub2.png', save_fig=save_fig)

    # Random VRP
    plot_waiting_time_distr(waiting_time_vrprand, perc_vrprand, bins, max_value, 'figures/times_vrprand.png', save_fig=save_fig)
    

plt.show()






