# Standard modules
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.optimize as opt
import scipy.special as spc
# My modules
import utilities
import utilities_field
import utilities_vrp

set_seed = True
save_fig = True
plot_field = False
run_flow = False

# Global settings
grid_size = 50
cell_size = 10 # meters
num_nodes = grid_size**2
epsilon = .01
epsilon2 = .001
vehicle_density = 0.35
num_vehicles = int(num_nodes * vehicle_density)
num_passengers = num_vehicles

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
    alpha = 2.0 # the higher, the more deterministic (greedy) the flow
    deterministic = False # choose edge leading to nearest passenger (greedy strategy)
    field = utilities_field.create_probability_field(graph, passenger_node_ind, alpha)
    max_force = utilities_field.max_field_force(field)
    if plot_field:
        utilities_field.plot_field(field, max_force, graph, cell_size, passenger_node_ind)
    waiting_time_flow = utilities_field.run_vehicle_flow(vehicle_nodes, passenger_nodes, graph, field, alpha, deterministic) 


# Run VRP
vehicle_node_ind_noisy = utilities.add_noise(graph, vehicle_node_ind, epsilon, grid_size, cell_size)
vehicle_node_ind_noisy2 = utilities.add_noise(graph, vehicle_node_ind, epsilon2, grid_size, cell_size)

print 'Computing optimal VRP...'
waiting_time_vrpopt = utilities_vrp.run_vrp_allocation(graph, vehicle_node_ind, passenger_node_ind)
print 'Computing subopt. VRP...'
waiting_time_vrpsub = utilities_vrp.run_vrp_allocation(graph, vehicle_node_ind_noisy, passenger_node_ind)
print 'Computing subopt. VRP no. 2...'
waiting_time_vrpsub2 = utilities_vrp.run_vrp_allocation(graph, vehicle_node_ind_noisy2, passenger_node_ind)
print 'Computing random VRP...'
waiting_time_vrprand = utilities_vrp.run_rand_allocation(graph, vehicle_node_ind, passenger_node_ind)

perc95_vrpopt = np.percentile(waiting_time_vrpopt, 95)
perc95_vrpsub = np.percentile(waiting_time_vrpsub, 95)
perc95_vrpsub2 = np.percentile(waiting_time_vrpsub2, 95)
perc95_vrprand = np.percentile(waiting_time_vrprand, 95)

if run_flow:
    perc95_flow = np.percentile(waiting_time_flow, 95)


# ---------------------------------------------------
# Plot
if run_flow:
    max_value = np.max(np.stack([waiting_time_flow, waiting_time_vrpopt, waiting_time_vrpsub, waiting_time_vrprand]))
else:
    max_value = np.max(np.stack([waiting_time_vrpopt, waiting_time_vrpsub, waiting_time_vrprand]))
num_bins = 60
bins = np.linspace(-0.5, max_value+0.5, num_bins+1)
a = 0.5

# Flow
if run_flow:
    plt.figure()
    hdata, hbins = np.histogram(waiting_time_flow, bins=bins)
    hdata_max = np.max(hdata)
    plt.hist(waiting_time_flow, bins=bins, color='cyan', alpha=a)
    plt.plot([perc95_flow, perc95_flow],[0, hdata_max], 'k--')
    plt.xlim([-1, max_value+1])
    plt.ylim([0, hdata_max])
    if save_fig:
        plt.savefig('figures/times_flow.eps',format='eps')

# Optimal VRP
plt.figure()
hdata, hbins = np.histogram(waiting_time_vrpopt, bins=bins)
hdata_max = np.max(hdata)
plt.hist(waiting_time_vrpopt, bins=bins, color='blue', alpha=a)
plt.plot([perc95_vrpopt, perc95_vrpopt],[0, hdata_max], 'k--')
plt.xlim([-1, max_value+1])
plt.ylim([0, hdata_max])
if save_fig:
    plt.savefig('figures/times_vrpopt.eps',format='eps')

# Subopt VRP
plt.figure()
hdata, hbins = np.histogram(waiting_time_vrpsub, bins=bins)
hdata_max = np.max(hdata)
plt.hist(waiting_time_vrpsub, bins=bins, color='green', alpha=a)
plt.plot([perc95_vrpsub, perc95_vrpsub],[0, hdata_max], 'k--')
plt.xlim([-1, max_value+1])
plt.ylim([0, hdata_max])
if save_fig:
    plt.savefig('figures/times_vrpsub.eps',format='eps')

# Subopt 2 VRP
plt.figure()
hdata, hbins = np.histogram(waiting_time_vrpsub2, bins=bins)
hdata_max = np.max(hdata)
plt.hist(waiting_time_vrpsub2, bins=bins, color='green', alpha=a)
plt.plot([perc95_vrpsub2, perc95_vrpsub2],[0, hdata_max], 'k--')
plt.xlim([-1, max_value+1])
plt.ylim([0, hdata_max])
if save_fig:
    plt.savefig('figures/times_vrpsub2.eps',format='eps')

# Random VRP
plt.figure()
hdata, hbins = np.histogram(waiting_time_vrprand, bins=bins)
hdata_max = np.max(hdata)
plt.hist(waiting_time_vrprand, bins=bins, color='red', alpha=a)
plt.plot([perc95_vrprand, perc95_vrprand],[0, hdata_max], 'k--')
plt.xlim([-1, max_value+1])
plt.ylim([0, hdata_max])
if save_fig:
    plt.savefig('figures/times_vrprand.eps',format='eps')

plt.show()






