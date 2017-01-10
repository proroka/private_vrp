# Standard modules
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.optimize as opt
import scipy.special as spc
import pickle
# My modules
import utilities
import utilities_vrp


set_seed = True
save_fig = False
save_data = False
verbose = False
plot_graph = False

# Global settings
num_nodes = 30
vehicle_density = 0.2
passenger_density = 0.2

num_vehicles = int(num_nodes * vehicle_density)
num_passengers = int(num_nodes * passenger_density)
vehicle_speed = 10.

# VRP
epsilon = .001
epsilon2 = .0001

if set_seed: 
    np.random.seed(1234)

# ---------------------------------------------------
# Generate graph

# Generate node positions and edges
node_pos = np.random.randint(100, size=(num_nodes,2))
node_pos_t = [tuple(r) for r in node_pos]
index_to_pos_dict = dict( (i, np.array(n)) for i, n in enumerate(node_pos_t))

# Create list of 3-tuples for edges (ID, ID, weight)
aux_graph = nx.connected_watts_strogatz_graph(num_nodes, 3, 0.6) # auxiliary graph
edges_t = [tuple(r) for r in aux_graph.edges()]
weighted_edges_t = [(u,v,np.linalg.norm(u-v)) for u,v in aux_graph.edges()]
edges_to_weight_dict = dict( ((u,v), (np.linalg.norm((u-v)))) for u, v in aux_graph.edges())

if verbose:
    print "Nodes tuple: ", node_pos_t
    print "Nodes dict: ", index_to_pos_dict
    print "Edges tuple: ", edges_t
    print "Weighted edges tuple: ", weighted_edges_t

# Create graph
graph = nx.Graph()
graph.add_nodes_from(range(len(node_pos_t)))
graph.add_weighted_edges_from(weighted_edges_t)

print "Nodes", graph.nodes()
print "Edges", graph.edges()
print edges_to_weight_dict

# Plot graph
if plot_graph:
    plt.axis('equal')
    nx.draw(graph, pos=index_to_pos_dict, linewidths=3)
    plt.show()

print nx.info(graph)
print "Number of vehicles: ", num_vehicles
print "Number of passengers: ", num_passengers


# ---------------------------------------------------

# Initialization
vehicle_node_ind = np.random.choice(np.arange(num_nodes), size=num_vehicles, replace=False)
passenger_node_ind = np.random.choice(np.arange(num_nodes), size=num_passengers, replace=False)

# Compute noisy vehicle indeces
vehicle_node_ind_noisy, vehicle_node_pos_noisy = utilities.add_noise(vehicle_node_ind, index_to_pos_dict, epsilon)

# Run VRP
print 'Computing optimal VRP...'
waiting_time_vrpopt = utilities_vrp.run_vrp_allocation(graph, vehicle_node_ind, passenger_node_ind)

print 'Computing subopt. VRP...'
waiting_time_vrpsub = utilities_vrp.run_vrp_allocation(graph, vehicle_node_ind_noisy, passenger_node_ind)

# print 'Computing subopt. VRP no. 2...'
# waiting_time_vrpsub2 = utilities_vrp.run_vrp_allocation(graph, vehicle_node_ind_noisy2, passenger_node_ind) * cell_size / vehicle_speed
# print 'Computing random VRP...'
# waiting_time_vrprand = utilities_vrp.run_rand_allocation(graph, vehicle_node_ind, passenger_node_ind) * cell_size / vehicle_speed

perc_vrpopt = [np.percentile(waiting_time_vrpopt, 50), np.percentile(waiting_time_vrpopt, 95)]
perc_vrpsub = [np.percentile(waiting_time_vrpsub, 50), np.percentile(waiting_time_vrpsub, 95)]
# perc_vrpsub2 = [np.percentile(waiting_time_vrpsub2, 50), np.percentile(waiting_time_vrpsub2, 95)]
# perc_vrprand = [np.percentile(waiting_time_vrprand, 50), np.percentile(waiting_time_vrprand, 95)]

# if save_data:
#     pickle.dump(waiting_time_vrpopt, open('data/waiting_time_vrpopt.p','wb'))
#     pickle.dump(waiting_time_vrpsub, open('data/waiting_time_vrpsub.p','wb'))
#     pickle.dump(waiting_time_vrpsub2, open('data/waiting_time_vrpsub2.p','wb'))
#     pickle.dump(waiting_time_vrprand, open('data/waiting_time_vrprand.p','wb'))



# # ---------------------------------------------------
# # Plot


def plot_waiting_time_distr(waiting_time, percentile, bins, filename, save_fig=False, max_value=None):
    a = 0.5
    fig = plt.figure(figsize=(6,6), frameon=False)
    hdata, hbins = np.histogram(waiting_time, bins=bins)
    hdata_max = np.max(hdata)
    plt.hist(waiting_time, bins=bins, color='blue', alpha=a)
    for i in range(len(percentile)):
        plt.plot([percentile[i], percentile[i]],[0, hdata_max], 'k--')
    if not max_value:
        max_value = np.max(waiting_time)
    plt.xlim([-1, max_value+1])
    plt.ylim([0, hdata_max])
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency')
    if save_fig:
        plt.savefig(filename, format='png', transparent=True)


#max_value = np.max(np.stack([waiting_time_vrpopt, waiting_time_vrpsub, waiting_time_vrprand]))
max_value = np.max(waiting_time_vrpopt)
num_bins = 25
bins = np.linspace(-0.5, max_value+0.5, num_bins+1)

# # Optimal VRP
plot_waiting_time_distr(waiting_time_vrpopt, perc_vrpopt, bins, 'figures/times_vrpopt.png', save_fig=save_fig, max_value=max_value)

# Subopt VRP
plot_waiting_time_distr(waiting_time_vrpsub, perc_vrpsub, bins, 'figures/times_vrpsub.png', save_fig=save_fig, max_value=max_value)

# # Subopt 2 VRP
# plot_waiting_time_distr(waiting_time_vrpsub2, perc_vrpsub2, bins, max_value, 'figures/times_vrpsub2.png', save_fig=save_fig)

# # Random VRP
# plot_waiting_time_distr(waiting_time_vrprand, perc_vrprand, bins, max_value, 'figures/times_vrprand.png', save_fig=save_fig)
    

plt.show()






