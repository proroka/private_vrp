# Standard modules
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.optimize as opt
import scipy.special as spc
import pickle

# My modules
import utilities.graph as util_graph
import utilities.noise as util_noise
import utilities.vrp as util_vrp
import utilities.probabilistic as util_prob
import utilities.plot as util_plot
import manhattan.data as manh_data

#-------------------------------------
# Global settings
vehicle_density = 0.3
passenger_density = 0.3
# Noise
epsilon = 0.02

set_seed = True

if set_seed: 
    np.random.seed(1019)

# ---------------------------------------------------
# Load small manhattan and initialize

use_small_graph = True
graph  = manh_data.LoadMapData(use_small_graph=use_small_graph)
nearest_neighbor_searcher = util_graph.NearestNeighborSearcher(graph)
route_lengths = manh_data.LoadShortestPathData(graph, use_small_graph=use_small_graph)
taxi_data = manh_data.LoadTaxiData(graph, route_lengths, use_small_graph=use_small_graph)

# Initialize 
num_nodes = len(graph.nodes())
num_vehicles = int(num_nodes * vehicle_density)
num_passengers = int(num_nodes * passenger_density)

vehicle_node_ind = np.random.choice(graph.nodes(), size=num_vehicles, replace=False)
passenger_node_ind = np.random.choice(graph.nodes(), size=num_passengers, replace=False)

# Generate noisy vehicle indeces
vehicle_node_pos = util_graph.GetNodePositions(graph, vehicle_node_ind)
vehicle_node_ind_noisy, vehicle_pos_noisy = util_noise.add_noise(vehicle_node_pos, nearest_neighbor_searcher, epsilon)


node_ind_noisy, prob, const = util_prob.compute_nearest_nodes(vehicle_pos_noisy[0,:], epsilon, nearest_neighbor_searcher, graph)

print 'Nodes indeces:', node_ind_noisy
print 'Probabilities:', prob
print 'Normalization const:', const








