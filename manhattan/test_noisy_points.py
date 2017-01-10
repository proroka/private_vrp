import matplotlib.pylab as plt
from matplotlib.collections import LineCollection
import networkx as nx
import numpy as np
import os
import osmnx as ox
import pandas as pd
import pickle
import sys
import utm
from scipy.spatial import cKDTree
import scipy.special as spc
# My modules
import utilities_manhattan as um
#import ../general_graphs/utilities as util

#-------------------------------------
# Functions

# Add noise to positions
def add_noise(point_locations, nearest_neighbor_searcher, epsilon):
    # Noisy positions
    radius, theta = sample_polar_laplace(epsilon, point_locations.shape[0])
    noise_vector = polar2euclid(radius, theta)
    noisy_point_locations = point_locations + noise_vector
    # Find nearest node
    nearest_nodes, dist = nearest_neighbor_searcher.Search(noisy_point_locations)

    return nearest_nodes, noisy_point_locations

# Add noise to node indeces
def add_noise_to_ind(point_indeces, graph, nearest_neighbor_searcher, epsilon):
    # Get position from indeces
    point_locations = um.GetNodePositions(graph, point_indeces)
    # Noisy positions
    radius, theta = sample_polar_laplace(epsilon, point_locations.shape[0])
    noise_vector = polar2euclid(radius, theta)
    noisy_point_locations = point_locations + noise_vector
    # Find nearest node
    nearest_nodes, dist = nearest_neighbor_searcher.Search(noisy_point_locations)
 
    return nearest_nodes, noisy_point_locations

# 
def sample_polar_laplace(epsilon, size):
    # Sample from 2D Laplace distribution
    theta = np.random.uniform(0, 2 * np.pi, size=size)
    aux = np.random.uniform(0,1, size=size)

    # Compute -1 branch of Lambbert function
    W = spc.lambertw((aux-1)/np.e, -1)
    radius = np.real(-(1/epsilon) * (W +1))

    return radius, theta

# Convert polar coordinates to Euclidean
def polar2euclid(radius, theta):
    aux = np.empty((radius.shape[0], 2))
    aux[:,0] = np.cos(theta)
    aux[:,1] = np.sin(theta)
    return aux * np.expand_dims(radius, axis=1)




#-------------------------------------
# Generate noisy points, scatter on map

use_small_graph = True
graph, nearest_neighbor_searcher = um.LoadMapData(use_small_graph=use_small_graph)
route_lengths = um.LoadShortestPathData(graph, use_small_graph=use_small_graph)
taxi_data = um.LoadTaxiData(use_small_graph=use_small_graph)

# Plot road network.
fig, ax = ox.plot_graph(graph, show=False, close=False)

# Plot GPS location of Flatiron as reference.
flatiron_lat_long = (40.741063, -73.989701)
flatiron_xy = um.FromLatLong(flatiron_lat_long)
ax.scatter(flatiron_xy[0], flatiron_xy[1], s=30, c='orange', alpha=0.9, edgecolor='k', zorder=100)
# Get closest point on map.
flatiron_node, distance = nearest_neighbor_searcher.Search(flatiron_xy)
flatiron_node_xy = um.GetNodePosition(graph, flatiron_node)
#ax.scatter(flatiron_node_xy[0], flatiron_node_xy[1], s=40, c='y', alpha=1, edgecolor='k', zorder=100)
print 'Closest node to Flatiron is %d : %g [m] away' % (flatiron_node, distance)

# Get noisy points and indeces
epsilon = 0.01
num_samples = 100
point_locations = np.ones((num_samples,2)) * flatiron_xy
nearest_nodes, noisy_point_locations = add_noise(point_locations, nearest_neighbor_searcher, epsilon)

# Count occurences of nodes, scale size of plot point
count = dict()
for v in nearest_nodes:
    if v in count:
        count[v] += 1
    else:
        count[v] = 1
key_node, count_node = zip(*count.items())
noisy_points_size = np.array(count_node) * 100. / float(max(count_node))

# Plot noisy samples
plt.scatter(noisy_point_locations[:,0], noisy_point_locations[:,1], s=50, c='b', alpha=0.3, edgecolor='none', zorder=10)
nearest_nodes_xy = um.GetNodePositions(graph, nearest_nodes)
plt.scatter(nearest_nodes_xy[:,0], nearest_nodes_xy[:,1], color='red', s=noisy_points_size, zorder=10)

plt.show()


