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
import utilities.graph as util_graph
import utilities.noise as util_noise
import manhattan.data as manh_data

#-------------------------------------


use_small_graph = True
graph  = manh_data.LoadMapData(use_small_graph=use_small_graph)
nearest_neighbor_searcher = util_graph.NearestNeighborSearcher(graph)

# Plot road network.
fig, ax = ox.plot_graph(graph, show=False, close=False)

# Plot GPS location of Flatiron as reference.
flatiron_lat_long = (40.741063, -73.989701)
flatiron_xy = util_graph.FromLatLong(flatiron_lat_long)
ax.scatter(flatiron_xy[0], flatiron_xy[1], s=30, c='orange', alpha=0.9, edgecolor='k', zorder=100)
# Get closest point on map.
flatiron_node, distance = nearest_neighbor_searcher.Search(flatiron_xy)
flatiron_node_xy = util_graph.GetNodePosition(graph, flatiron_node)
#ax.scatter(flatiron_node_xy[0], flatiron_node_xy[1], s=40, c='y', alpha=1, edgecolor='k', zorder=100)
print 'Closest node to Flatiron is %d : %g [m] away' % (flatiron_node, distance)

# Get noisy points and indeces
epsilon = 0.02
num_samples = 100
point_locations = np.ones((num_samples,2)) * flatiron_xy
nearest_nodes, noisy_point_locations = util_noise.add_noise(point_locations, nearest_neighbor_searcher, epsilon)

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
nearest_nodes_xy = util_graph.GetNodePositions(graph, nearest_nodes)
plt.scatter(nearest_nodes_xy[:,0], nearest_nodes_xy[:,1], color='red', s=noisy_points_size, zorder=10)

plt.show()



