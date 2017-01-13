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


use_small_graph = False
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

print len(graph.nodes())
print len(graph.edges())

filename = 'figures/manhattan_map.png'
plt.savefig(filename, format='png', transparent=True, frameon=False)

plt.show(block=False)
raw_input('Hit ENTER to close figure')

plt.close()