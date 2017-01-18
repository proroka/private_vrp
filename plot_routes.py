
# Standard modules
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.optimize as opt
import scipy.special as spc
import pickle
import collections
from collections import Counter
import msgpack
import osmnx as ox

# My modules
import utilities.graph as util_graph
import utilities.noise as util_noise
import utilities.vrp as util_vrp
import utilities.plot as util_plot
import utilities.probabilistic as util_prob
import manhattan.data as manh_data


use_small_graph = True
use_real_taxi_data = False
must_recompute = False

# Loads graph, uses default avg travel time fro all edges; attribut is 'time'
graph = manh_data.LoadMapData(use_small_graph=use_small_graph)
nearest_neighbor_searcher = util_graph.NearestNeighborSearcher(graph)
taxi_data = manh_data.LoadTaxiData(graph, synthetic_rides=not use_real_taxi_data, must_recompute=must_recompute,
                                   num_synthetic_rides=100, max_rides=1000000)

# Graph data structure is modified to accound for empirical travel time (updates 'time' attribute)
manh_data.UpdateEdgeTime(graph, taxi_data, nearest_neighbor_searcher, must_recompute=must_recompute)

route_lengths = manh_data.LoadShortestPathData(graph, must_recompute=False)

# Plot road network.
fig, ax = ox.plot_graph(graph, show=False, close=False)

# Plot GPS location of Flatiron as reference.
flatiron_lat_long = (40.741063, -73.989701)
flatiron_xy = util_graph.FromLatLong(flatiron_lat_long)
# Flatiron exact position 
ax.scatter(flatiron_xy[0], flatiron_xy[1], s=40, c='orange', alpha=1., edgecolor='k', zorder=4)
# Get closest point on map.
flatiron_node, distance = nearest_neighbor_searcher.Search(flatiron_xy)
flatiron_node_xy = util_graph.GetNodePosition(graph, flatiron_node)
# Flatiron node 
ax.scatter(flatiron_node_xy[0], flatiron_node_xy[1], s=40, c='b', alpha=0.3, edgecolor='none', zorder=4)
print 'Closest node to Flatiron is %d : %g [m] away' % (flatiron_node, distance)

# Generate 100 rides
num_rides = 100
vehicle_node_ind, dist = nearest_neighbor_searcher.Search(taxi_data['dropoff_xy'][:num_rides])
passenger_node_ind, dist = nearest_neighbor_searcher.Search(taxi_data['pickup_xy'][:num_rides])

# Compute optimal assignment
allocation_cost = util_vrp.get_allocation_cost(route_lengths, vehicle_node_ind, passenger_node_ind)
print 'Computing optimal VRP...'
cost, row_ind, col_ind = util_vrp.get_routing_assignment(allocation_cost)
waiting_time = allocation_cost[row_ind, col_ind]

# Plot selected rides
num_routes = 2
rand_ind = np.random.choice(range(num_rides),num_routes)
for i in rand_ind:
    destination_node = passenger_node_ind[row_ind[i]]
    origin_node = vehicle_node_ind[row_ind[i]]
    
    route = nx.shortest_path(graph, origin_node, destination_node, 'time')
    distance = route_lengths[origin_node][destination_node]
    util_graph.PlotRoute(graph, route, ax)
    print 'Distance from %s to %s: %g [m]' % (origin_node, destination_node, distance)

filename = 'figures/routes_map.eps'
plt.savefig(filename, format='eps', transparent=True, frameon=False)
filename = 'figures/routes_map.png'
plt.savefig(filename, format='png', transparent=True, frameon=False)


plt.show()




