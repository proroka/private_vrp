import matplotlib.cm
import matplotlib.pylab as plt
from matplotlib.collections import LineCollection
import numpy as np
import osmnx as ox

# My modules
import utilities.graph as util_graph
import manhattan.data as manh_data

#-------------------------------------

use_small_graph = True
use_real_taxi_data = True
must_recompute = True

# Loads graph, uses default avg travel time fro all edges; attribut is 'time'
graph = manh_data.LoadMapData(use_small_graph=use_small_graph)
nearest_neighbor_searcher = util_graph.NearestNeighborSearcher(graph)
taxi_data = manh_data.LoadTaxiData(graph, synthetic_rides=not use_real_taxi_data, must_recompute=must_recompute,
                                   num_synthetic_rides=1000, max_rides=1000000)

# Graph data structure is modified to accound for empirical travel time (updates 'time' attribute)
manh_data.UpdateEdgeTime(graph, taxi_data, nearest_neighbor_searcher, must_recompute=must_recompute)

# Get speed attribute for plotting 
max_speed = 0.
min_speed = float('inf')
for u, v, data in graph.edges(data=True):
    max_speed = max(data['speed'], max_speed)
    min_speed = min(data['speed'], min_speed)

# Go through all edges in graph, append with corresponding color from map
cmap = matplotlib.cm.get_cmap('RdYlGn')
lines = []
route_colors = []
for u, v, data in graph.edges(data=True):
    # if it has a geometry attribute (ie, a list of line segments)
    if 'geometry' in data:
        xs, ys = data['geometry'].xy
        lines.append(list(zip(xs, ys)))
    else:
        x1 = graph.node[u]['x']
        y1 = graph.node[u]['y']
        x2 = graph.node[v]['x']
        y2 = graph.node[v]['y']
        line = [(x1, y1), (x2, y2)]
        lines.append(line)
    route_colors.append(cmap((data['speed'] - min_speed) / (max_speed - min_speed)))

# Plot road network.
fig, ax = ox.plot_graph(graph, show=False, close=False)
lc = LineCollection(lines, colors=route_colors, linewidths=3, alpha=0.5, zorder=3)
ax.add_collection(lc)
# Colorbar trick
cax = ax.imshow([[min_speed, max_speed]], vmin=min_speed, vmax=max_speed, visible=False, cmap=cmap)  # This won't show.
min_km_speed = int(np.ceil(min_speed * 3.6))
max_km_speed = int(np.floor(max_speed * 3.6))
cbar = fig.colorbar(cax, ticks=[min_km_speed / 3.6, max_km_speed / 3.6])
cbar.ax.set_yticklabels(['%d km/h' % min_km_speed, '%d km/h' % max_km_speed])
plt.show(block=False)

filename = 'figures/manhattan_speed_map_small.eps'
plt.savefig(filename, format='eps', transparent=True, frameon=False)
filename = 'figures/manhattan_speed_map_small.png'
plt.savefig(filename, format='png', transparent=True, frameon=False)


raw_input('Hit ENTER to close figure')
plt.close()
