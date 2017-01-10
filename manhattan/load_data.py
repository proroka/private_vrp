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

# Installation:
# sudo apt-get install libspatialindex-dev libgeos-dev libgdal-dev
# sudo pip install osmnx
# (Possibly: pip install fiona rtree osgeo)

# Here is an awesome post:
# http://toddwschneider.com/posts/analyzing-1-1-billion-nyc-taxi-and-uber-trips-with-a-vengeance/

def LoadMapData(cache_folder='data', use_small_graph=False):
  # Load Manhattan from disk if possible. Otherwise, download.
  graph_filename = ('small_' if use_small_graph else '') + 'manhattan.graphml'
  try:
    graph = ox.load_graphml(graph_filename, folder=cache_folder)
    print 'Loading map from disk...'
  except IOError:
    if use_small_graph:
      # Get map around the Flatiron.
      bbox = ox.bbox_from_point((40.741063, -73.989701), distance=500)
      north, south, east, west = bbox
      graph = ox.graph_from_bbox(north, south, east, west, network_type='drive')
    else:
      graph = ox.graph_from_place('Manhattan, New York, USA', network_type='drive')
    # graph = ox.project_graph(graph)
    print 'Saving map to disk...'
    ox.save_graphml(graph, filename=graph_filename, folder=cache_folder)
  searcher = NearestNeighborSearcher(graph)
  return graph, searcher

def LoadShortestPathData(graph, cache_folder='data', use_small_graph=False):
  # Load shortest pre-computed path lengths.
  route_lengths_filename = os.path.join(cache_folder, ('small_' if use_small_graph else '') + 'manhattan_route_lengths.pickle')
  try:
    with open(route_lengths_filename, 'rb') as fp:
      print 'Loading shortest paths from disk...'
      route_lengths = pickle.load(fp)
  except (IOError, EOFError):
    print 'Computing all shortest path lengths and saving to disk'
    route_lengths = dict(nx.shortest_path_length(graph, weight='length'))
    # We need to convert all defaultdict to dict before saving.
    for k, v in route_lengths.iteritems():
      route_lengths[k] = dict(v)
    with open(route_lengths_filename, 'wb') as fp:
      pickle.dump(route_lengths, fp)
  return route_lengths

def LoadTaxiData(cache_folder='data', use_small_graph=False):
  # Load taxi data.
  original_data_filename = os.path.join(cache_folder, ('small_' if use_small_graph else '') + 'yellow_tripdata_2016-05.csv')
  binary_data_filename = os.path.join(cache_folder, ('small_' if use_small_graph else '') + 'manhattan_taxi.pickle')
  try:
    with open(binary_data_filename, 'rb') as fp:
      print 'Loading taxi data from disk...'
      taxi_data = pickle.load(fp)
  except (IOError, EOFError):
    print 'Loading original taxi data from disk...'
    taxi_data_orig = pd.read_csv(original_data_filename)
    taxi_data = {}
    if not use_small_graph:
      pickup_lat_long = np.vstack([taxi_data_orig.pickup_latitude, taxi_data_orig.pickup_longitude]).T
      dropoff_lat_long = np.vstack([taxi_data_orig.dropoff_latitude, taxi_data_orig.dropoff_longitude]).T
      pickup_times = np.array(taxi_data_orig.tpep_pickup_datetime, dtype='datetime64').astype(np.uint64)
      dropoff_times = np.array(taxi_data_orig.tpep_pickup_datetime, dtype='datetime64').astype(np.uint64)
    else:
      # For testing on small map, generate random trips.
      pickup_times = []
      dropoff_times = []
      pickup_lat_long = []
      dropoff_lat_long = []
      pickup_node_array = np.random.choice(graph.nodes(), size=200)
      dropoff_node_array = np.random.choice(graph.nodes(), size=200)
      last_pickup = 0.
      for node_u, node_v in zip(pickup_node_array, dropoff_node_array):
        if (node_u not in route_lengths or node_v not in route_lengths[node_u] or 
            route_lengths[node_u][node_v] > 100000): 
          continue
        u = np.array([graph.node[node_u]['y'], graph.node[node_u]['x']])
        v = np.array([graph.node[node_v]['y'], graph.node[node_v]['x']])
        pickup_times.append(int(last_pickup))
        last_pickup += np.random.rand() * 60
        dropoff_times.append(int(last_pickup + route_lengths[node_u][node_v] / 10.))  # About 36 km/h.
        pickup_lat_long.append(u)
        dropoff_lat_long.append(v)
    taxi_data['pickup_time'] = np.array(pickup_times, dtype='uint64')
    taxi_data['dropoff_time'] = np.array(dropoff_times, dtype='uint64')
    taxi_data['pickup_lat_long'] = np.array(pickup_lat_long)
    taxi_data['dropoff_lat_long'] = np.array(dropoff_lat_long)
    with open(binary_data_filename, 'wb') as fp:
      pickle.dump(taxi_data, fp)
  return taxi_data


class NearestNeighborSearcher(object):

  def __init__(self, graph):
    points = []
    indices = []
    for k, v in graph.node.iteritems():
      x, y, _,  _ = utm.from_latlon(v['y'], v['x'])
      points.append([x, y])
      indices.append(k)
    self.indices = indices
    self.kdtree = cKDTree(points, 10)

  def Search(self, lat_long):
    x, y, _, _ = utm.from_latlon(*lat_long)
    distances, indices = self.kdtree.query([[x, y]])
    return self.indices[indices[0]], distances[0]


# Modified from https://github.com/gboeing/osmnx/blob/master/osmnx/plot.py.
def PlotRoute(G, route, ax):
  # the origin and destination nodes are the first and last nodes in the route
  origin_node = route[0]
  destination_node = route[-1]
  origin_destination_lats = (G.node[origin_node]['y'], G.node[destination_node]['y'])
  origin_destination_lons = (G.node[origin_node]['x'], G.node[destination_node]['x'])
  ax.scatter(origin_destination_lons, origin_destination_lats, s=100,
             c='r', alpha=0.3, edgecolor='none', zorder=4)
  edge_nodes = list(zip(route[:-1], route[1:]))
  lines = []
  for u, v in edge_nodes:
    data = min([data for data in G.edge[u][v].values()], key=lambda x: x['length'])
    if 'geometry' in data:
      xs, ys = data['geometry'].xy
      lines.append(list(zip(xs, ys)))
    else:
      x1 = G.node[u]['x']
      y1 = G.node[u]['y']
      x2 = G.node[v]['x']
      y2 = G.node[v]['y']
      line = [(x1, y1), (x2, y2)]
      lines.append(line)
  lc = LineCollection(lines, colors='r', linewidths=4, alpha=0.3, zorder=3)
  ax.add_collection(lc)


if __name__ == '__main__':
  use_small_graph = True
  graph, nearest_neighbor_searcher = LoadMapData(use_small_graph=use_small_graph)
  route_lengths = LoadShortestPathData(graph, use_small_graph=use_small_graph)
  taxi_data = LoadTaxiData(use_small_graph=use_small_graph)
  # Plot first 5 taxi routes.
  fig, ax = ox.plot_graph(graph, show=False, close=False)
  for i in range(5):
    pickup_node, d1 = nearest_neighbor_searcher.Search(taxi_data['pickup_lat_long'][i])
    dropoff_node, d2 = nearest_neighbor_searcher.Search(taxi_data['dropoff_lat_long'][i])
    # Check that route exists.
    if (d1 > 300. or d2 > 300. or pickup_node not in route_lengths or
        dropoff_node not in route_lengths[pickup_node] or
        route_lengths[pickup_node][dropoff_node] > 100000): continue
    route = nx.shortest_path(graph, pickup_node, dropoff_node, 'length')
    distance = route_lengths[pickup_node][dropoff_node]
    PlotRoute(graph, route, ax)
    print 'Distance from %s to %s: %g [m]' % (pickup_node, dropoff_node, distance)
  plt.show()
