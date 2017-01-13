import hashlib
import matplotlib.pylab as plt
from matplotlib.collections import LineCollection
import networkx as nx
import numpy as np
import os
import osmnx as ox
import pandas as pd
import msgpack
import sys
import utm
from scipy.spatial import cKDTree


# Installation:
# sudo apt-get install libspatialindex-dev libgeos-dev libgdal-dev
# sudo pip install osmnx
# (Possibly: pip install fiona rtree osgeo)

# Here is an awesome post:
# http://toddwschneider.com/posts/analyzing-1-1-billion-nyc-taxi-and-uber-trips-with-a-vengeance/

# Average vehicle speed in Manhattan
AVG_SPEED = 4.18 # m/s

def GetCachedFilenamePrefix(graph):
  return hashlib.md5(str(len(graph.nodes())) + str(len(graph.edges()))).hexdigest()

def LoadMapData(cache_folder='data', use_small_graph=False):
    # Load Manhattan from disk if possible. Otherwise, download.
    graph_filename = ('small_' if use_small_graph else '') + 'manhattan.graphml'
    try:
        graph = ox.load_graphml(graph_filename, folder=cache_folder)
        print 'Loading map from disk...'
    except IOError:
        print 'Generating map...'
        if use_small_graph:
            # Get map around the Flatiron.
            bbox = ox.bbox_from_point((40.741063, -73.989701), distance=500)
            north, south, east, west = bbox
            graph = ox.graph_from_bbox(north, south, east, west, network_type='drive')
        else:
            graph = ox.graph_from_place('Manhattan, New York, USA', network_type='drive')
        print 'Saving map to disk...'
        graph = max(nx.strongly_connected_component_subgraphs(graph), key=len)
        graph = ox.project_graph(graph)
        ox.save_graphml(graph, filename=graph_filename, folder=cache_folder)
    return graph

def LoadShortestPathData(graph, cache_folder='data', must_recompute=False):
    # Load shortest pre-computed path lengths.
    route_lengths_filename = os.path.join(cache_folder, 'route_lengths_%s.pickle' % GetCachedFilenamePrefix(graph))

    try:
        if must_recompute: raise IOError('Forced dummy error')
        with open(route_lengths_filename, 'rb') as fp:
            print 'Loading shortest paths from disk...'
            route_lengths = msgpack.unpackb(fp.read())
    except (IOError, EOFError):
        print 'Computing all shortest path lengths and saving to disk'
        route_lengths = dict(nx.shortest_path_length(graph, weight='length'))
        # We need to convert all defaultdict to dict before saving.
        for k, v in route_lengths.iteritems():
            route_lengths[k] = dict((m, n / AVG_SPEED) for m, n in v.iteritems())
        with open(route_lengths_filename, 'wb') as fp:
            fp.write(msgpack.packb(route_lengths))
    return route_lengths

def LoadTaxiData(graph, route_lengths, cache_folder='data', synthetic_rides=False, must_recompute=False):
    # Load taxi data.
    original_data_filename = os.path.join(cache_folder, 'yellow_tripdata_2016-05.csv')
    binary_data_filename = os.path.join(cache_folder, 'taxi_%s%s.pickle' % ('synthetic_' if synthetic_rides else '', GetCachedFilenamePrefix(graph)))
    try:
        if must_recompute: raise IOError('Forced dummy error')
        with open(binary_data_filename, 'rb') as fp:
            print 'Loading taxi data from disk...'
            taxi_data = msgpack.unpackb(fp.read())
    except (IOError, EOFError):
      print 'Loading original taxi data from disk...'
      taxi_data = {}
      if not synthetic_rides:
          taxi_data_orig = pd.read_csv(original_data_filename)
          pickup_xy = []
          dropoff_xy = []
          pickup_times = []
          dropoff_times = []
          for _, row in taxi_data_orig.iterrows():
              pickup_lat_long = (row.pickup_latitude, row.pickup_longitude)
              dropoff_lat_long = (row.dropoff_latitude, row.dropoff_longitude)
              pickup_xy.append(FromLatLong(pickup_lat_long))
              dropoff_xy.append(FromLatLong(dropoff_lat_long))
              pickup_times.append(row.tpep_pickup_datetime)
              dropoff_times.append(row.tpep_dropoff_datetime)
          pickup_times = np.array(pickup_times, dtype='datetime64').astype(np.uint64)
          dropoff_times = np.array(dropoff_times, dtype='datetime64').astype(np.uint64)
      else:
        # For testing on small map, generate random trips.
        pickup_times = []
        dropoff_times = []
        pickup_xy = []
        dropoff_xy = []
        pickup_node_array = np.random.choice(graph.nodes(), size=200)
        dropoff_node_array = np.random.choice(graph.nodes(), size=200)
        last_pickup = 0.
        for node_u, node_v in zip(pickup_node_array, dropoff_node_array):
            if (node_u not in route_lengths or node_v not in route_lengths[node_u] or
                route_lengths[node_u][node_v] > 100000):
                continue
            u = np.array([graph.node[node_u]['x'], graph.node[node_u]['y']])
            v = np.array([graph.node[node_v]['x'], graph.node[node_v]['y']])
            pickup_times.append(int(last_pickup))
            last_pickup += np.random.rand() * 60
            dropoff_times.append(int(last_pickup + route_lengths[node_u][node_v] / 10.))  # About 36 km/h.
            pickup_xy.append(u)
            dropoff_xy.append(v)
      taxi_data['pickup_time'] = np.array(pickup_times, dtype='uint64')
      taxi_data['dropoff_time'] = np.array(dropoff_times, dtype='uint64')
      taxi_data['pickup_xy'] = np.array(pickup_xy)
      taxi_data['dropoff_xy'] = np.array(dropoff_xy)
      with open(binary_data_filename, 'wb') as fp:
          fp.write(msgpack.packb(taxi_data))
    return taxi_data
