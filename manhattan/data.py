import collections
import csv
import hashlib
import glob
import matplotlib.pylab as plt
from matplotlib.collections import LineCollection
import networkx as nx
import numpy as np
import os
import osmnx as ox
import msgpack
import sys
import utm
import tqdm  # Progress bar.
from scipy.spatial import cKDTree


# Installation:
# sudo apt-get install libspatialindex-dev libgeos-dev libgdal-dev
# sudo pip install osmnx
# (Possibly: pip install fiona rtree osgeo)

# Here is an awesome post:
# http://toddwschneider.com/posts/analyzing-1-1-billion-nyc-taxi-and-uber-trips-with-a-vengeance/


def GetCachedFilenamePrefix(graph):
  return hashlib.md5(str(len(graph.nodes())) + str(len(graph.edges()))).hexdigest()

def LoadMapData(cache_folder='data', use_small_graph=False, default_speed=4.18):
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
    # Add a time attribute to edges.
    for u, v, key, data in graph.edges(data=True, keys=True):
        if 'time' not in data:
          time = data['length'] / default_speed
          data['time'] = time
          data['speed'] = default_speed
        else:
          data['time'] = float(data['time'])
          data['speed'] = float(data['speed'])
        graph.add_edge(u, v, key, **data)
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
        route_lengths = dict(nx.shortest_path_length(graph, weight='time'))
        # We need to convert all defaultdict to dict before saving.
        for k, v in route_lengths.iteritems():
            route_lengths[k] = dict((m, n) for m, n in v.iteritems())
        with open(route_lengths_filename, 'wb') as fp:
            fp.write(msgpack.packb(route_lengths))
    return route_lengths

def LoadTaxiData(graph, cache_folder='data', synthetic_rides=False, must_recompute=False,
                 num_synthetic_rides=200, synthetic_ride_speed=10., max_rides=None):
    def FromLatLong(lat_long):
        x, y, _, _ = utm.from_latlon(*lat_long)
        return np.array([x, y])

    # Load taxi data.
    original_data_filename_pattern = os.path.join(cache_folder, 'yellow_tripdata_*.csv')
    binary_data_filename = os.path.join(cache_folder, 'taxi_%s%s.pickle' % ('synthetic_' if synthetic_rides else '', GetCachedFilenamePrefix(graph)))
    try:
        if must_recompute: raise IOError('Forced dummy error')
        with open(binary_data_filename, 'rb') as fp:
            print 'Loading taxi data from disk...'
            taxi_data = msgpack.unpackb(fp.read())
    except (IOError, EOFError):
      taxi_data = {}
      if not synthetic_rides:
          original_data_filename = max(glob.glob(original_data_filename_pattern))
          print 'Loading original taxi data from disk (%s)...' % original_data_filename
          pickup_xy = []
          dropoff_xy = []
          pickup_times = []
          dropoff_times = []
          count_rows = -1  # Remove header line.
          with open(original_data_filename, 'rb') as fp:
            for line in fp.xreadlines():
                count_rows += 1
          with open(original_data_filename, 'rb') as fp:
            reader = csv.reader(fp)
            header_row_index = dict((h, i) for i, h in enumerate(reader.next()))
            # Unfortunately, we have to read everything as the files are not in order.
            for row in tqdm.tqdm(reader, total=count_rows):
                pickup_lat_long = (float(row[header_row_index['pickup_latitude']]), float(row[header_row_index['pickup_longitude']]))
                dropoff_lat_long = (float(row[header_row_index['dropoff_latitude']]), float(row[header_row_index['dropoff_longitude']]))
                pickup_xy.append(FromLatLong(pickup_lat_long).tolist())
                dropoff_xy.append(FromLatLong(dropoff_lat_long).tolist())
                pickup_times.append(row[header_row_index['tpep_pickup_datetime']])
                dropoff_times.append(row[header_row_index['tpep_dropoff_datetime']])
          pickup_times = np.array(pickup_times, dtype='datetime64').astype(np.uint64).tolist()
          dropoff_times = np.array(dropoff_times, dtype='datetime64').astype(np.uint64).tolist()
          max_rides = max_rides if max_rides else len(pickup_times)
          pickup_times, dropoff_times, pickup_xy, dropoff_xy = zip(*sorted(zip(pickup_times, dropoff_times, pickup_xy, dropoff_xy))[:max_rides])
      else:
        print 'Generating synthetic taxi rides...'
        # For testing on small map, generate random trips.
        pickup_times = []
        dropoff_times = []
        pickup_xy = []
        dropoff_xy = []
        pickup_node_array = np.random.choice(graph.nodes(), size=num_synthetic_rides)
        dropoff_node_array = np.random.choice(graph.nodes(), size=num_synthetic_rides)
        last_pickup = 0.
        for node_u, node_v in zip(pickup_node_array, dropoff_node_array):
            u = np.array([graph.node[node_u]['x'], graph.node[node_u]['y']])
            v = np.array([graph.node[node_v]['x'], graph.node[node_v]['y']])
            pickup_times.append(int(last_pickup))
            last_pickup += np.random.rand() * 60.
            expected_length = np.sum(np.abs(u - v))  # Using manhattan distance.
            dropoff_times.append(int(last_pickup + 1.0 + ((np.random.rand() * 0.2 - 0.1) * expected_length + 0.9 * expected_length) / synthetic_ride_speed))  # 36 km/h.
            pickup_xy.append(u)
            dropoff_xy.append(v)
      taxi_data['pickup_time'] = np.array(pickup_times, dtype='uint64').tolist()
      taxi_data['dropoff_time'] = np.array(dropoff_times, dtype='uint64').tolist()
      taxi_data['pickup_xy'] = np.array(pickup_xy).tolist()
      taxi_data['dropoff_xy'] = np.array(dropoff_xy).tolist()
      with open(binary_data_filename, 'wb') as fp:
          fp.write(msgpack.packb(taxi_data))
    return taxi_data

def UpdateEdgeTime(graph, taxi_data, nearest_neighbor_searcher, default_speed=None, min_counts=10,
                   cache_folder='data', must_recompute=False, ignore_ride_distance=300., ignore_ride_duration=20):
    edge_times_filename = os.path.join(cache_folder, 'edge_times_%s.pickle' % GetCachedFilenamePrefix(graph))
    try:
        if must_recompute: raise IOError('Forced dummy error')
        with open(edge_times_filename, 'rb') as fp:
            print 'Loading edge times from disk...'
            edge_times = msgpack.unpackb(fp.read())
    except IOError:
        print 'Computing all routes to calculate edge time...'
        routes = nx.shortest_path(graph, weight='length')
        print 'Computing edge statistics...'
        edge_times = collections.defaultdict(lambda: collections.defaultdict(lambda: []))
        for start_time, end_time, u, v in tqdm.tqdm(zip(
                taxi_data['pickup_time'], taxi_data['dropoff_time'],
                taxi_data['pickup_xy'], taxi_data['dropoff_xy'])):
            if end_time - start_time < ignore_ride_duration: continue
            # Assume taxi took shortest route.
            u_node, du = nearest_neighbor_searcher.Search(u)
            if du > ignore_ride_distance: continue
            v_node, dv = nearest_neighbor_searcher.Search(v)
            if dv > ignore_ride_distance: continue
            route = routes[u_node][v_node]

            # Get route length.
            length = 0.
            for a, b in zip(route[:-1], route[1:]):
                length += min([data for data in graph.edge[a][b].values()], key=lambda x: x['length'])['length']

            # Assume vehicle drove at a constant speed.
            speed = length / float(end_time - start_time)

            for a, b in zip(route[:-1], route[1:]):
                length = min([data for data in graph.edge[a][b].values()], key=lambda x: x['length'])['length']
                edge_times[a][b].append(length / speed)
        # Take median time (thus removing outliers).
        edge_times = dict(edge_times)
        for u, neighbors in edge_times.iteritems():
            to_delete = []
            for v, times in neighbors.iteritems():
                if len(times) >= min_counts:
                    edge_times[u][v] = np.median(times)
                else:
                    to_delete.append(v)
            for v in to_delete:
                del neighbors[v]
            edge_times[u] = dict(neighbors)
        with open(edge_times_filename, 'wb') as fp:
            fp.write(msgpack.packb(edge_times))
    # Clip outliers (due to wrongly reported road lengths).
    all_speeds = []
    all_lengths = []
    for u, v, key, data in graph.edges(data=True, keys=True):
        if u in edge_times and v in edge_times[u]:
            all_speeds.append(data['length'] / edge_times[u][v])
            all_lengths.append(data['length'])
    all_speeds = np.array(all_speeds)
    all_lengths = np.array(all_lengths)
    mean_speed = np.mean(all_speeds)
    std_speed = np.std(all_speeds)
    print 'Average speed: %.2f +- %.2f km/h' % (mean_speed * 3.6, std_speed * 3.6)
    if default_speed is None:
        default_speed = mean_speed
    # Add a time attribute to edges.
    for u, v, key, data in graph.edges(data=True, keys=True):
        if u in edge_times and v in edge_times[u]:
            speed = np.clip(data['length'] / edge_times[u][v], mean_speed - 2. * std_speed, mean_speed + 2. * std_speed)
        else:
            speed = default_speed
        data['time'] = data['length'] / speed
        data['speed'] = speed
        graph.add_edge(u, v, key, **data)
