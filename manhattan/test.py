import networkx as nx
import osmnx as ox

# Installation:
# sudo apt-get install libspatialindex-dev libgeos-dev libgdal-dev
# sudo pip install osmnx
# (Possibly: pip install fiona rtree osgeo)

# Load Manhattan from disk if possible. Otherwise, download.
try:
  graph = ox.load_graphml('manhattan.graphml', folder='data')
  print 'Loading from disk...'
except IOError:
  graph = ox.graph_from_place('Manhattan, New York, USA', network_type='drive')
  # graph = ox.project_graph(graph)
  print 'Saving to disk...'
  ox.save_graphml(graph, filename='manhattan.graphml', folder='data')

# Using http://www.findlatitudeandlongitude.com/ for LatLong.
# Get intersection closest to the flatiron building:
flatiron_lat_long = (40.741063, -73.989701)
flatiron_node = ox.get_nearest_node(graph, flatiron_lat_long)

# Get intersection closest to Chelsea Market.
google_lat_long = (40.742152, -74.005151)
google_node = ox.get_nearest_node(graph, google_lat_long)

# Get shortest path.
route = nx.shortest_path(graph, flatiron_node, google_node)

fig, ax = ox.plot_graph_route(graph, route)
