import matplotlib.pylab as plt
import numpy as np
import osmnx as ox
import networkx as nx

# My modules
import utilities.graph as util_graph
import utilities.noise as util_noise
import manhattan.data as manh_data

#-------------------------------------

noise_model = 'gauss'

# Load graph
grid_size = 10
edge_length = 100.
speed = 10.

graph = util_graph.create_grid_map(grid_size=grid_size, edge_length=edge_length, default_speed=speed, std_dev=2.0)
nearest_neighbor_searcher = util_graph.NearestNeighborSearcher(graph)


# Random XY position of Point Of Interest
randx = np.random.rand() * (grid_size-1) * edge_length
randy = np.random.rand() * (grid_size-1) * edge_length
poi_xy = np.array([randx, randy])


if noise_model == 'laplace': epsilons = [] #[0.005, 0.01, 0.02, 0.05, 0.1]
elif noise_model == 'gauss': epsilons = [150.0] #[10.0, 30.0, 60.0]

print epsilons
num_samples = 100
point_locations = np.ones((num_samples, 2)) * poi_xy


# Create auxiliary graph to generate node positions
aux_graph = nx.grid_2d_graph(grid_size, grid_size, periodic=False, create_using=None)
pos = dict(zip(graph.nodes(), np.array(aux_graph.nodes())* edge_length)) # Only works for grid graph



for epsilon in epsilons:
  # Plot road network.
  fig = plt.figure(figsize=(8,8))
  ax = plt.gca()

  pos = dict(zip(graph.nodes(), np.array(aux_graph.nodes())* edge_length)) 
  nx.draw_networkx(graph, pos=pos, with_labels=False, node_size=2, node_color='lightblue')

  # Plot GPS location of POI as reference.
  ax.scatter(poi_xy[0], poi_xy[1], s=30, c='orange', alpha=0.9, edgecolor='k', zorder=100)
  
  # Get closest point on map.
  poi_node, distance = nearest_neighbor_searcher.Search(poi_xy)
  poi_node_xy = util_graph.GetNodePosition(graph, poi_node)
  ax.scatter(poi_node_xy[0], poi_node_xy[1], s=40, c='y', alpha=1, edgecolor='k', zorder=100)
  print 'Closest node to POI is %d : %g [m] away' % (poi_node, distance)

  # Get noisy points and indeces
  nearest_nodes, noisy_point_locations = util_noise.add_noise(point_locations, nearest_neighbor_searcher, epsilon, noise_model)

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
  plt.scatter(noisy_point_locations[:, 0], noisy_point_locations[:, 1], s=40, c='b', alpha=0.3, edgecolor='none', zorder=10)
  nearest_nodes_xy = util_graph.GetNodePositions(graph, key_node)
  plt.scatter(nearest_nodes_xy[:, 0], nearest_nodes_xy[:, 1], color='red', s=noisy_points_size, zorder=10)

  filename = 'figures/graph_%s_noisy_%.3f.eps' % (noise_model, epsilon)
  plt.savefig(filename, format='eps', transparent=True, frameon=False)
  plt.title('Epsilon = %g' % epsilon)


plt.show(block=False)
raw_input('Hit ENTER to close figure')
plt.close()

