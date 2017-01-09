# Standard modules
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.optimize as opt
import utilities
import collections as coll

verbose = False
plot_on = False
set_seed = True


# Global settings
grid_size = 10
cell_size = 100 # meters
num_nodes = grid_size**2
num_samples = 1000 # number of random samples to be taken
epsilon = .02

# Load graph
graph = nx.grid_2d_graph(grid_size, grid_size, periodic=False, create_using=None)

print nx.info(graph)

if plot_on:
    fig = plt.figure(figsize=(10,10))
    pos = dict(zip(graph.nodes(), np.array(graph.nodes())*cell_size)) # Only works for grid graph
    nx.draw_networkx(graph, pos=pos, with_labels=False, node_size=2, node_color='lightblue')
    plt.axis('equal')
    plt.show()


# Select 1 node
aux = 37 # choose a random node 
node_ind = np.ones(num_samples) * aux

# Generate noisy samples
node_locations = utilities.index_to_location(graph, node_ind)
# Add Laplace noise to all locations
radius, theta = utilities.sample_polar_laplace(epsilon, node_locations.shape[0])
noise_vector = utilities.polar2euclid(radius, theta)
# Noisy positions
noisy_points = node_locations * cell_size + noise_vector


# Round to nearest node, and clip to fit grid
node_locations_noisy = (np.around((noisy_points) / cell_size))
node_locations_noisy = node_locations_noisy.astype(np.int32)
node_locations_noisy = np.clip(node_locations_noisy, 0, grid_size-1)
# Noisy positions on nearest grid node
noisy_points_grid = node_locations_noisy * cell_size

# Count occurences of grid locations
count = dict()
for row in noisy_points_grid:
    p = tuple(row)
    if p in count:
        count[p] += 1
    else:
        count[p] = 1

key_node, count_node = zip(*count.items())
noisy_points_grid = np.array(key_node)
noisy_points_size = np.array(count_node) * 1000. / float(max(count_node))

# Plot graph
fig1 = plt.figure(figsize=(10,10))
pos = dict(zip(graph.nodes(), np.array(graph.nodes())*cell_size)) # Only works for grid graph
nx.draw_networkx(graph, pos=pos, with_labels=False, node_size=2, node_color='lightblue')
# Plot noisy points
plt.scatter(noisy_points[:,0], noisy_points[:,1])
plt.scatter(node_locations[0,0]*cell_size, node_locations[0,1]*cell_size, color='yellow')
plt.axis('equal')
plt.savefig('./figures/noisy_points.eps', format='eps')

# Plot noisy nodes
fig2 = plt.figure(figsize=(10,10))
nx.draw_networkx(graph, pos=pos, with_labels=False, node_size=2, node_color='lightblue')
# Plot nearest grid points
plt.scatter(noisy_points_grid[:,0], noisy_points_grid[:,1],color='red',s=noisy_points_size)
plt.axis('equal')
plt.savefig('./figures/noisy_grid_points.eps', format='eps')

plt.show()



