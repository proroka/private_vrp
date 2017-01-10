# Standard modules
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.optimize as opt
import scipy.special as spc
import pickle
# My modules
import utilities
import utilities_vrp


set_seed = True
save_fig = False
save_data = False
verbose = False
plot_graph = False

# Global settings
num_nodes = 30
num_samples = 100

epsilon = .2

if set_seed: 
    np.random.seed(1234)

# ---------------------------------------------------
# Generate graph

# Generate node positions and indeces
node_pos = np.random.randint(100, size=(num_nodes,2))
node_pos_t = [tuple(r) for r in node_pos]
index_to_pos_dict = dict( (i, np.array(n)) for i, n in enumerate(node_pos_t))
pos_to_index_dict = dict( (n, i) for i, n in enumerate(node_pos_t))

# Create list of 3-tuples for edges (ID, ID, weight)
aux_graph = nx.connected_watts_strogatz_graph(num_nodes, 3, 0.6) # auxiliary graph
edges_t = [tuple(r) for r in aux_graph.edges()]
weighted_edges_t = [(u,v,np.linalg.norm(u-v)) for u,v in aux_graph.edges()]
edges_to_weight_dict = dict( ((u,v), (np.linalg.norm((u-v)))) for u, v in aux_graph.edges())

if verbose:
    print "Nodes tuple: ", node_pos_t
    print "Nodes dict: ", index_to_pos_dict
    print "Edges tuple: ", edges_t
    print "Weighted edges tuple: ", weighted_edges_t

# Create graph
graph = nx.Graph()
graph.add_nodes_from(range(len(node_pos_t)))
graph.add_weighted_edges_from(weighted_edges_t)
print nx.info(graph)

# Plot graph
if plot_graph:
    plt.axis('equal')
    nx.draw(graph, pos=index_to_pos_dict, linewidths=3)
    plt.show()



# ---------------------------------------------------
# Plot noisy samples around 1 grid point; projection

# Select 1 node
aux = np.random.randint(num_nodes) # choose a random node 
point_indeces = np.ones(num_samples) * aux

# Add noise, get indeces
noisy_point_indeces, noisy_point_locations = utilities.add_noise(point_indeces, index_to_pos_dict, epsilon)

print "Indeces: ", noisy_point_indeces

# Count occurences of grid locations
count = dict()
for v in noisy_point_indeces:
    if v in count:
        count[v] += 1
    else:
        count[v] = 1

# Map samples to graph nodes
key_node, count_node = zip(*count.items())
noisy_points_mapped = np.array(key_node)
noisy_points_mapped_pos = utilities.index_to_location(noisy_points_mapped,index_to_pos_dict)

# Plot graph
noisy_points_size = np.array(count_node) * 1000. / float(max(count_node))
fig1 = plt.figure(figsize=(10,10))
nx.draw(graph, pos=index_to_pos_dict,node_size=50, node_color='lightblue') #,node_size=2, node_color='lightblue')

# Plot noisy points
plt.scatter(noisy_point_locations[:,0], noisy_point_locations[:,1])
point = utilities.index_to_location(np.array([aux]), index_to_pos_dict)
plt.scatter(point[0,0], point[0,1], color='green', zorder=100)
plt.scatter(noisy_points_mapped_pos[:,0], noisy_points_mapped_pos[:,1],color='red',s=noisy_points_size)

plt.axis('equal')
plt.show()



