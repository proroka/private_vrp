

import numpy as np

# Add noise to node positions, return node on graph closest to noisy position; clip to fit grid
# Nodes:  [(0, 1), (1, 2), ... ]
def add_noise(graph, nodes_ind, noise, grid_size):

    
    # Convert to array
    node_locations = index_to_location(graph, nodes_ind) #np.array(graph.nodes())[nodes_ind]
    # Add noise
    noise_vector = np.random.normal(loc=0, scale=noise, size=node_locations.shape)
    node_locations_noisy = (np.around(node_locations + noise_vector))
    node_locations_noisy = node_locations_noisy.astype(np.int32)
    node_locations_noisy = np.clip(node_locations_noisy, 0, grid_size-1)

    #print "Vehicle locations, original:\n ", node_locations
    #print "Vehicle locations, noisy:\n ", node_locations_noisy

    return location_to_index(graph, node_locations_noisy)


def index_to_location(graph, indeces):
    # key: index, value: position of node
    graph_dict = dict((k, v) for k, v in enumerate(graph.nodes()))
    #print graph_dict
    a = [graph_dict[i] for i in indeces]
    return np.array(a)


def location_to_index(graph, locations):
    # key: position of node, value: index
    graph_dict = dict((v, k) for k, v in enumerate(graph.nodes()))
    
    #print "Dict of graph:\n", graph_dict
    a = [graph_dict[tuple(i)] for i in locations]

    return np.array(a)