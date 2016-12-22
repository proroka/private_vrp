

import numpy as np

# Add noise to node positions, return node on graph closest to noisy position; clip to fit grid
# Nodes:  [(0, 1), (1, 2), ... ]
def add_noise(graph, nodes_ind, noise, grid_size):

    
    # Convert to array
    node_locations = np.array(graph.nodes()[nodes_ind])
    noise_vector = np.random.normal(loc=0, scale=noise, size=node_locations.shape)
    node_locations_noisy = np.around(node_locations + noise_vector)

    print "Original: ", node_locations
    print "Noisy:    ", node_locations_noisy

    noisy_nodes = nodes
    return noisy_nodes