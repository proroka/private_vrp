

import numpy as np

# Add noise to node positions, return node on graph closest to noisy position; clip to fit grid
# Nodes:  [(0, 1), (1, 2), ... ]
def add_noise(graph, nodes_ind, noise, grid_size):

    
    # Convert to array
    node_locations = np.array(graph.nodes())[nodes_ind]
    #node_locations[nodes_ind]
    noise_vector = np.random.normal(loc=0, scale=noise, size=node_locations.shape)
    node_locations_noisy = (np.around(node_locations + noise_vector))
    #node_locations_noisy.astype(int)
    node_locations_noisy.astype(np.int32)
    np.clip(node_locations_noisy, 0, grid_size)

    print "Vehicle locations, original:\n ", node_locations
    print "Vehicle locations, noisy:\n ", node_locations_noisy


    return None