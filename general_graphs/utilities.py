# Modules
import numpy as np
import networkx as nx
import scipy.optimize as opt
import scipy.special as spc


# Add noise to node positions, return node on graph closest to noisy position; clip to fit grid
# Nodes:  [(0, 1), (1, 2), ... ]

def sample_polar_laplace(epsilon, size):
    # Sample from 2D Laplace distribution
    theta = np.random.uniform(0, 2 * np.pi, size=size)
    aux = np.random.uniform(0,1, size=size)

    # Compute -1 branch of Lambbert function
    W = spc.lambertw((aux-1)/np.e, -1)
    radius = np.real(-(1/epsilon) * (W +1))

    return radius, theta

# Convert polar coordinates to Euclidean
def polar2euclid(radius, theta):
    aux = np.empty((radius.shape[0], 2))
    aux[:,0] = np.cos(theta)
    aux[:,1] = np.sin(theta)
    return aux * np.expand_dims(radius, axis=1)


# Add noise to all vehicle positions
def add_noise(graph, nodes_ind, epsilon, grid_size, cell_size):
    # Get all vehicle node locations
    node_locations = index_to_location(graph, nodes_ind) #np.array(graph.nodes())[nodes_ind]
    # Add Gaussian noise
    # noise_vector = np.random.normal(loc=0, scale=noise, size=node_locations.shape)
    # Add Laplace noise to all locations
    radius, theta = sample_polar_laplace(epsilon, node_locations.shape[0])
    noise_vector = polar2euclid(radius, theta)
    # Scale to true grid size, round to nearest node, and clip to fit grid
    node_locations_noisy = (np.around((node_locations * cell_size + noise_vector) / cell_size))
    node_locations_noisy = node_locations_noisy.astype(np.int32)
    node_locations_noisy = np.clip(node_locations_noisy, 0, grid_size-1)

    return location_to_index(graph, node_locations_noisy)


def index_to_location(indeces, index_to_pos_dict):
    # key: index, value: position of node
    #graph_dict = dict((k, v) for k, v in enumerate(graph.nodes()))
    #print graph_dict
    a = [index_to_pos_dict[i] for i in indeces]
    
    return np.array(a)


def location_to_index(graph, locations, pos_to_index_dict):
    # key: position of node, value: index
    #graph_dict = dict((v, k) for k, v in enumerate(graph.nodes()))
    
    #print "Dict of graph:\n", graph_dict
    a = [pos_to_index_dict[tuple(i)] for i in locations]

    return np.array(a)





