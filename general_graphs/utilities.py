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
def add_noise(point_indeces, index_to_pos_dict, epsilon):
  
    point_locations = index_to_location(point_indeces, index_to_pos_dict)
    node_pos = index_to_location(range(len(index_to_pos_dict)), index_to_pos_dict)

    radius, theta = sample_polar_laplace(epsilon, point_locations.shape[0])
    noise_vector = polar2euclid(radius, theta)
    # Noisy positions
    noisy_point_locations = point_locations + noise_vector

    # Round to nearest node, and clip to fit grid
    nearest = np.zeros(noisy_point_locations.shape[0],dtype=np.int32)
    for i in range(noisy_point_locations.shape[0]):
        min_dist = 100000000.
        for j in range(node_pos.shape[0]):
            dist_ij = np.linalg.norm((noisy_point_locations[i,:]-node_pos[j,:]))
            if dist_ij < min_dist:
                min_dist = dist_ij
                nearest[i] = j
    
    return nearest, noisy_point_locations


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





